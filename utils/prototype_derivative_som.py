"""Prototype-based derivative SOM calibration.

This module keeps the existing derivative SOM untouched and adds a variant
that softly groups local d1/d2 windows into learnable derivative prototypes.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.curvature_features import (
    _EPS,
    _validate_timeseries,
    compute_derivative_heatmaps,
)


class PrototypeDerivativeSOM(nn.Module):
    """Calibrate forecasts with a soft codebook over local d1/d2 patterns."""

    def __init__(
        self,
        num_channels: int,
        seq_len: int,
        pred_len: int,
        num_prototypes: int = 8,
        prototype_window: int = 12,
        prototype_temperature: float = 1.0,
        prototype_stride: Optional[int] = None,
        prototype_mode: str = "cosine",
        prototype_distance: str = "l2",
        prototype_latent_dim: Optional[int] = None,
        commitment_weight: float = 0.25,
        ema_codebook_update: bool = False,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        correction_scale: float = 1.0,
        gate_init: float = 0.1,
        num_knots: int = 8,
        gamma_bound: float = 0.5,
        beta_bound: float = 0.25,
        gamma_init: float = -4.0,
        beta_init: float = 0.0,
        grid_width: float = 0.25,
        learnable_grid: bool = True,
        moving_avg_kernel: int = 3,
        d2_weight: float = 1.0,
        sharpness_temperature: float = 1.0,
        adaptive_grid: bool = True,
        adaptive_grid_sharpness: float = 1.0,
        adaptive_grid_min_scale: float = 0.5,
        adaptive_grid_max_scale: float = 1.0,
        horizon_decay_floor: float = 1.0,
        horizon_decay_power: float = 1.0,
        init_scale: float = 0.02,
        eps: float = _EPS,
        **_,
    ):
        super().__init__()
        if num_channels < 1:
            raise ValueError("num_channels must be positive")
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        if pred_len < 1:
            raise ValueError("pred_len must be positive")
        if num_prototypes < 1:
            raise ValueError("num_prototypes must be positive")
        if prototype_window < 1:
            raise ValueError("prototype_window must be positive")
        if prototype_stride is None:
            prototype_stride = prototype_window
        if prototype_stride < 1:
            raise ValueError("prototype_stride must be positive")
        if prototype_temperature <= 0:
            raise ValueError("prototype_temperature must be positive")
        if prototype_mode not in ("cosine", "vq_vae"):
            raise ValueError("prototype_mode must be either cosine or vq_vae")
        if prototype_distance not in ("l2", "cosine"):
            raise ValueError("prototype_distance must be either l2 or cosine")
        if prototype_latent_dim is not None and prototype_latent_dim < 1:
            raise ValueError("prototype_latent_dim must be positive when provided")
        if commitment_weight < 0:
            raise ValueError("commitment_weight must be non-negative")
        if ema_decay < 0 or ema_decay >= 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if ema_eps <= 0:
            raise ValueError("ema_eps must be positive")
        if gate_init <= 0 or gate_init >= 1:
            raise ValueError("gate_init must be in (0, 1)")
        if num_knots < 2:
            raise ValueError("num_knots must be >= 2")
        if grid_width <= 0:
            raise ValueError("grid_width must be positive")
        if moving_avg_kernel < 1 or moving_avg_kernel % 2 == 0:
            raise ValueError("moving_avg_kernel must be a positive odd integer")
        if sharpness_temperature <= 0:
            raise ValueError("sharpness_temperature must be positive")
        if adaptive_grid_min_scale <= 0 or adaptive_grid_max_scale <= 0:
            raise ValueError("adaptive grid scales must be positive")
        if adaptive_grid_min_scale > adaptive_grid_max_scale:
            raise ValueError("adaptive_grid_min_scale must be <= adaptive_grid_max_scale")
        if horizon_decay_floor <= 0 or horizon_decay_floor > 1:
            raise ValueError("horizon_decay_floor must be in (0, 1]")
        if horizon_decay_power <= 0:
            raise ValueError("horizon_decay_power must be positive")

        self.num_channels = int(num_channels)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.num_prototypes = int(num_prototypes)
        self.prototype_window = int(prototype_window)
        self.prototype_stride = int(prototype_stride)
        self.prototype_temperature = float(prototype_temperature)
        self.prototype_mode = prototype_mode
        self.prototype_distance = prototype_distance
        self.commitment_weight = float(commitment_weight)
        self.ema_codebook_update = bool(ema_codebook_update)
        self.ema_decay = float(ema_decay)
        self.ema_eps = float(ema_eps)
        self.correction_scale = float(correction_scale)
        self.num_knots = int(num_knots)
        self.gamma_bound = float(gamma_bound)
        self.beta_bound = float(beta_bound)
        self.moving_avg_kernel = int(moving_avg_kernel)
        self.d2_weight = float(d2_weight)
        self.sharpness_temperature = float(sharpness_temperature)
        self.adaptive_grid = bool(adaptive_grid)
        self.adaptive_grid_sharpness = float(adaptive_grid_sharpness)
        self.adaptive_grid_min_scale = float(adaptive_grid_min_scale)
        self.adaptive_grid_max_scale = float(adaptive_grid_max_scale)
        self.horizon_decay_floor = float(horizon_decay_floor)
        self.horizon_decay_power = float(horizon_decay_power)
        self.horizon_decay_ref_len = 720
        self.horizon_decay_length_power = 1.5
        self.eps = eps

        feature_dim = 2 * self.prototype_window
        latent_dim = int(prototype_latent_dim or feature_dim)
        self.prototype_latent_dim = latent_dim
        self.token_encoder = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.token_decoder = nn.Linear(latent_dim, 2)
        self.token_reconstructor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, feature_dim),
        )
        self.prototypes = nn.Parameter(init_scale * torch.randn(self.num_prototypes, latent_dim))
        if self.ema_codebook_update:
            self.prototypes.requires_grad_(False)
            self.register_buffer("ema_cluster_size", torch.zeros(self.num_prototypes))
            self.register_buffer("ema_prototypes", self.prototypes.detach().clone())
        self.derivative_to_pred = nn.Conv1d(in_channels=self.seq_len, out_channels=self.pred_len, kernel_size=1)

        centers = torch.linspace(0.0, 1.0, self.num_knots)
        self.register_buffer("knot_centers", centers)
        width = torch.full((self.num_knots,), float(grid_width))
        if learnable_grid:
            self.log_grid_width = nn.Parameter(torch.log(width))
        else:
            self.register_buffer("log_grid_width", torch.log(width))
        self.gamma_logits = nn.Parameter(torch.full((self.num_knots, self.num_channels), float(gamma_init)))
        self.beta_logits = nn.Parameter(torch.full((self.num_knots, self.num_channels), float(beta_init)))

        gate_logit = torch.logit(torch.tensor(float(gate_init)))
        self.gate_logit = nn.Parameter(gate_logit)

    def _check_inputs(self, y_hat: torch.Tensor, x_enc: torch.Tensor) -> None:
        _validate_timeseries(y_hat, name="y_hat")
        _validate_timeseries(x_enc, name="x_enc")
        if y_hat.size(-1) != self.num_channels:
            raise ValueError(
                f"y_hat channel mismatch: expected {self.num_channels}, got {y_hat.size(-1)}"
            )
        if x_enc.size(-1) != self.num_channels:
            raise ValueError(
                f"x_enc channel mismatch: expected {self.num_channels}, got {x_enc.size(-1)}"
            )
        if x_enc.size(1) != self.seq_len:
            raise ValueError(
                f"x_enc length mismatch: expected {self.seq_len}, got {x_enc.size(1)}"
            )

    def _local_derivative_tokens(self, x_enc: torch.Tensor) -> torch.Tensor:
        heatmap = compute_derivative_heatmaps(x_enc)
        b, length, c = x_enc.shape
        series = x_enc.permute(0, 2, 1).reshape(b * c, 1, length)

        left = (self.prototype_window - 1) // 2
        right = self.prototype_window - 1 - left
        if left > 0 or right > 0:
            series = F.pad(series, (left, right), mode="replicate")

        windows = series.unfold(dimension=2, size=self.prototype_window, step=self.prototype_stride).squeeze(1)
        d1 = torch.zeros_like(windows)
        d1[..., 1:] = windows[..., 1:] - windows[..., :-1]
        d2 = torch.zeros_like(windows)
        d2[..., 1:] = d1[..., 1:] - d1[..., :-1]

        token_len = windows.size(1)
        tokens = torch.cat([d1, d2], dim=-1).reshape(b, c, token_len, -1)
        scale = tokens.detach().abs().mean(dim=-1, keepdim=True).clamp_min(self.eps)
        return torch.tanh(tokens / scale), heatmap

    def _prototype_distances(self, latent: torch.Tensor) -> torch.Tensor:
        prototypes = self.prototypes.view(1, 1, 1, self.num_prototypes, -1)
        if self.prototype_distance == "cosine":
            latent = F.normalize(latent, p=2, dim=-1)
            prototypes = F.normalize(prototypes, p=2, dim=-1)
        return (latent.unsqueeze(-2) - prototypes).pow(2).sum(dim=-1)

    def _ema_update_codebook(self, assignments: torch.Tensor, latent: torch.Tensor) -> None:
        if not self.ema_codebook_update:
            return

        flat_assignments = assignments.reshape(-1, self.num_prototypes)
        flat_latent = latent.reshape(-1, self.prototype_latent_dim)
        cluster_size = flat_assignments.sum(dim=0)
        proto_sum = flat_assignments.t() @ flat_latent

        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1.0 - self.ema_decay)
        self.ema_prototypes.mul_(self.ema_decay).add_(proto_sum, alpha=1.0 - self.ema_decay)

        total_count = self.ema_cluster_size.sum()
        normalized_size = (
            (self.ema_cluster_size + self.ema_eps)
            / (total_count + self.num_prototypes * self.ema_eps)
            * total_count.clamp_min(self.ema_eps)
        )
        updated = self.ema_prototypes / normalized_size.unsqueeze(-1).clamp_min(self.ema_eps)
        self.prototypes.data.copy_(updated)

    def _prototype_assignments(self, tokens: torch.Tensor):
        latent = self.token_encoder(tokens)
        distances = self._prototype_distances(latent)

        if self.prototype_mode == "vq_vae":
            indices = distances.argmin(dim=-1)
            assignments = F.one_hot(indices, num_classes=self.num_prototypes).to(dtype=latent.dtype)
            quantized = torch.matmul(assignments, self.prototypes)
            if self.ema_codebook_update:
                if self.training:
                    self._ema_update_codebook(assignments.detach(), latent.detach())
                codebook_loss = latent.new_zeros(())
                commitment_loss = F.mse_loss(latent, quantized.detach())
                vq_loss = self.commitment_weight * commitment_loss
            else:
                codebook_loss = F.mse_loss(quantized, latent.detach())
                commitment_loss = F.mse_loss(latent, quantized.detach())
                vq_loss = codebook_loss + self.commitment_weight * commitment_loss
            proto_context = latent + (quantized - latent).detach()
            return assignments, proto_context, distances, codebook_loss, commitment_loss, vq_loss, latent

        assignments = torch.softmax(-distances / self.prototype_temperature, dim=-1)
        proto_context = torch.matmul(assignments, self.prototypes)
        codebook_loss = tokens.new_zeros(())
        commitment_loss = tokens.new_zeros(())
        vq_loss = tokens.new_zeros(())
        return assignments, proto_context, distances, codebook_loss, commitment_loss, vq_loss, latent

    def _prototype_heatmap(self, proto_context: torch.Tensor) -> torch.Tensor:
        return self.token_decoder(proto_context)

    def _reconstruct_tokens(self, proto_context: torch.Tensor) -> torch.Tensor:
        return self.token_reconstructor(proto_context)

    @torch.no_grad()
    def encode_derivative_latents(self, x_enc: torch.Tensor) -> torch.Tensor:
        tokens, _ = self._local_derivative_tokens(x_enc)
        latent = self.token_encoder(tokens)
        return latent.reshape(-1, self.prototype_latent_dim)

    @torch.no_grad()
    def initialize_prototypes_kmeans(
        self,
        latents: torch.Tensor,
        num_iters: int = 50,
    ):
        if latents.dim() != 2 or latents.size(-1) != self.prototype_latent_dim:
            raise ValueError(
                f"latents must have shape [N, {self.prototype_latent_dim}], got {tuple(latents.shape)}"
            )
        latents = latents.detach().to(device=self.prototypes.device, dtype=self.prototypes.dtype)
        finite = torch.isfinite(latents).all(dim=-1)
        latents = latents[finite]
        if latents.numel() == 0:
            return {"initialized": False, "reason": "no finite latents"}

        n = latents.size(0)
        k = self.num_prototypes
        if n >= k:
            init_indices = torch.randperm(n, device=latents.device)[:k]
        else:
            init_indices = torch.randint(0, n, (k,), device=latents.device)
        centers = latents[init_indices].clone()

        num_iters = max(1, int(num_iters))
        for _ in range(num_iters):
            distances = (latents.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(dim=-1)
            labels = distances.argmin(dim=1)
            new_centers = centers.clone()
            for idx in range(k):
                mask = labels == idx
                if mask.any():
                    new_centers[idx] = latents[mask].mean(dim=0)
                else:
                    replacement = torch.randint(0, n, (), device=latents.device)
                    new_centers[idx] = latents[replacement]
            shift = (new_centers - centers).pow(2).sum(dim=-1).sqrt().mean()
            centers = new_centers
            if shift.item() < 1e-6:
                break

        self.prototypes.copy_(centers)
        distances = (latents.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(dim=-1)
        labels = distances.argmin(dim=1)
        usage = torch.bincount(labels, minlength=k).to(dtype=latents.dtype) / float(n)
        return {
            "initialized": True,
            "num_latents": int(n),
            "num_iters": num_iters,
            "usage_min": float(usage.min().item()),
            "usage_max": float(usage.max().item()),
        }

    def _project_derivatives_to_prediction(self, prototype_heatmap: torch.Tensor):
        b, c, length, derivative_dim = prototype_heatmap.shape
        if length == self.seq_len:
            aligned_heatmap = prototype_heatmap
        else:
            heatmap_t = prototype_heatmap.permute(0, 1, 3, 2).reshape(b * c, derivative_dim, length)
            heatmap_t = F.interpolate(heatmap_t, size=self.seq_len, mode="linear", align_corners=False)
            aligned_heatmap = heatmap_t.reshape(b, c, derivative_dim, self.seq_len).permute(0, 1, 3, 2)
            length = self.seq_len
        projected = self.derivative_to_pred(aligned_heatmap.reshape(b * c, length, derivative_dim))
        projected = projected.reshape(b, c, self.pred_len, derivative_dim).permute(0, 2, 1, 3)
        return projected[..., 0], projected[..., 1]

    def _temporal_moving_average(self, x: torch.Tensor) -> torch.Tensor:
        if self.moving_avg_kernel == 1:
            return x
        padding = self.moving_avg_kernel // 2
        x_t = x.permute(0, 2, 1)
        x_t = F.pad(x_t, (padding, padding), mode="replicate")
        trend = F.avg_pool1d(x_t, kernel_size=self.moving_avg_kernel, stride=1)
        return trend.permute(0, 2, 1)

    def _sharpness_score(self, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        d1_scale = d1.abs().mean(dim=1, keepdim=True).detach().clamp_min(self.eps)
        d2_scale = d2.abs().mean(dim=1, keepdim=True).detach().clamp_min(self.eps)
        d1_norm = d1.abs() / d1_scale
        d2_norm = d2.abs() / d2_scale
        score = torch.sqrt(d1_norm.pow(2) + self.d2_weight * d2_norm.pow(2) + self.eps)
        return 1.0 - torch.exp(-score / self.sharpness_temperature)

    def _knot_basis(self, sharpness: torch.Tensor):
        centers = self.knot_centers.view(1, 1, 1, self.num_knots)
        base_widths = self.log_grid_width.exp().clamp_min(self.eps).view(1, 1, 1, self.num_knots)
        widths = base_widths
        if self.adaptive_grid:
            width_scale = 1.0 / (1.0 + self.adaptive_grid_sharpness * sharpness.unsqueeze(-1))
            width_scale = width_scale.clamp(self.adaptive_grid_min_scale, self.adaptive_grid_max_scale)
            widths = (base_widths * width_scale).clamp_min(self.eps)
        basis = F.relu(1.0 - (sharpness.unsqueeze(-1) - centers).abs() / widths)
        return basis / basis.sum(dim=-1, keepdim=True).clamp_min(self.eps), widths

    def _project_params(self, basis: torch.Tensor):
        gamma_table = self.gamma_bound * torch.sigmoid(self.gamma_logits)
        beta_table = self.beta_bound * torch.tanh(self.beta_logits)
        gamma = (basis * gamma_table.t().view(1, 1, self.num_channels, self.num_knots)).sum(dim=-1)
        beta = (basis * beta_table.t().view(1, 1, self.num_channels, self.num_knots)).sum(dim=-1)
        return gamma, beta

    def _horizon_decay(self, y_hat: torch.Tensor) -> torch.Tensor:
        horizon_len = y_hat.size(1)
        horizon_scale = max(float(horizon_len) / float(self.horizon_decay_ref_len), 1e-6)
        effective_power = float(self.horizon_decay_power) * (horizon_scale ** float(self.horizon_decay_length_power))
        return torch.linspace(
            1.0,
            self.horizon_decay_floor,
            horizon_len,
            device=y_hat.device,
            dtype=y_hat.dtype,
        ).pow(effective_power).view(1, -1, 1)

    def forward(
        self,
        y_hat: torch.Tensor,
        x_enc: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        self._check_inputs(y_hat, x_enc)
        tokens, heatmap = self._local_derivative_tokens(x_enc)
        assignments, proto_context, distances, codebook_loss, commitment_loss, vq_loss, latent_tokens = self._prototype_assignments(tokens)
        reconstructed_tokens = self._reconstruct_tokens(proto_context)
        reconstruction_loss = F.mse_loss(reconstructed_tokens, tokens)

        prototype_heatmap = self._prototype_heatmap(proto_context)
        proto_d1, proto_d2 = self._project_derivatives_to_prediction(prototype_heatmap)
        sharpness = self._sharpness_score(proto_d1, proto_d2)
        basis, widths = self._knot_basis(sharpness)
        gamma, beta = self._project_params(basis)

        trend = self._temporal_moving_average(y_hat)
        local_deviation = y_hat - trend
        gate = torch.sigmoid(self.gate_logit)
        correction = gate * (gamma * local_deviation + beta * proto_d2)
        correction = self.correction_scale * correction * self._horizon_decay(y_hat)
        y = y_hat + correction

        if not return_aux:
            return y

        aux = {
            "heatmap": heatmap,
            "tokens": tokens,
            "latent_tokens": latent_tokens,
            "prototypes": self.prototypes,
            "prototype_assignments": assignments,
            "prototype_context": proto_context,
            "prototype_distances": distances,
            "prototype_usage": assignments.mean(dim=(0, 1, 2)),
            "prototype_heatmap": prototype_heatmap,
            "prototype_stride": self.prototype_stride,
            "prototype_mode": self.prototype_mode,
            "prototype_distance": self.prototype_distance,
            "ema_codebook_update": self.ema_codebook_update,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "vq_loss": vq_loss,
            "reconstructed_tokens": reconstructed_tokens,
            "reconstruction_loss": reconstruction_loss,
            "proto_d1": proto_d1,
            "proto_d2": proto_d2,
            "sharpness": sharpness,
            "knot_basis": basis,
            "adaptive_widths": widths,
            "gamma": gamma,
            "beta": beta,
            "local_deviation": local_deviation,
            "correction": correction,
            "gate": gate.detach(),
            "context": proto_context,
        }
        return y, aux


class PrototypeDerivativeSOMCalibration(nn.Module):
    """Wrapper with the same call shape as post-hoc calibration modules."""

    def __init__(self, **kwargs):
        super().__init__()
        self.calibrator = PrototypeDerivativeSOM(**kwargs)

    def encode_derivative_latents(self, x_context: torch.Tensor) -> torch.Tensor:
        return self.calibrator.encode_derivative_latents(x_context)

    def initialize_prototypes_kmeans(self, latents: torch.Tensor, num_iters: int = 50):
        return self.calibrator.initialize_prototypes_kmeans(latents, num_iters=num_iters)

    def forward(self, y_hat: torch.Tensor, x_context: Optional[torch.Tensor] = None, return_params: bool = False):
        if x_context is None:
            if return_params:
                return y_hat, {"context": None, "correction": None}
            return y_hat
        return self.calibrator(y_hat, x_context, return_aux=return_params)


class Model(PrototypeDerivativeSOMCalibration):
    """Compatibility alias for code paths that expect a Model class."""

    pass
