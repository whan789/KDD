import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def finite_differences(x):
    if x.dim() != 3:
        raise ValueError("Expected input shape [B, L, D].")

    d1 = torch.zeros_like(x)
    d1[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]

    d2 = torch.zeros_like(x)
    d2[:, 1:, :] = d1[:, 1:, :] - d1[:, :-1, :]
    return d1, d2


class KANResidualHead(nn.Module):
    """
    Additive feature-wise spline head in the KAN spirit.

    Each scalar feature is passed through its own learnable univariate spline,
    then the transformed feature values are summed into one residual logit.
    """

    def __init__(self, in_dim, num_knots=8, grid_min=-1.0, grid_max=1.0):
        super(KANResidualHead, self).__init__()
        if in_dim < 1:
            raise ValueError("in_dim must be >= 1.")
        if num_knots < 2:
            raise ValueError("num_knots must be >= 2.")
        if grid_min >= grid_max:
            raise ValueError("grid_min must be smaller than grid_max.")

        self.in_dim = in_dim
        self.num_knots = num_knots
        centers = torch.linspace(float(grid_min), float(grid_max), num_knots)
        self.register_buffer("centers", centers)
        self.basis_width = float(grid_max - grid_min) / float(num_knots - 1)

        self.spline_coeff = nn.Parameter(torch.zeros(in_dim, num_knots))
        self.base_weight = nn.Parameter(torch.zeros(in_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if x.size(-1) != self.in_dim:
            raise ValueError(f"Expected last dimension {self.in_dim}, got {x.size(-1)}.")

        centers = self.centers.view(*([1] * x.dim()), self.num_knots)
        basis = F.relu(1.0 - (x.unsqueeze(-1) - centers).abs() / self.basis_width)
        basis = basis / basis.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        coeff = self.spline_coeff.view(*([1] * (x.dim() - 1)), self.in_dim, self.num_knots)
        spline_terms = (basis * coeff).sum(dim=-1)
        linear_terms = self.base_weight.view(*([1] * (x.dim() - 1)), self.in_dim) * x
        return (spline_terms + linear_terms).sum(dim=-1, keepdim=True) + self.bias


class SignedSplineFluctuationCalibrator(nn.Module):
    """
    PIR-style hierarchical residual calibrator.

    Stage 1 retrieves similar train-set contexts and adds a gated global target
    prototype to the backbone forecast. Stage 2 applies a spline residual head to
    refine local sharpness on top of the retrieval-scaled forecast.
    """

    def __init__(
        self,
        num_knots=8,
        use_mlp_head=True,
        residual_head_type="kan",
        mlp_hidden_dim=64,
        mlp_num_layers=2,
        mlp_dropout=0.0,
        mlp_kernel_size=5,
        mlp_width_scale_bound=1.0,
        horizon_decay_floor=1.0,
        horizon_decay_power=1.0,
        retrieval_topk=10,
        retrieval_temperature=1.0,
        retrieval_beta_init=0.1,
        use_dynamics_gate=False,
        dynamics_gate_init=0.5,
        dynamics_gate_floor=0.0,
        d2_weight=1.0,
        sharpness_temperature=1.0,
        eps=1e-6,
        **legacy_kwargs,
    ):
        super(SignedSplineFluctuationCalibrator, self).__init__()

        if num_knots < 2:
            raise ValueError("num_knots must be >= 2.")
        if residual_head_type not in ("kan", "mlp", "conv"):
            raise ValueError("residual_head_type must be one of kan, mlp, or conv.")
        if mlp_num_layers < 1:
            raise ValueError("mlp_num_layers must be >= 1.")
        if mlp_hidden_dim < 1:
            raise ValueError("mlp_hidden_dim must be >= 1.")
        if mlp_kernel_size < 1 or mlp_kernel_size % 2 == 0:
            raise ValueError("mlp_kernel_size must be a positive odd integer.")
        if mlp_width_scale_bound <= 0:
            raise ValueError("mlp_width_scale_bound must be positive.")
        if horizon_decay_floor <= 0 or horizon_decay_floor > 1:
            raise ValueError("horizon_decay_floor must be in (0, 1].")
        if horizon_decay_power <= 0:
            raise ValueError("horizon_decay_power must be positive.")
        if retrieval_topk < 1:
            raise ValueError("retrieval_topk must be >= 1.")
        if retrieval_temperature <= 0:
            raise ValueError("retrieval_temperature must be positive.")
        if retrieval_beta_init <= 0 or retrieval_beta_init >= 1:
            raise ValueError("retrieval_beta_init must be in (0, 1).")
        if dynamics_gate_init <= 0 or dynamics_gate_init >= 1:
            raise ValueError("dynamics_gate_init must be in (0, 1).")
        if dynamics_gate_floor < 0 or dynamics_gate_floor >= 1:
            raise ValueError("dynamics_gate_floor must be in [0, 1).")
        if d2_weight < 0:
            raise ValueError("d2_weight must be >= 0.")
        if sharpness_temperature <= 0:
            raise ValueError("sharpness_temperature must be positive.")

        self.num_knots = num_knots
        self.use_mlp_head = use_mlp_head
        self.residual_head_type = residual_head_type
        self.mlp_kernel_size = mlp_kernel_size
        self.mlp_residual_bound = mlp_width_scale_bound
        self.horizon_decay_floor = horizon_decay_floor
        self.horizon_decay_power = horizon_decay_power
        self.horizon_decay_ref_len = 720
        self.horizon_decay_length_power = 1.5
        self.retrieval_topk = retrieval_topk
        self.retrieval_temperature = retrieval_temperature
        self.use_dynamics_gate = use_dynamics_gate
        self.dynamics_gate_floor = dynamics_gate_floor
        self.d2_weight = d2_weight
        self.sharpness_temperature = sharpness_temperature
        self.eps = eps

        beta_logit = math.log(retrieval_beta_init / (1.0 - retrieval_beta_init))
        self.retrieval_beta_bias = nn.Parameter(torch.tensor(beta_logit, dtype=torch.float32))
        self.retrieval_beta_sim_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        gate_logit = math.log(dynamics_gate_init / (1.0 - dynamics_gate_init))
        self.dynamics_gate_bias = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))
        self.dynamics_gate_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.register_buffer("memory_keys", torch.empty(0), persistent=False)
        self.register_buffer("memory_values", torch.empty(0), persistent=False)
        self.memory_seq_len = None
        self.memory_channels = None

        self.stage2_in_dim = 5
        if self.residual_head_type == "kan":
            self.kan_head = KANResidualHead(self.stage2_in_dim, num_knots=num_knots)
            self.mlp_head = None
        else:
            mlp_layers = []
            in_dim = self.stage2_in_dim
            if self.residual_head_type == "conv":
                padding = mlp_kernel_size // 2
                for _ in range(max(mlp_num_layers - 1, 0)):
                    mlp_layers.append(nn.Conv1d(in_dim, mlp_hidden_dim, kernel_size=mlp_kernel_size, padding=padding))
                    mlp_layers.append(nn.GELU())
                    if mlp_dropout > 0:
                        mlp_layers.append(nn.Dropout(mlp_dropout))
                    in_dim = mlp_hidden_dim
                mlp_layers.append(nn.Conv1d(in_dim, 1, kernel_size=mlp_kernel_size, padding=padding))
            else:
                for _ in range(max(mlp_num_layers - 1, 0)):
                    mlp_layers.append(nn.Linear(in_dim, mlp_hidden_dim))
                    mlp_layers.append(nn.GELU())
                    if mlp_dropout > 0:
                        mlp_layers.append(nn.Dropout(mlp_dropout))
                    in_dim = mlp_hidden_dim
                mlp_layers.append(nn.Linear(in_dim, 1))
            self.mlp_head = nn.Sequential(*mlp_layers)
            self.kan_head = None

    def _resize_time_length(self, x, target_len):
        if x.size(1) == target_len:
            return x
        x_t = x.permute(0, 2, 1)
        x_t = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
        return x_t.permute(0, 2, 1)

    def _align_context_channels(self, x, target_channels):
        if target_channels is None or x.size(-1) == target_channels:
            return x
        if target_channels == 1:
            return x[:, :, -1:]
        if x.size(-1) > target_channels:
            return x[:, :, -target_channels:]
        raise ValueError(
            "Input context has fewer channels than the prediction: "
            f"{x.size(-1)} vs {target_channels}."
        )

    def _temporal_moving_average(self, x, kernel_size=3):
        if kernel_size == 1:
            return x
        padding = kernel_size // 2
        x_t = x.permute(0, 2, 1)
        x_t = F.pad(x_t, (padding, padding), mode="replicate")
        trend = F.avg_pool1d(x_t, kernel_size=kernel_size, stride=1)
        return trend.permute(0, 2, 1)

    def _normalize_residual_features(self, feats):
        scale = feats.detach().abs().mean(dim=1, keepdim=True).clamp_min(self.eps)
        return torch.tanh(feats / scale)

    def _normalize_series(self, x, means=None, stdev=None):
        x = x.float()
        if means is None:
            means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        if stdev is None:
            stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return centered / stdev.clamp_min(self.eps), means, stdev

    def _prepare_residual_context(self, y_hat, x_context):
        if x_context is None or x_context.dim() != 3:
            return torch.zeros_like(y_hat)

        x_tail = x_context[:, -y_hat.size(1):, :]
        x_tail = self._align_context_channels(x_tail, y_hat.size(-1))
        return self._resize_time_length(x_tail, y_hat.size(1))

    def _prepare_retrieval_context(self, y_hat, x_context):
        if x_context is None or x_context.dim() != 3:
            return None
        x_source = self._align_context_channels(x_context, y_hat.size(-1))
        if self.memory_seq_len is not None:
            x_source = self._resize_time_length(x_source, self.memory_seq_len)
        return x_source

    @torch.no_grad()
    def set_retrieval_memory(self, x_memory, y_memory):
        if x_memory is None or y_memory is None:
            self.memory_keys = torch.empty(0, device=self.memory_keys.device)
            self.memory_values = torch.empty(0, device=self.memory_values.device)
            self.memory_seq_len = None
            self.memory_channels = None
            return
        if x_memory.dim() != 3 or y_memory.dim() != 3:
            raise ValueError("Retrieval memory expects x_memory and y_memory with shape [N, L, C].")
        if x_memory.size(0) != y_memory.size(0):
            raise ValueError("x_memory and y_memory must have the same number of instances.")

        target_channels = y_memory.size(-1)
        x_memory = self._align_context_channels(x_memory, target_channels)
        device = self.memory_keys.device if self.memory_keys.numel() > 0 else y_memory.device
        x_norm, means, stdev = self._normalize_series(x_memory)
        y_norm = (y_memory.float() - means) / stdev.clamp_min(self.eps)
        keys = x_norm.to(device=device, dtype=torch.float32)
        values = y_norm.to(device=device, dtype=torch.float32)
        self.memory_keys = keys
        self.memory_values = values
        self.memory_seq_len = x_memory.size(1)
        self.memory_channels = target_channels

    def _global_retrieval(self, y_hat, x_context):
        if self.memory_keys.numel() == 0 or self.memory_values.numel() == 0:
            zeros = torch.zeros_like(y_hat)
            beta = torch.zeros(y_hat.size(0), 1, 1, device=y_hat.device, dtype=y_hat.dtype)
            sim = torch.zeros_like(beta)
            return zeros, beta, sim

        x_retrieval = self._prepare_retrieval_context(y_hat, x_context)
        if x_retrieval is None:
            zeros = torch.zeros_like(y_hat)
            beta = torch.zeros(y_hat.size(0), 1, 1, device=y_hat.device, dtype=y_hat.dtype)
            sim = torch.zeros_like(beta)
            return zeros, beta, sim

        keys = self.memory_keys.to(device=y_hat.device, dtype=torch.float32)
        values = self.memory_values.to(device=y_hat.device, dtype=torch.float32)
        query, means, stdev = self._normalize_series(x_retrieval)
        query = query.to(device=y_hat.device, dtype=torch.float32)

        q_norm = F.normalize(query.permute(2, 0, 1), p=2, dim=-1)
        k_norm = F.normalize(keys.permute(2, 0, 1), p=2, dim=-1)
        sims = torch.matmul(q_norm, k_norm.permute(0, 2, 1))
        topk = min(self.retrieval_topk, sims.size(-1))
        topv, topi = torch.topk(sims, k=topk, dim=-1)
        weights = F.softmax(topv / float(self.retrieval_temperature), dim=-1)

        retrieved_channels = []
        for channel_idx in range(query.size(-1)):
            values_ch = values[:, :, channel_idx]
            indices_ch = topi[channel_idx]
            retrieved_ch = values_ch[indices_ch]
            weighted_ch = (weights[channel_idx].unsqueeze(-1) * retrieved_ch).sum(dim=1)
            retrieved_channels.append(weighted_ch.unsqueeze(-1))
        y_global_norm = torch.cat(retrieved_channels, dim=-1).to(dtype=y_hat.dtype)
        y_global = y_global_norm * stdev.to(dtype=y_hat.dtype) + means.to(dtype=y_hat.dtype)
        y_global = self._resize_time_length(y_global, y_hat.size(1))
        y_global = self._align_context_channels(y_global, y_hat.size(-1))

        sim_score = topv.permute(1, 0, 2).mean(dim=(1, 2), keepdim=True).to(dtype=y_hat.dtype)
        beta = torch.sigmoid(
            self.retrieval_beta_bias.to(dtype=y_hat.dtype)
            + self.retrieval_beta_sim_weight.to(dtype=y_hat.dtype) * sim_score
        )
        return y_global, beta, sim_score

    def _sharpness_score(self, d1, d2):
        d1_scale = d1.abs().mean(dim=1, keepdim=True).detach() + self.eps
        d2_scale = d2.abs().mean(dim=1, keepdim=True).detach() + self.eps

        d1_norm = d1.abs() / d1_scale
        d2_norm = d2.abs() / d2_scale
        score = torch.sqrt(d1_norm.pow(2) + self.d2_weight * d2_norm.pow(2) + self.eps)
        temperature = max(float(self.sharpness_temperature), self.eps)
        return 1.0 - torch.exp(-score / temperature)

    def _compute_dynamics_gate(self, sharpness):
        gate = torch.sigmoid(self.dynamics_gate_scale * sharpness + self.dynamics_gate_bias)
        if self.dynamics_gate_floor > 0.0:
            gate = self.dynamics_gate_floor + (1.0 - self.dynamics_gate_floor) * gate
        return gate

    def _predict_stage2_residual(self, y_scaled, y_hat, x_context, y_global, beta):
        x_tail = self._prepare_residual_context(y_hat, x_context)
        x_trend = self._temporal_moving_average(x_tail)
        x_sharp = x_tail - x_trend
        global_delta = beta * y_global
        feats = torch.stack([
            y_scaled,
            global_delta,
            y_global,
            x_tail,
            x_sharp,
        ], dim=-1)
        feats = self._normalize_residual_features(feats)
        b, l, c, f = feats.shape
        if self.residual_head_type == "kan":
            residual_logits = self.kan_head(feats.reshape(-1, f)).reshape(b, l, c)
        elif self.residual_head_type == "conv":
            conv_in = feats.permute(0, 2, 3, 1).reshape(b * c, f, l)
            residual_logits = self.mlp_head(conv_in).reshape(b, c, l).permute(0, 2, 1)
        else:
            residual_logits = self.mlp_head(feats.reshape(-1, f)).reshape(b, l, c)

        residual_scale = (
            global_delta.detach().abs().mean(dim=1, keepdim=True)
            + x_sharp.detach().abs().mean(dim=1, keepdim=True)
            + y_scaled.detach().abs().mean(dim=1, keepdim=True)
        ).clamp_min(self.eps)
        residual = self.mlp_residual_bound * torch.tanh(residual_logits) * residual_scale
        return residual, x_tail, x_sharp, global_delta

    def forward(self, y_hat, x_context=None, return_params=False):
        """
        Args:
            y_hat: Backbone prediction, shape [B, L, D].
            return_params: If True, also return auxiliary tensors.

        Returns:
            calibrated: Calibrated prediction, shape [B, L, D].
            aux: Optional dictionary for analysis/debugging.
        """
        if y_hat.dim() != 3:
            raise ValueError("Expected y_hat shape [B, L, D].")

        y_global, beta, retrieval_similarity = self._global_retrieval(y_hat, x_context)
        y_scaled = y_hat + beta * y_global

        stage2_residual = None
        x_tail = None
        x_sharp = None
        global_delta = beta * y_global
        if self.use_mlp_head:
            stage2_residual, x_tail, x_sharp, global_delta = self._predict_stage2_residual(
                y_scaled, y_hat, x_context, y_global, beta
            )
            correction = stage2_residual
        else:
            correction = torch.zeros_like(y_hat)

        dynamics_gate = None
        y_d1, y_d2 = finite_differences(y_scaled)
        y_sharpness = self._sharpness_score(y_d1, y_d2)
        if self.use_dynamics_gate:
            dynamics_gate = self._compute_dynamics_gate(y_sharpness)
            correction = correction * dynamics_gate

        horizon_len = y_hat.size(1)
        horizon_scale = max(float(horizon_len) / float(self.horizon_decay_ref_len), 1e-6)
        effective_power = float(self.horizon_decay_power) * (horizon_scale ** float(self.horizon_decay_length_power))
        horizon_decay = torch.linspace(
            1.0,
            float(self.horizon_decay_floor),
            horizon_len,
            device=y_hat.device,
            dtype=y_hat.dtype,
        ).pow(effective_power).view(1, -1, 1)
        correction = correction * horizon_decay
        calibrated = y_scaled + correction

        if not return_params:
            return calibrated

        aux = {
            "y_scaled": y_scaled,
            "y_global": y_global,
            "retrieval_beta": beta,
            "retrieval_similarity": retrieval_similarity,
            "global_delta": global_delta,
            "stage2_residual": stage2_residual,
            "correction": correction,
            "y_d1": y_d1,
            "y_d2": y_d2,
            "y_sharpness": y_sharpness,
            "dynamics_gate": dynamics_gate,
            "x_context_aligned": x_tail,
            "x_context_sharp": x_sharp,
            "horizon_decay": horizon_decay,
            "mlp_residual": stage2_residual,
            "kan_residual": stage2_residual if self.residual_head_type == "kan" else None,
        }
        return calibrated, aux


class PostHocSignedSplineCalibration(nn.Module):
    """
    Thin wrapper name for plug-in use after any forecasting backbone.
    """

    def __init__(self, **kwargs):
        super(PostHocSignedSplineCalibration, self).__init__()
        self.calibrator = SignedSplineFluctuationCalibrator(**kwargs)

    def set_retrieval_memory(self, x_memory, y_memory):
        self.calibrator.set_retrieval_memory(x_memory, y_memory)

    def forward(self, y_hat, x_context=None, return_params=False):
        return self.calibrator(y_hat, x_context=x_context, return_params=return_params)


class Model(PostHocSignedSplineCalibration):
    """Compatibility alias for code paths that expect a Model class."""

    pass
