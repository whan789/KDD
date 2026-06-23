import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.StandardNorm import Normalize


class VectorSplineBank(nn.Module):
    def __init__(
        self,
        feature_dim,
        n_knots=8,
        init_range=0.3,
        max_range=5.0,
        learnable_knot_offsets=False,
        knot_offset_scale=1.0,
    ):
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        if n_knots < 2:
            raise ValueError("n_knots must be >= 2.")
        if init_range <= 0:
            raise ValueError("init_range must be positive.")
        if max_range <= 0:
            raise ValueError("max_range must be positive.")
        if knot_offset_scale < 0.0:
            raise ValueError("knot_offset_scale must be non-negative.")

        self.feature_dim = int(feature_dim)
        self.n_knots = int(n_knots)
        self.max_range = float(max_range)
        self.learnable_knot_offsets = bool(learnable_knot_offsets)
        self.knot_offset_scale = float(knot_offset_scale)

        base_knots = torch.linspace(-1.0, 1.0, self.n_knots)
        self.register_buffer("base_knots", base_knots)
        self.range_param = nn.Parameter(torch.ones(self.feature_dim) * float(init_range))
        if self.learnable_knot_offsets:
            self.knot_offsets = nn.Parameter(torch.zeros(self.feature_dim, self.n_knots - 1))
        else:
            self.register_parameter("knot_offsets", None)
        self.weight = nn.Parameter(torch.zeros(self.feature_dim, self.n_knots))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError("Expected x shape [N, feature_dim].")
        _, feature_dim = x.shape
        if feature_dim != self.feature_dim:
            raise ValueError(f"Feature dim mismatch: expected {self.feature_dim}, got {feature_dim}.")

        range_d = F.softplus(self.range_param)
        range_d = torch.clamp(range_d, max=self.max_range)
        base_knots = self.base_knots.view(1, 1, self.n_knots) * range_d.view(1, self.feature_dim, 1)
        if self.learnable_knot_offsets:
            knot_steps = F.softplus(self.knot_offsets).view(1, self.feature_dim, self.n_knots - 1)
            knot_steps = knot_steps / knot_steps.mean(dim=-1, keepdim=True).clamp_min(1e-6)
            knot_steps = knot_steps * (
                2.0 * range_d.view(1, self.feature_dim, 1) / max(self.n_knots - 1, 1)
            ) * self.knot_offset_scale
            knot_offsets = torch.cumsum(knot_steps, dim=-1)
            knot_offsets = F.pad(knot_offsets, (1, 0), mode="constant", value=0.0)
            knot_offsets = knot_offsets - knot_offsets.mean(dim=-1, keepdim=True)
            knots = base_knots + knot_offsets
        else:
            knots = base_knots

        step = (2.0 * range_d / max(self.n_knots - 1, 1)).clamp_min(1e-6)
        u = (x.unsqueeze(-1) - knots).abs() / step.view(1, self.feature_dim, 1)
        basis_inner = (4.0 - 6.0 * u.pow(2) + 3.0 * u.pow(3)) / 6.0
        basis_outer = (2.0 - u).clamp(min=0.0).pow(3) / 6.0
        basis = torch.where(u < 1.0, basis_inner, basis_outer)
        basis = torch.where(u < 2.0, basis, torch.zeros_like(basis))
        weight = self.weight.view(1, self.feature_dim, self.n_knots)
        return torch.sum(basis * weight, dim=-1)


class PatchSplineCalibrator(nn.Module):
    def __init__(
        self,
        num_channels=None,
        num_knots=8,
        spline_init_range=0.3,
        spline_max_range=5.0,
        learnable_knot_offsets=False,
        knot_offset_scale=1.0,
        patch_len=5,
        patch_hidden_dim=64,
        patch_init_scale=0.1,
        eps=1e-6,
        **_,
    ):
        super().__init__()
        if num_channels is None or num_channels < 1:
            raise ValueError("num_channels must be a positive integer.")
        if patch_len < 1:
            raise ValueError("patch_len must be positive.")
        if patch_hidden_dim < 1:
            raise ValueError("patch_hidden_dim must be positive.")
        if patch_init_scale < 0.0:
            raise ValueError("patch_init_scale must be non-negative.")

        self.eps = float(eps)
        self.patch_len = int(patch_len)
        self.num_channels = int(num_channels)
        self.patch_hidden_dim = int(patch_hidden_dim)
        self.revin = Normalize(num_features=self.num_channels, eps=self.eps, affine=False)

        conv_kernel = 3 if self.patch_len >= 3 else 1
        conv_padding = conv_kernel // 2
        self.patch_embed_conv = nn.Conv1d(1, self.patch_hidden_dim, kernel_size=conv_kernel, padding=conv_padding, bias=True)
        self.patch_embed_act = nn.GELU()
        self.patch_spline = VectorSplineBank(
            feature_dim=self.patch_hidden_dim,
            n_knots=num_knots,
            init_range=spline_init_range,
            max_range=spline_max_range,
            learnable_knot_offsets=learnable_knot_offsets,
            knot_offset_scale=knot_offset_scale,
        )
        self.patch_out = nn.Linear(self.patch_hidden_dim, self.patch_len, bias=True)
        with torch.no_grad():
            self.patch_out.weight.zero_()
            self.patch_out.bias.zero_()

        self.patch_gate_param = nn.Parameter(torch.ones(self.num_channels) * float(patch_init_scale))


    def _normalize(self, x):
        z = self.revin(x, mode="norm")
        means = getattr(self.revin, "mean", None)
        if means is None:
            means = getattr(self.revin, "last", None)
        stdev = self.revin.stdev.clamp_min(self.eps)
        return z, means, stdev

    def _patch_context(self, z):
        batch_size, seq_len, _ = z.shape
        remainder = seq_len % self.patch_len
        pad_right = (self.patch_len - remainder) % self.patch_len

        z_ch = z.transpose(1, 2)
        if pad_right > 0:
            z_ch = F.pad(z_ch, (0, pad_right), mode="replicate")

        seq_len_pad = z_ch.size(-1)
        num_patches = seq_len_pad // self.patch_len
        patches = z_ch.view(batch_size, self.num_channels, num_patches, self.patch_len)

        patch_vectors = patches.reshape(batch_size * self.num_channels * num_patches, self.patch_len)
        patch_conv = self.patch_embed_conv(patch_vectors.unsqueeze(1))
        patch_conv = self.patch_embed_act(patch_conv)
        patch_embed = patch_conv.mean(dim=-1)
        patch_spline = self.patch_spline(patch_embed)
        patch_delta = self.patch_out(patch_spline)

        patch_delta = patch_delta.view(batch_size, self.num_channels, num_patches, self.patch_len)
        patch_delta = patch_delta.reshape(batch_size, self.num_channels, seq_len_pad)
        if pad_right > 0:
            patch_delta = patch_delta[:, :, :seq_len]
        patch_delta = patch_delta.transpose(1, 2)

        patch_gate = torch.tanh(self.patch_gate_param).view(1, 1, self.num_channels)
        patch_correction = patch_gate * patch_delta
        return (
            patch_correction,
            patch_gate,
            patch_embed.view(batch_size, self.num_channels, num_patches, self.patch_hidden_dim),
            patch_spline.view(batch_size, self.num_channels, num_patches, self.patch_hidden_dim),
            patch_vectors.view(batch_size, self.num_channels, num_patches, self.patch_len),
            patch_conv.view(batch_size, self.num_channels, num_patches, self.patch_hidden_dim, self.patch_len),
        )

    def forward(self, y_hat, x_context=None, return_params=False):
        if y_hat.dim() != 3:
            raise ValueError("Expected y_hat shape [B, L, D].")

        z, means, stdev = self._normalize(y_hat)
        patch_correction, patch_gate, patch_embed, patch_spline, patch_vectors, patch_conv = self._patch_context(z)
        patch_correction = patch_correction - patch_correction.mean(dim=1, keepdim=True)
        z_calibrated = z + patch_correction
        calibrated = self.revin(z_calibrated, mode="denorm")
        correction = calibrated - y_hat

        if not return_params:
            return calibrated

        aux = {
            "z": z,
            "patch_vectors": patch_vectors,
            "patch_conv": patch_conv,
            "patch_embed": patch_embed,
            "patch_spline": patch_spline,
            "patch_delta": patch_correction,
            "z_calibrated": z_calibrated,
            "patch_gate": patch_gate.detach(),
            "correction": correction,
            "context_mean": means,
            "context_std": stdev,
            "patch_spline_range": F.softplus(self.patch_spline.range_param).detach(),
            "patch_spline_weight": self.patch_spline.weight,
            "patch_spline_knot_offsets": self.patch_spline.knot_offsets.detach() if self.patch_spline.knot_offsets is not None else None,
        }
        return calibrated, aux


class PostHocPatchSplineCalibration(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.calibrator = PatchSplineCalibrator(**kwargs)

    def forward(self, y_hat, x_context=None, return_params=False):
        return self.calibrator(y_hat, x_context=x_context, return_params=return_params)


class Model(PostHocPatchSplineCalibration):
    pass
