import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWiseSpline1D(nn.Module):
    def __init__(
        self,
        n_channels,
        n_knots=8,
        init_range=0.3,
        max_range=5.0,
        learnable_knot_offsets=False,
        knot_offset_scale=1.0,
    ):
        super().__init__()
        if n_channels is None or n_channels < 1:
            raise ValueError("n_channels must be a positive integer.")
        if n_knots < 2:
            raise ValueError("n_knots must be >= 2.")
        if init_range <= 0:
            raise ValueError("init_range must be positive.")
        if max_range <= 0:
            raise ValueError("max_range must be positive.")
        if knot_offset_scale < 0.0:
            raise ValueError("knot_offset_scale must be non-negative.")

        self.n_channels = int(n_channels)
        self.n_knots = int(n_knots)
        self.max_range = float(max_range)
        self.learnable_knot_offsets = bool(learnable_knot_offsets)
        self.knot_offset_scale = float(knot_offset_scale)

        base_knots = torch.linspace(-1.0, 1.0, self.n_knots)
        self.register_buffer("base_knots", base_knots)
        self.range_param = nn.Parameter(torch.ones(self.n_channels) * float(init_range))
        if self.learnable_knot_offsets:
            self.knot_offsets = nn.Parameter(torch.zeros(self.n_channels, self.n_knots - 1))
        else:
            self.register_parameter("knot_offsets", None)
        self.weight = nn.Parameter(torch.zeros(self.n_channels, self.n_knots))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, y):
        if y.dim() != 3:
            raise ValueError("Expected y shape [B, L, D].")
        _, _, d = y.shape
        if d != self.n_channels:
            raise ValueError(f"Channel mismatch: expected {self.n_channels}, got {d}.")

        range_d = F.softplus(self.range_param)
        range_d = torch.clamp(range_d, max=self.max_range)
        base_knots = self.base_knots.view(1, 1, 1, self.n_knots) * range_d.view(1, 1, d, 1)
        if self.learnable_knot_offsets:
            knot_steps = F.softplus(self.knot_offsets).view(1, 1, d, self.n_knots - 1)
            knot_steps = knot_steps / knot_steps.mean(dim=-1, keepdim=True).clamp_min(1e-6)
            knot_steps = knot_steps * (
                2.0 * range_d.view(1, 1, d, 1) / max(self.n_knots - 1, 1)
            ) * self.knot_offset_scale
            knot_offsets = torch.cumsum(knot_steps, dim=-1)
            knot_offsets = F.pad(knot_offsets, (1, 0), mode="constant", value=0.0)
            knot_offsets = knot_offsets - knot_offsets.mean(dim=-1, keepdim=True)
            knots = base_knots + knot_offsets
        else:
            knots = base_knots
        step = (2.0 * range_d / max(self.n_knots - 1, 1)).clamp_min(1e-6)
        u = (y.unsqueeze(-1) - knots).abs() / step.view(1, 1, d, 1)

        # Uniform cubic B-spline basis with compact support on |u| < 2.
        basis_inner = (4.0 - 6.0 * u.pow(2) + 3.0 * u.pow(3)) / 6.0
        basis_outer = (2.0 - u).clamp(min=0.0).pow(3) / 6.0
        basis = torch.where(u < 1.0, basis_inner, basis_outer)
        basis = torch.where(u < 2.0, basis, torch.zeros_like(basis))

        weight = self.weight.view(1, 1, d, self.n_knots)
        return torch.sum(basis * weight, dim=-1)


class KANLiteSplineCalibrator(nn.Module):
    def __init__(
        self,
        num_channels=None,
        num_knots=8,
        spline_init_range=0.3,
        spline_max_range=5.0,
        learnable_knot_offsets=False,
        knot_offset_scale=1.0,
        mixer_init_scale=0.1,
        eps=1e-6,
        **_,
    ):
        super().__init__()
        if mixer_init_scale < 0.0:
            raise ValueError("mixer_init_scale must be non-negative.")
        self.eps = float(eps)
        self.spline = ChannelWiseSpline1D(
            n_channels=num_channels,
            n_knots=num_knots,
            init_range=spline_init_range,
            max_range=spline_max_range,
            learnable_knot_offsets=learnable_knot_offsets,
            knot_offset_scale=knot_offset_scale,
        )
        self.mixer = nn.Linear(num_channels, num_channels, bias=True)
        with torch.no_grad():
            self.mixer.weight.zero_()
            eye = torch.eye(num_channels, dtype=self.mixer.weight.dtype)
            self.mixer.weight.add_(eye)
            self.mixer.bias.zero_()
        self.mix_gate_param = nn.Parameter(torch.tensor(float(mixer_init_scale)))

    def _normalize(self, x):
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        stdev = stdev.clamp_min(self.eps)
        return (x - means) / stdev, means, stdev

    def forward(self, y_hat, x_context=None, return_params=False):
        if y_hat.dim() != 3:
            raise ValueError("Expected y_hat shape [B, L, D].")

        z, means, stdev = self._normalize(y_hat)
        mixed = self.mixer(z)
        mix_gate = torch.tanh(self.mix_gate_param)
        z_mixed = z + mix_gate * mixed

        spline_res = self.spline(z_mixed)
        spline_res = spline_res - spline_res.mean(dim=1, keepdim=True)
        correction = spline_res * stdev
        calibrated = y_hat + correction

        if not return_params:
            return calibrated

        aux = {
            "z": z,
            "z_mixed": z_mixed,
            "mixed": mixed,
            "mix_gate": mix_gate.detach(),
            "correction": correction,
            "spline_res": spline_res,
            "context_mean": means,
            "context_std": stdev,
            "spline_range": F.softplus(self.spline.range_param).detach(),
            "spline_weight": self.spline.weight,
            "knot_offsets": self.spline.knot_offsets.detach() if self.spline.knot_offsets is not None else None,
        }
        return calibrated, aux


class PostHocKANLiteSplineCalibration(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.calibrator = KANLiteSplineCalibrator(**kwargs)

    def forward(self, y_hat, x_context=None, return_params=False):
        return self.calibrator(y_hat, x_context=x_context, return_params=return_params)


class Model(PostHocKANLiteSplineCalibration):
    pass
