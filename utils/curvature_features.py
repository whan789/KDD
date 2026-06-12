"""Times2D-style derivative feature and calibration utilities.

This module keeps d1/d2 derivative helpers for [B, L, C] tensors and provides
a lightweight FSDH-only calibration layer inspired by Times2D.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-6


def _validate_timeseries(x: torch.Tensor, name: str = "x") -> None:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.dim() != 3:
        raise ValueError(f"{name} must have shape [B, L, C]")


def first_order_diff(x: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
    """Return first-order temporal differences with the same shape as ``x``."""
    _validate_timeseries(x)
    if x.size(1) == 0:
        return x.clone()

    d1 = x[:, 1:, :] - x[:, :-1, :]
    pad = x.new_full((x.size(0), 1, x.size(2)), pad_value)
    return torch.cat([pad, d1], dim=1)


def second_order_diff(x: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
    """Return second-order temporal differences with the same shape as ``x``."""
    _validate_timeseries(x)
    d1 = first_order_diff(x, pad_value=pad_value)
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    pad = x.new_full((x.size(0), 1, x.size(2)), pad_value)
    return torch.cat([pad, d2], dim=1)


def curvature_feature(x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Approximate local curvature from first and second temporal differences."""
    _validate_timeseries(x)
    d1 = first_order_diff(x)
    d2 = second_order_diff(x)
    return d2 / torch.pow(1.0 + d1.pow(2), 1.5).clamp_min(eps)


def recent_shape_context(
    x: torch.Tensor,
    recent_len: int = 12,
    eps: float = _EPS,
) -> torch.Tensor:
    """Return recent slope, curvature, and volatility context.

    Args:
        x: Input sequence with shape [B, L, C].
        recent_len: Number of most recent input steps used for context.

    Returns:
        Tensor with shape [B, C, 3]. The last dimension contains recent slope
        magnitude, curvature magnitude, and volatility.
    """
    _validate_timeseries(x)
    if recent_len < 1:
        raise ValueError("recent_len must be positive")

    recent_len = min(recent_len, x.size(1))
    d1 = first_order_diff(x)
    d2 = second_order_diff(x)

    x_recent = x[:, -recent_len:, :]
    d1_recent = d1[:, -recent_len:, :]
    d2_recent = d2[:, -recent_len:, :]

    slope_mag = d1_recent.abs().mean(dim=1)
    curv_mag = d2_recent.abs().mean(dim=1)
    volatility = x_recent.std(dim=1, unbiased=False)

    context = torch.stack([slope_mag, curv_mag, volatility], dim=-1)
    mean = context.mean(dim=1, keepdim=True)
    std = context.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (context - mean) / std


def segment_shape_context(
    x: torch.Tensor,
    segment_len: int = 12,
    stride: Optional[int] = None,
    sharpness_temperature: float = 1.0,
    eps: float = _EPS,
    return_aux: bool = False,
):
    """Summarize local shape contexts from multiple segments in ``x``.

    Each segment keeps signed and magnitude derivative information. Segment
    contexts are combined by sharpness-weighted softmax pooling, but the output
    remains a single context vector for one conservative correction branch.
    """
    _validate_timeseries(x)
    if segment_len < 1:
        raise ValueError("segment_len must be positive")
    if stride is None:
        stride = segment_len
    if stride < 1:
        raise ValueError("stride must be positive")
    if sharpness_temperature <= 0:
        raise ValueError("sharpness_temperature must be positive")

    length = x.size(1)
    segment_len = min(segment_len, length)
    if length == segment_len:
        starts = [0]
    else:
        starts = list(range(0, length - segment_len + 1, stride))
        last_start = length - segment_len
        if starts[-1] != last_start:
            starts.append(last_start)

    d1 = first_order_diff(x)
    d2 = second_order_diff(x)
    contexts = []
    scores = []
    for start in starts:
        end = start + segment_len
        x_seg = x[:, start:end, :]
        d1_seg = d1[:, start:end, :]
        d2_seg = d2[:, start:end, :]

        mean_d1 = d1_seg.mean(dim=1)
        mean_d2 = d2_seg.mean(dim=1)
        slope_mag = d1_seg.abs().mean(dim=1)
        curv_mag = d2_seg.abs().mean(dim=1)
        volatility = x_seg.std(dim=1, unbiased=False)
        context = torch.stack([mean_d1, mean_d2, slope_mag, curv_mag, volatility], dim=-1)
        contexts.append(context)
        scores.append(curv_mag + 0.5 * slope_mag)

    context_stack = torch.stack(contexts, dim=2)
    score_stack = torch.stack(scores, dim=-1)
    weights = torch.softmax(score_stack / sharpness_temperature, dim=-1)
    context = (context_stack * weights.unsqueeze(-1)).sum(dim=2)

    mean = context.mean(dim=1, keepdim=True)
    std = context.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    context = (context - mean) / std

    if not return_aux:
        return context
    return context, {
        "segment_contexts": context_stack,
        "segment_scores": score_stack,
        "segment_weights": weights,
        "segment_starts": starts,
    }


def compute_derivative_heatmaps(x: torch.Tensor) -> torch.Tensor:
    """Times2D-style first/second derivative heatmap.

    Args:
        x: Input sequence with shape [B, L, C].

    Returns:
        Tensor with shape [B, C, L, 2], where the last dimension contains
        first-order and second-order temporal differences.
    """
    _validate_timeseries(x)
    d1 = first_order_diff(x)
    d2 = second_order_diff(x)
    return torch.stack([d1, d2], dim=-1).permute(0, 2, 1, 3)


def derivative_feature_stack(
    x: torch.Tensor,
    normalize: bool = False,
    eps: float = _EPS,
) -> torch.Tensor:
    """Return derivative-only features: first diff and second diff."""
    _validate_timeseries(x)
    features = [first_order_diff(x), second_order_diff(x)]
    if normalize:
        features = [_normalize_feature(feature, eps=eps) for feature in features]
    return torch.cat(features, dim=-1)


def append_times2d_features(
    x: torch.Tensor,
    include_original: bool = True,
    normalize: bool = False,
    eps: float = _EPS,
) -> torch.Tensor:
    """Append Times2D FSDH derivative channels to a [B, L, C] tensor."""
    _validate_timeseries(x)
    features = []
    if include_original:
        features.append(x)
    features.append(derivative_feature_stack(x, normalize=normalize, eps=eps))
    return torch.cat(features, dim=-1)


def _normalize_feature(x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std


class Times2DFeatureAugment(nn.Module):
    """Small module wrapper for adding d1/d2 derivative feature channels."""

    def __init__(self, include_original: bool = True, normalize: bool = False, eps: float = _EPS):
        super().__init__()
        self.include_original = include_original
        self.normalize = normalize
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return append_times2d_features(
            x,
            include_original=self.include_original,
            normalize=self.normalize,
            eps=self.eps,
        )


class ShapeAwareSOM(nn.Module):
    """Times2D derivative correction with optional spline-style SOM modulation."""

    def __init__(
        self,
        num_channels: int,
        seq_len: int,
        pred_len: int,
        correction_scale: float = 1.0,
        init_scale: float = 0.01,
        use_times2d_modulation: bool = False,
        num_knots: int = 8,
        modulation_bound: float = 0.5,
        modulation_init: float = 0.0,
        d2_weight: float = 1.0,
        sharpness_temperature: float = 1.0,
        channel_wise: bool = True,
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
        if num_knots < 2:
            raise ValueError("num_knots must be >= 2")
        if modulation_bound < 0:
            raise ValueError("modulation_bound must be non-negative")
        if sharpness_temperature <= 0:
            raise ValueError("sharpness_temperature must be positive")

        self.num_channels = int(num_channels)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.correction_scale = float(correction_scale)
        self.use_times2d_modulation = bool(use_times2d_modulation)
        self.num_knots = int(num_knots)
        self.modulation_bound = float(modulation_bound)
        self.d2_weight = float(d2_weight)
        self.sharpness_temperature = float(sharpness_temperature)
        self.channel_wise = bool(channel_wise)
        self.eps = eps

        self.derivative_weight = nn.Parameter(init_scale * torch.randn(num_channels, 2))
        self.heatmap_to_pred = nn.Conv1d(in_channels=seq_len, out_channels=pred_len, kernel_size=1)

        self.register_buffer("knot_centers", torch.linspace(0.0, 1.0, self.num_knots))
        param_shape = (self.num_knots, self.num_channels) if self.channel_wise else (self.num_knots, 1)
        self.modulation_logits = nn.Parameter(torch.full(param_shape, float(modulation_init)))

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

    def _sharpness_score(self, x: torch.Tensor):
        d1 = first_order_diff(x)
        d2 = second_order_diff(x)
        d1_scale = d1.abs().mean(dim=1, keepdim=True).detach().clamp_min(self.eps)
        d2_scale = d2.abs().mean(dim=1, keepdim=True).detach().clamp_min(self.eps)
        d1_norm = d1.abs() / d1_scale
        d2_norm = d2.abs() / d2_scale
        score = torch.sqrt(d1_norm.pow(2) + self.d2_weight * d2_norm.pow(2) + self.eps)
        sharpness = 1.0 - torch.exp(-score / max(self.sharpness_temperature, self.eps))
        return d1, d2, sharpness

    def _spline_basis(self, sharpness: torch.Tensor) -> torch.Tensor:
        centers = self.knot_centers.view(1, 1, 1, self.num_knots)
        width = 1.0 / float(max(self.num_knots - 1, 1))
        basis = F.relu(1.0 - (sharpness.unsqueeze(-1) - centers).abs() / max(width, self.eps))
        return basis / basis.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def _modulation_map(self, basis: torch.Tensor) -> torch.Tensor:
        table = self.modulation_bound * torch.tanh(self.modulation_logits)
        if self.channel_wise:
            delta = (basis * table.t().view(1, 1, self.num_channels, self.num_knots)).sum(dim=-1)
        else:
            delta = torch.einsum("blck,kd->blcd", basis, table).squeeze(-1)
        return 1.0 + delta, delta

    def _resize_time_length(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.size(1) == target_len:
            return x
        x_t = x.permute(0, 2, 1)
        x_t = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
        return x_t.permute(0, 2, 1)

    def forward(
        self,
        y_hat: torch.Tensor,
        x_enc: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        self._check_inputs(y_hat, x_enc)
        heatmap = compute_derivative_heatmaps(x_enc)
        weight = self.derivative_weight.view(1, self.num_channels, 1, 2)
        heatmap_features = (heatmap * weight).sum(dim=-1)
        base_correction = self.heatmap_to_pred(heatmap_features.permute(0, 2, 1))
        base_correction = self.correction_scale * base_correction[:, -y_hat.size(1):, :]

        x_d1, x_d2, sharpness = self._sharpness_score(x_enc)
        sharpness = self._resize_time_length(sharpness, y_hat.size(1))
        x_d1 = self._resize_time_length(x_d1, y_hat.size(1))
        x_d2 = self._resize_time_length(x_d2, y_hat.size(1))
        knot_basis = None
        modulation_map = None
        modulation_delta = None
        correction = base_correction
        if self.use_times2d_modulation:
            knot_basis = self._spline_basis(sharpness)
            modulation_map, modulation_delta = self._modulation_map(knot_basis)
            correction = correction * modulation_map

        y = y_hat + correction

        if not return_aux:
            return y

        aux = {
            "heatmap": heatmap,
            "derivative_weight": self.derivative_weight,
            "heatmap_features": heatmap_features,
            "base_correction": base_correction,
            "correction": correction,
            "x_d1": x_d1,
            "x_d2": x_d2,
            "sharpness": sharpness,
            "x_sharpness": sharpness,
            "knot_basis": knot_basis,
            "modulation_map": modulation_map,
            "modulation_delta": modulation_delta,
            "context": None,
        }
        return y, aux


class ShapeAwareSOMCalibration(nn.Module):
    """Wrapper with the same call shape as post-hoc calibration modules."""

    def __init__(self, **kwargs):
        super().__init__()
        self.calibrator = ShapeAwareSOM(**kwargs)

    def forward(self, y_hat: torch.Tensor, x_context: Optional[torch.Tensor] = None, return_params: bool = False):
        if x_context is None:
            if return_params:
                return y_hat, {"context": None, "correction": None}
            return y_hat
        return self.calibrator(y_hat, x_context, return_aux=return_params)
