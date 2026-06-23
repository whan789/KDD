"""Derivative-conditioned post-hoc forecast sharpening.

    This module intentionally keeps the post-hoc correction simple. It predicts a
    residual correction from a frozen backbone forecast so that ``y_hat + delta``
    can be trained toward the ground truth.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def finite_differences(x: torch.Tensor):
    if x.dim() != 3:
        raise ValueError("Expected input shape [B, L, D].")
    d1 = torch.zeros_like(x)
    d1[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
    d2 = torch.zeros_like(x)
    d2[:, 1:, :] = d1[:, 1:, :] - d1[:, :-1, :]
    return d1, d2


def need_score_from_forecast(
    values: torch.Tensor,
    window: int = 5,
    excursion_weight: float = 1.0,
    derivative_weight: float = 0.5,
    d2_weight: float = 0.5,
    eps: float = 1e-6,
):
    if values.dim() != 3:
        raise ValueError("Expected values shape [B, L, D].")

    if window < 1:
        window = 1
    if window % 2 == 0:
        window += 1

    d1, d2 = finite_differences(values)
    values_t = values.permute(0, 2, 1)
    if window <= 1:
        excursion = torch.zeros_like(values)
        d1_energy = d1.abs()
        d2_energy = d2.abs()
    else:
        local_max = F.max_pool1d(values_t, kernel_size=window, stride=1, padding=window // 2)
        local_min = -F.max_pool1d(-values_t, kernel_size=window, stride=1, padding=window // 2)
        excursion = (local_max - local_min).permute(0, 2, 1)

        d1_energy = F.avg_pool1d(
            d1.abs().permute(0, 2, 1),
            kernel_size=window,
            stride=1,
            padding=window // 2,
            count_include_pad=False,
        ).permute(0, 2, 1)
        d2_energy = F.avg_pool1d(
            d2.abs().permute(0, 2, 1),
            kernel_size=window,
            stride=1,
            padding=window // 2,
            count_include_pad=False,
        ).permute(0, 2, 1)

    derivative_energy = d1_energy
    if d2_weight > 0.0:
        derivative_energy = derivative_energy + float(d2_weight) * d2_energy

    def _normalize_component(x: torch.Tensor):
        return x / x.mean(dim=1, keepdim=True).clamp_min(eps)

    score = float(excursion_weight) * _normalize_component(excursion)
    if derivative_weight > 0.0:
        score = score + float(derivative_weight) * _normalize_component(derivative_energy)
    score = score / score.mean(dim=1, keepdim=True).clamp_min(eps)
    score = score / score.amax(dim=1, keepdim=True).clamp_min(eps)
    return score.detach()


class DerivativeConditionedSharpening(nn.Module):
    """Sharpen a forecast from y_hat + derivative features."""

    def __init__(
        self,
        num_channels: int,
        pred_len: int,
        hidden_dim: int = 16,
        kernel_size: int = 3,
        correction_scale: float = 1.0,
        need_window: int = 5,
        need_excursion_weight: float = 1.0,
        need_derivative_weight: float = 0.5,
        need_d2_weight: float = 0.5,
        dropout: float = 0.0,
        eps: float = 1e-6,
        **_,
    ):
        super().__init__()
        if num_channels < 1:
            raise ValueError("num_channels must be positive.")
        if pred_len < 1:
            raise ValueError("pred_len must be positive.")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if correction_scale < 0.0:
            raise ValueError("correction_scale must be non-negative.")
        self.num_channels = int(num_channels)
        self.pred_len = int(pred_len)
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        self.correction_scale = float(correction_scale)
        self.need_window = int(need_window)
        self.need_excursion_weight = float(need_excursion_weight)
        self.need_derivative_weight = float(need_derivative_weight)
        self.need_d2_weight = float(need_d2_weight)
        self.eps = float(eps)

        padding = kernel_size // 2
        in_channels = self.num_channels
        hidden_channels = self.hidden_dim * self.num_channels

        # Keep the module channel-wise so each variable can learn its own
        # oversmoothing-to-sharpness mapping.
        self.feature_in = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=self.num_channels,
            bias=True,
        )
        self.feature_mid = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=self.num_channels,
            bias=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.correction_head = nn.Conv1d(
            hidden_channels,
            self.num_channels,
            kernel_size=1,
            groups=self.num_channels,
            bias=True,
        )

    def forward(
        self,
        y_hat: torch.Tensor,
        x_context: Optional[torch.Tensor] = None,
        return_params: bool = False,
    ):
        if y_hat.dim() != 3:
            raise ValueError("Expected y_hat shape [B, L, D].")
        if y_hat.size(1) != self.pred_len:
            raise ValueError(f"Expected prediction length {self.pred_len}, got {y_hat.size(1)}.")
        if y_hat.size(2) != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {y_hat.size(2)}.")

        features = y_hat.permute(0, 2, 1)
        hidden = F.gelu(self.feature_in(features))
        hidden = self.dropout(hidden)
        hidden = F.gelu(self.feature_mid(hidden))
        hidden = self.dropout(hidden)

        raw_correction = self.correction_scale * torch.tanh(self.correction_head(hidden).permute(0, 2, 1))
        correction = raw_correction
        y = y_hat + correction

        if not return_params:
            return y

        aux = {
            "raw_correction": raw_correction,
            "correction": correction,
            "hidden_features": hidden.permute(0, 2, 1),
            "context": x_context,
        }
        return y, aux


class DerivativeConditionedSharpeningCalibration(nn.Module):
    """Wrapper matching the existing post-hoc calibration interface."""

    def __init__(self, **kwargs):
        super().__init__()
        self.calibrator = DerivativeConditionedSharpening(**kwargs)

    def forward(self, y_hat: torch.Tensor, x_context: Optional[torch.Tensor] = None, return_params: bool = False):
        return self.calibrator(y_hat, x_context=x_context, return_params=return_params)
