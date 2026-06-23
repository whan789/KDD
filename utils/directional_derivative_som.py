"""Directional derivative SOM calibration.

This module extends derivative-aware post-hoc calibration with separate peak
and valley correction branches. The backbone forecast is kept frozen; the
calibrator predicts signed residual corrections from input derivatives.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.curvature_features import (
    _EPS,
    _validate_timeseries,
    compute_derivative_heatmaps,
    first_order_diff,
    second_order_diff,
)


class DirectionalDerivativeSOM(nn.Module):
    """Directional derivative-based calibration with peak/valley separation."""

    def __init__(
        self,
        num_channels: int,
        seq_len: int,
        pred_len: int,
        correction_scale: float = 1.0,
        init_scale: float = 0.01,
        correction_head_type: str = "mlp",
        correction_mlp_hidden_dim: int = 128,
        use_prediction_features: bool = False,
        prediction_feature_hidden_dim: int = 64,
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
        if correction_head_type not in ("conv", "mlp"):
            raise ValueError("correction_head_type must be either 'conv' or 'mlp'")
        if correction_mlp_hidden_dim < 1:
            raise ValueError("correction_mlp_hidden_dim must be positive")
        if prediction_feature_hidden_dim < 1:
            raise ValueError("prediction_feature_hidden_dim must be positive")

        self.num_channels = int(num_channels)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.correction_scale = float(correction_scale)
        self.correction_head_type = correction_head_type
        self.correction_mlp_hidden_dim = int(correction_mlp_hidden_dim)
        self.use_prediction_features = bool(use_prediction_features)
        self.prediction_feature_hidden_dim = int(prediction_feature_hidden_dim)
        self.eps = eps

        self.derivative_weight = nn.Parameter(init_scale * torch.randn(num_channels, 2))
        self.gate_d1_weight = nn.Parameter(torch.ones(num_channels))
        self.gate_d2_weight = nn.Parameter(torch.ones(num_channels))

        self.peak_head = self._build_head()
        self.valley_head = self._build_head()

        if self.use_prediction_features:
            self.prediction_feature_head = nn.Sequential(
                nn.Linear(4, self.prediction_feature_hidden_dim),
                nn.GELU(),
                nn.Linear(self.prediction_feature_hidden_dim, 2),
            )

    def _build_head(self):
        if self.correction_head_type == "mlp":
            return nn.Sequential(
                nn.Linear(self.seq_len, self.correction_mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(self.correction_mlp_hidden_dim, self.pred_len),
            )
        return nn.Conv1d(in_channels=self.seq_len, out_channels=self.pred_len, kernel_size=1)

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

    def _resize_time_length(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.size(1) == target_len:
            return x
        x_t = x.permute(0, 2, 1)
        x_t = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
        return x_t.permute(0, 2, 1)

    def _temporal_moving_average(self, x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        if kernel_size <= 1:
            return x
        padding = kernel_size // 2
        x_t = x.permute(0, 2, 1)
        x_t = F.pad(x_t, (padding, padding), mode="replicate")
        trend = F.avg_pool1d(x_t, kernel_size=kernel_size, stride=1)
        return trend.permute(0, 2, 1)

    def _prediction_feature_correction(self, y_hat: torch.Tensor) -> torch.Tensor:
        y_d1 = first_order_diff(y_hat)
        y_d2 = second_order_diff(y_hat)
        y_trend = self._temporal_moving_average(y_hat, kernel_size=3)
        y_local_deviation = y_hat - y_trend
        pred_features = torch.stack([y_hat, y_d1, y_d2, y_local_deviation], dim=-1)
        b, l, c, f = pred_features.shape
        projected = self.prediction_feature_head(pred_features.reshape(b * l * c, f))
        return projected.reshape(b, l, c, 2)

    def _project_head(self, head: nn.Module, heatmap_features: torch.Tensor) -> torch.Tensor:
        if self.correction_head_type == "mlp":
            b, c, length = heatmap_features.shape
            projected = head(heatmap_features.reshape(b * c, length))
            return projected.reshape(b, c, self.pred_len).permute(0, 2, 1)
        return head(heatmap_features.permute(0, 2, 1))

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

        peak_correction = self._project_head(self.peak_head, heatmap_features)
        valley_correction = self._project_head(self.valley_head, heatmap_features)

        x_d1 = self._resize_time_length(first_order_diff(x_enc), y_hat.size(1))
        x_d2 = self._resize_time_length(second_order_diff(x_enc), y_hat.size(1))
        gate_score = (
            self.gate_d2_weight.view(1, 1, self.num_channels) * x_d2
            + self.gate_d1_weight.view(1, 1, self.num_channels) * x_d1
        )
        peak_gate = torch.sigmoid(gate_score)
        valley_gate = torch.sigmoid(-gate_score)

        prediction_feature_correction = None
        if self.use_prediction_features:
            prediction_feature_correction = self._prediction_feature_correction(y_hat)
            peak_correction = peak_correction + prediction_feature_correction[..., 0]
            valley_correction = valley_correction + prediction_feature_correction[..., 1]

        correction = peak_gate * peak_correction + valley_gate * valley_correction
        correction = self.correction_scale * correction
        y = y_hat + correction

        if not return_aux:
            return y

        aux = {
            "heatmap": heatmap,
            "heatmap_features": heatmap_features,
            "derivative_weight": self.derivative_weight,
            "x_d1": x_d1,
            "x_d2": x_d2,
            "gate_d1_weight": self.gate_d1_weight,
            "gate_d2_weight": self.gate_d2_weight,
            "gate_score": gate_score,
            "peak_gate": peak_gate,
            "valley_gate": valley_gate,
            "peak_correction": peak_correction,
            "valley_correction": valley_correction,
            "prediction_feature_correction": prediction_feature_correction,
            "correction": correction,
            "base_correction": correction,
            "correction_head_type": self.correction_head_type,
            "context": None,
        }
        return y, aux


class DirectionalDerivativeSOMCalibration(nn.Module):
    """Wrapper matching the post-hoc calibration interface."""

    def __init__(self, **kwargs):
        super().__init__()
        self.calibrator = DirectionalDerivativeSOM(**kwargs)

    def forward(self, y_hat: torch.Tensor, x_context: Optional[torch.Tensor] = None, return_params: bool = False):
        if x_context is None:
            if return_params:
                return y_hat, {"context": None, "correction": None}
            return y_hat
        return self.calibrator(y_hat, x_context, return_aux=return_params)
