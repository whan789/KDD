import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def finite_differences(x):
    """
    Compute first and second temporal finite differences with the same shape as x.

    Args:
        x: Tensor with shape [B, L, D].

    Returns:
        d1: First-order difference, shape [B, L, D].
        d2: Second-order difference, shape [B, L, D].
    """
    if x.dim() != 3:
        raise ValueError("Expected input shape [B, L, D].")

    d1 = torch.zeros_like(x)
    d1[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]

    d2 = torch.zeros_like(x)
    d2[:, 1:, :] = d1[:, 1:, :] - d1[:, :-1, :]
    return d1, d2


class TrendFluctuationCalibrator(nn.Module):
    """
    Trend-first calibration branch.

    This module predicts a future trend from the input context and uses it to
    correct the backbone forecast. Instead of trying to learn local sharpness
    corrections, it explicitly models the low-frequency component that the
    backbone may miss under distribution shift.
    """

    def __init__(
        self,
        seq_len=None,
        pred_len=None,
        moving_avg_kernel=3,
        channel_wise=False,
        num_channels=None,
        trend_mix=1.0,
        eps=1e-6,
        **_unused,
    ):
        super(TrendFluctuationCalibrator, self).__init__()

        if moving_avg_kernel < 1 or moving_avg_kernel % 2 == 0:
            raise ValueError("moving_avg_kernel must be a positive odd integer.")
        if channel_wise and num_channels is None:
            raise ValueError("num_channels is required when channel_wise=True.")
        if seq_len is not None and seq_len < 1:
            raise ValueError("seq_len must be positive when provided.")
        if pred_len is not None and pred_len < 1:
            raise ValueError("pred_len must be positive when provided.")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.moving_avg_kernel = moving_avg_kernel
        self.channel_wise = channel_wise
        self.num_channels = num_channels
        self.eps = eps
        trend_mix = max(float(trend_mix), 1e-4)
        self.mix_raw = nn.Parameter(torch.tensor(math.log(math.exp(trend_mix) - 1.0)))

        self.trend_proj = None
        self.trend_proj_list = None
        if self.seq_len is not None and self.pred_len is not None:
            self._build_projector(self.seq_len, self.pred_len)

    def _build_projector(self, seq_len, pred_len, device=None, dtype=None):
        if self.channel_wise:
            self.trend_proj_list = nn.ModuleList()
            for _ in range(self.num_channels):
                layer = nn.Linear(seq_len, pred_len)
                layer.weight = nn.Parameter((1.0 / seq_len) * torch.ones([pred_len, seq_len], device=device, dtype=dtype))
                self.trend_proj_list.append(layer.to(device=device, dtype=dtype) if device is not None or dtype is not None else layer)
        else:
            self.trend_proj = nn.Linear(seq_len, pred_len)
            self.trend_proj.weight = nn.Parameter((1.0 / seq_len) * torch.ones([pred_len, seq_len], device=device, dtype=dtype))
            if device is not None or dtype is not None:
                self.trend_proj = self.trend_proj.to(device=device, dtype=dtype)

    def _temporal_moving_average(self, x):
        if self.moving_avg_kernel == 1:
            return x

        padding = self.moving_avg_kernel // 2
        x_t = x.permute(0, 2, 1)
        x_t = F.pad(x_t, (padding, padding), mode="replicate")
        trend = F.avg_pool1d(x_t, kernel_size=self.moving_avg_kernel, stride=1)
        return trend.permute(0, 2, 1)

    def _select_trend_source(self, x_context, y_hat):
        if x_context.size(-1) == y_hat.size(-1):
            return x_context
        return x_context[:, :, -1:].contiguous()

    def _ensure_projector(self, x_context, y_hat):
        seq_len = x_context.size(1)
        pred_len = y_hat.size(1)
        channels = x_context.size(-1)

        if self.seq_len is None:
            self.seq_len = seq_len
        if self.pred_len is None:
            self.pred_len = pred_len
        if self.channel_wise and self.num_channels is None:
            self.num_channels = channels

        needs_build = False
        if self.channel_wise:
            needs_build = self.trend_proj_list is None
        else:
            needs_build = self.trend_proj is None

        if needs_build:
            device = x_context.device
            dtype = x_context.dtype
            self._build_projector(self.seq_len, self.pred_len, device=device, dtype=dtype)

    def _project_trend(self, x_trend):
        if self.channel_wise:
            if x_trend.size(-1) != self.num_channels:
                raise ValueError(
                    "Input channel size does not match num_channels: "
                    f"{x_trend.size(-1)} vs {self.num_channels}."
                )
            trend_out = []
            x_t = x_trend.permute(0, 2, 1)
            for idx, layer in enumerate(self.trend_proj_list):
                trend_out.append(layer(x_t[:, idx, :]).unsqueeze(-1))
            return torch.cat(trend_out, dim=-1)

        trend_out = self.trend_proj(x_trend.permute(0, 2, 1))
        return trend_out.permute(0, 2, 1)

    def forward(self, y_hat, x_context=None, return_params=False):
        if y_hat.dim() != 3:
            raise ValueError("Expected y_hat shape [B, L, D].")

        d1, d2 = finite_differences(y_hat)

        if x_context is None or x_context.dim() != 3:
            calibrated = y_hat
            if not return_params:
                return calibrated
            aux = {
                "future_trend": None,
                "backbone_trend": None,
                "trend_delta": None,
                "trend_gain": F.softplus(self.mix_raw).detach(),
                "d1": d1,
                "d2": d2,
            }
            return calibrated, aux

        x_source = self._select_trend_source(x_context, y_hat)
        self._ensure_projector(x_source, y_hat)

        x_trend = self._temporal_moving_average(x_source)
        future_trend = self._project_trend(x_trend)
        backbone_trend = self._temporal_moving_average(y_hat)

        trend_delta = future_trend - backbone_trend
        trend_gain = F.softplus(self.mix_raw)
        calibrated = y_hat + trend_gain * trend_delta

        if not return_params:
            return calibrated

        aux = {
            "future_trend": future_trend,
            "backbone_trend": backbone_trend,
            "trend_delta": trend_delta,
            "trend_gain": trend_gain.detach(),
            "d1": d1,
            "d2": d2,
            "x_trend": x_trend,
        }
        return calibrated, aux


class PostHocTrendCalibration(nn.Module):
    """
    Thin wrapper name for plug-in use after any forecasting backbone.
    """

    def __init__(self, **kwargs):
        super(PostHocTrendCalibration, self).__init__()
        self.calibrator = TrendFluctuationCalibrator(**kwargs)

    def forward(self, y_hat, x_context=None, return_params=False):
        return self.calibrator(y_hat, x_context=x_context, return_params=return_params)


class Model(PostHocTrendCalibration):
    """Compatibility alias for code paths that expect a Model class."""

    pass
