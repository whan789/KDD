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


class ChannelWiseSpline1D(nn.Module):
    def __init__(
        self,
        n_channels,
        n_knots=8,
        init_range=0.3,
        max_range=5.0,
        spline_type="linear",
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
        if spline_type not in {"linear", "quadratic", "cubic"}:
            raise ValueError("spline_type must be one of {'linear', 'quadratic', 'cubic'}.")
        if knot_offset_scale < 0.0:
            raise ValueError("knot_offset_scale must be non-negative.")

        self.n_channels = int(n_channels)
        self.n_knots = int(n_knots)
        self.max_range = float(max_range)
        self.spline_type = spline_type
        self.learnable_knot_offsets = bool(learnable_knot_offsets)
        self.knot_offset_scale = float(knot_offset_scale)

        base_knots = torch.linspace(-1.0, 1.0, self.n_knots)
        self.register_buffer("base_knots", base_knots)
        self.range_param = nn.Parameter(torch.ones(self.n_channels) * float(init_range))

        if self.learnable_knot_offsets:
            # Raw knot-step logits. We transform these into positive increments
            # and take a cumulative sum in forward() so knot order is preserved.
            self.knot_offsets = nn.Parameter(torch.zeros(self.n_channels, self.n_knots - 1))
        else:
            self.register_parameter("knot_offsets", None)

        if self.spline_type == "linear":
            self.n_basis = self.n_knots
        elif self.spline_type == "quadratic":
            self.n_basis = self.n_knots + 3
        else:
            self.n_basis = self.n_knots + 4

        self.weight = nn.Parameter(torch.zeros(self.n_channels, self.n_basis))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(
        self,
        y,
        weight_delta=None,
        range_delta=None,
        knot_offset_delta=None,
        modulation_scale=1.0,
    ):
        if y.dim() != 3:
            raise ValueError("Expected y shape [B, L, D].")
        batch_size, _, d = y.shape
        if d != self.n_channels:
            raise ValueError(f"Channel mismatch: expected {self.n_channels}, got {d}.")

        modulation_scale = float(modulation_scale)
        if modulation_scale < 0.0:
            raise ValueError("modulation_scale must be non-negative.")

        seq_len = y.size(1)
        range_raw = self.range_param.view(1, 1, d).expand(batch_size, seq_len, -1)
        if range_delta is not None:
            if range_delta.shape == (batch_size, d):
                range_delta = range_delta.unsqueeze(1).expand(-1, seq_len, -1)
            elif range_delta.shape != (batch_size, seq_len, d):
                raise ValueError(
                    f"range_delta must have shape {(batch_size, d)} or {(batch_size, seq_len, d)}, got {tuple(range_delta.shape)}."
                )
            range_raw = range_raw + modulation_scale * torch.tanh(range_delta)
        range_d = F.softplus(range_raw)
        range_d = torch.clamp(range_d, max=self.max_range)
        base_knots = self.base_knots.view(1, 1, 1, self.n_knots) * range_d.view(batch_size, seq_len, d, 1)
        if self.learnable_knot_offsets:
            knot_steps_raw = self.knot_offsets.view(1, 1, d, self.n_knots - 1).expand(batch_size, seq_len, -1, -1)
            if knot_offset_delta is not None:
                if knot_offset_delta.shape == (batch_size, d, self.n_knots - 1):
                    knot_offset_delta = knot_offset_delta.unsqueeze(1).expand(-1, seq_len, -1, -1)
                elif knot_offset_delta.shape != (batch_size, seq_len, d, self.n_knots - 1):
                    raise ValueError(
                        "knot_offset_delta must have shape "
                        f"{(batch_size, d, self.n_knots - 1)} or {(batch_size, seq_len, d, self.n_knots - 1)}, got {tuple(knot_offset_delta.shape)}."
                    )
                knot_steps_raw = knot_steps_raw + modulation_scale * torch.tanh(knot_offset_delta)
            knot_steps = F.softplus(knot_steps_raw).view(batch_size, seq_len, d, self.n_knots - 1)
            knot_steps = knot_steps / knot_steps.mean(dim=-1, keepdim=True).clamp_min(1e-6)
            knot_steps = knot_steps * (
                2.0 * range_d.view(batch_size, seq_len, d, 1) / max(self.n_knots - 1, 1)
            ) * self.knot_offset_scale
            knot_offsets = torch.cumsum(knot_steps, dim=-1)
            knot_offsets = F.pad(knot_offsets, (1, 0), mode="constant", value=0.0)
            knot_offsets = knot_offsets - knot_offsets.mean(dim=-1, keepdim=True)
            knots = base_knots + knot_offsets
        else:
            knots = base_knots
        y_expanded = y.unsqueeze(-1)

        if self.spline_type == "linear":
            basis = F.relu(y_expanded - knots)
        elif self.spline_type == "quadratic":
            poly_basis = torch.cat(
                [
                    torch.ones_like(y_expanded),
                    y_expanded,
                    y_expanded.pow(2),
                ],
                dim=-1,
            )
            quad_hinge = F.relu(y_expanded - knots).pow(2)
            basis = torch.cat([poly_basis, quad_hinge], dim=-1)
        else:
            poly_basis = torch.cat(
                [
                    torch.ones_like(y_expanded),
                    y_expanded,
                    y_expanded.pow(2),
                    y_expanded.pow(3),
                ],
                dim=-1,
            )
            cubic_hinge = F.relu(y_expanded - knots).pow(3)
            basis = torch.cat([poly_basis, cubic_hinge], dim=-1)

        weight = self.weight.view(1, 1, d, self.n_basis).expand(batch_size, seq_len, -1, -1)
        if weight_delta is not None:
            if weight_delta.shape == (batch_size, d, self.n_basis):
                weight_delta = weight_delta.unsqueeze(1).expand(-1, seq_len, -1, -1)
            elif weight_delta.shape != (batch_size, seq_len, d, self.n_basis):
                raise ValueError(
                    f"weight_delta must have shape {(batch_size, d, self.n_basis)} or {(batch_size, seq_len, d, self.n_basis)}, got {tuple(weight_delta.shape)}."
                )
            weight = weight + modulation_scale * torch.tanh(weight_delta)
        return torch.sum(basis * weight, dim=-1)


class ValueSplineCalibrator(nn.Module):
    """
    Post-hoc spline calibration module for SOM (SCINet) output.

    Usage with SCINet:
        calibrator = ValueSplineCalibrator(num_channels=enc_in, ...)
        # y_hat: SCINet raw output [B, pred_len, D]
        # x_context: original encoder input [B, seq_len, D] (optional)
        calibrated = calibrator(y_hat, x_context=x_enc)
    """

    def __init__(
        self,
        num_knots=8,
        num_channels=None,
        spline_init_range=0.3,
        spline_max_range=5.0,
        spline_type="linear",

        learnable_knot_offsets=False,
        knot_offset_scale=1.0,
        use_sharpness_gate=False,
        use_error_aware_gate=False,
        error_gate_hidden_dim=16,
        error_gate_kernel_size=3,
        error_gate_max_boost=1.0,
        error_gate_temperature=1.0,
        sharpness_gate_floor=0.0,
        sharpness_gate_power=1.0,
        sharpness_d2_weight=1.0,
        sharpness_temperature=1.0,
        use_sharp_residual=False,
        sharp_residual_hidden_dim=16,
        sharp_residual_kernel_size=3,
        sharp_residual_scale=0.25,
        use_context_spline=False,
        context_spline_attention_dim=32,
        context_spline_num_heads=4,
        context_spline_ff_hidden_dim=64,
        context_spline_modulation_scale=0.5,
        eps=1e-6,
        **_,
    ):
        super().__init__()
        if sharpness_gate_floor < 0.0 or sharpness_gate_floor > 1.0:
            raise ValueError("sharpness_gate_floor must be in [0, 1].")
        if sharpness_gate_power <= 0.0:
            raise ValueError("sharpness_gate_power must be positive.")
        if sharpness_d2_weight < 0.0:
            raise ValueError("sharpness_d2_weight must be non-negative.")
        if sharpness_temperature <= 0.0:
            raise ValueError("sharpness_temperature must be positive.")
        if sharp_residual_hidden_dim < 1:
            raise ValueError("sharp_residual_hidden_dim must be positive.")
        if sharp_residual_kernel_size < 1 or sharp_residual_kernel_size % 2 == 0:
            raise ValueError("sharp_residual_kernel_size must be a positive odd integer.")
        if sharp_residual_scale < 0.0:
            raise ValueError("sharp_residual_scale must be non-negative.")
        if context_spline_attention_dim < 1:
            raise ValueError("context_spline_attention_dim must be positive.")
        if context_spline_num_heads < 1:
            raise ValueError("context_spline_num_heads must be positive.")
        if context_spline_attention_dim % context_spline_num_heads != 0:
            raise ValueError("context_spline_attention_dim must be divisible by context_spline_num_heads.")
        if context_spline_ff_hidden_dim < 1:
            raise ValueError("context_spline_ff_hidden_dim must be positive.")
        if context_spline_modulation_scale < 0.0:
            raise ValueError("context_spline_modulation_scale must be non-negative.")
        if error_gate_hidden_dim < 1:
            raise ValueError("error_gate_hidden_dim must be positive.")
        if error_gate_kernel_size < 1 or error_gate_kernel_size % 2 == 0:
            raise ValueError("error_gate_kernel_size must be a positive odd integer.")
        if error_gate_max_boost < 0.0:
            raise ValueError("error_gate_max_boost must be non-negative.")
        if error_gate_temperature <= 0.0:
            raise ValueError("error_gate_temperature must be positive.")

        self.eps = float(eps)
        self.use_sharpness_gate = bool(use_sharpness_gate)
        self.use_error_aware_gate = bool(use_error_aware_gate)
        self.spline_type = spline_type
        self.learnable_knot_offsets = bool(learnable_knot_offsets)
        self.knot_offset_scale = float(knot_offset_scale)
        self.sharpness_gate_floor = float(sharpness_gate_floor)
        self.sharpness_gate_power = float(sharpness_gate_power)
        self.sharpness_d2_weight = float(sharpness_d2_weight)
        self.sharpness_temperature = float(sharpness_temperature)
        self.use_sharp_residual = bool(use_sharp_residual)
        self.sharp_residual_scale = float(sharp_residual_scale)
        self.use_context_spline = bool(use_context_spline)
        self.context_spline_attention_dim = int(context_spline_attention_dim)
        self.context_spline_num_heads = int(context_spline_num_heads)
        self.context_spline_ff_hidden_dim = int(context_spline_ff_hidden_dim)
        self.context_spline_modulation_scale = float(context_spline_modulation_scale)
        self.error_gate_hidden_dim = int(error_gate_hidden_dim)
        self.error_gate_kernel_size = int(error_gate_kernel_size)
        self.error_gate_max_boost = float(error_gate_max_boost)
        self.error_gate_temperature = float(error_gate_temperature)
        self.spline = ChannelWiseSpline1D(
            n_channels=num_channels,
            n_knots=num_knots,
            init_range=spline_init_range,
            max_range=spline_max_range,
            spline_type=spline_type,
            learnable_knot_offsets=learnable_knot_offsets,
            knot_offset_scale=knot_offset_scale,
        )
        # [개선 2] correction 크기 제한 — 학습 초기 폭발 방지
        self.correction_scale = nn.Parameter(torch.ones(1) * 0.1)
        # [개선 3] temporal mixing — depthwise conv로 인접 시점 정보 혼합
        self.temporal_mix = nn.Conv1d(
            num_channels, num_channels,
            kernel_size=3, padding=1, groups=num_channels, bias=False,
        )
        nn.init.dirac_(self.temporal_mix.weight)   # identity 초기화 → 학습 전 동작 보존
        if self.use_context_spline:
            self.context_spline_query_in = nn.Linear(3 * num_channels, self.context_spline_attention_dim)
            self.context_spline_key_value_in = nn.Linear(3 * num_channels, self.context_spline_attention_dim)
            self.context_spline_attention = nn.MultiheadAttention(
                embed_dim=self.context_spline_attention_dim,
                num_heads=self.context_spline_num_heads,
                batch_first=True,
            )
            self.context_spline_ff = nn.Sequential(
                nn.Linear(self.context_spline_attention_dim, self.context_spline_ff_hidden_dim),
                nn.GELU(),
                nn.Linear(self.context_spline_ff_hidden_dim, self.context_spline_attention_dim),
            )
            self.context_spline_norm1 = nn.LayerNorm(self.context_spline_attention_dim)
            self.context_spline_norm2 = nn.LayerNorm(self.context_spline_attention_dim)
            self.context_spline_weight_out = nn.Linear(self.context_spline_attention_dim, num_channels * self.spline.n_basis)
            self.context_spline_range_out = nn.Linear(self.context_spline_attention_dim, num_channels)
            if self.learnable_knot_offsets:
                self.context_spline_knot_out = nn.Linear(
                    self.context_spline_attention_dim,
                    num_channels * (num_knots - 1),
                )
            else:
                self.context_spline_knot_out = None
        else:
            self.context_spline_query_in = None
            self.context_spline_key_value_in = None
            self.context_spline_attention = None
            self.context_spline_ff = None
            self.context_spline_norm1 = None
            self.context_spline_norm2 = None
            self.context_spline_weight_out = None
            self.context_spline_range_out = None
            self.context_spline_knot_out = None
        if self.use_error_aware_gate:
            padding = error_gate_kernel_size // 2
            self.error_gate_in = nn.Conv1d(
                3 * num_channels, self.error_gate_hidden_dim * num_channels,
                kernel_size=error_gate_kernel_size, padding=padding,
                groups=num_channels, bias=True,
            )
            self.error_gate_mid = nn.Conv1d(
                self.error_gate_hidden_dim * num_channels, self.error_gate_hidden_dim * num_channels,
                kernel_size=error_gate_kernel_size, padding=padding,
                groups=num_channels, bias=True,
            )
            self.error_gate_out = nn.Conv1d(
                self.error_gate_hidden_dim * num_channels, num_channels,
                kernel_size=1, groups=num_channels, bias=True,
            )
        else:
            self.error_gate_in = None
            self.error_gate_mid = None
            self.error_gate_out = None

        if self.use_sharp_residual:
            padding = sharp_residual_kernel_size // 2
            # [개선 4] 채널별 독립 conv — 각 채널이 자신만의 가중치로 패턴 학습
            # grouped conv: input channel 축을 [2*D], [H*D], [D] 로 구성하고
            # groups=D 로 묶으면 채널 간 가중치 공유가 완전히 사라짐
            self.sharp_residual_in = nn.Conv1d(
                2 * num_channels, sharp_residual_hidden_dim * num_channels,
                kernel_size=sharp_residual_kernel_size, padding=padding,
                groups=num_channels, bias=True,
            )
            self.sharp_residual_mid = nn.Conv1d(
                sharp_residual_hidden_dim * num_channels, sharp_residual_hidden_dim * num_channels,
                kernel_size=sharp_residual_kernel_size, padding=padding,
                groups=num_channels, bias=True,
            )
            self.sharp_residual_out = nn.Conv1d(
                sharp_residual_hidden_dim * num_channels, num_channels,
                kernel_size=1, groups=num_channels, bias=True,
            )
            self._sharp_hidden = sharp_residual_hidden_dim
            self._num_channels = num_channels
        else:
            self.sharp_residual_in = None
            self.sharp_residual_mid = None
            self.sharp_residual_out = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, x):
        """Returns (z, mean, std) where z = (x - mean) / std.
        mean/std are computed over the time axis and detached from the graph.
        """
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach().clamp_min(self.eps)
        return (x - means) / stdev, means, stdev

    def _compute_sharpness(self, z):
        """Compute per-position sharpness score from a *normalized* signal z.

        Returns (sharpness, d1, d2).  d1/d2 are the raw finite differences
        (before normalisation) and are reused by _compute_sharp_residual to
        avoid a second call to finite_differences.
        """
        d1, d2 = finite_differences(z)
        d1_scale = d1.abs().mean(dim=1, keepdim=True).detach().clamp_min(self.eps)
        d2_scale = d2.abs().mean(dim=1, keepdim=True).detach().clamp_min(self.eps)
        d1_norm = d1.abs() / d1_scale
        d2_norm = d2.abs() / d2_scale
        score = torch.sqrt(d1_norm.pow(2) + self.sharpness_d2_weight * d2_norm.pow(2) + self.eps)
        sharpness = 1.0 - torch.exp(-score / self.sharpness_temperature)
        return sharpness, d1, d2

    def _compute_sharpness_gate(self, sharpness):
        gate = (1.0 - sharpness).clamp(0.0, 1.0)
        if self.sharpness_gate_power != 1.0:
            gate = gate.pow(self.sharpness_gate_power)
        if self.sharpness_gate_floor > 0.0:
            gate = self.sharpness_gate_floor + (1.0 - self.sharpness_gate_floor) * gate
        return gate

    def _compute_error_aware_gate(self, gate_z, d1, d2):
        """Predict an error-aware amplification gate from observable features.

        The gate is shaped as a positive scale in [1, 1 + max_boost].
        """
        batch_size, seq_len, num_channels = gate_z.shape
        d1_scale = d1.abs().mean(dim=1, keepdim=True).detach().clamp_min(self.eps)
        d2_scale = d2.abs().mean(dim=1, keepdim=True).detach().clamp_min(self.eps)
        d1_norm = d1 / d1_scale
        d2_norm = d2 / d2_scale

        features = torch.stack([gate_z, d1_norm, d2_norm], dim=2)  # [B, L, 3, D]
        features = features.permute(0, 3, 2, 1).reshape(batch_size, 3 * num_channels, seq_len)
        x = F.gelu(self.error_gate_in(features))
        x = F.gelu(self.error_gate_mid(x))
        logits = self.error_gate_out(x)
        gate = 1.0 + self.error_gate_max_boost * torch.sigmoid(logits / self.error_gate_temperature)
        return gate.permute(0, 2, 1)

    def _compute_sharp_residual(self, gate_z, d1, d2, sharpness):
        """Compute sharp residual with fully channel-wise independent convolutions.

        Layout: interleave d1/d2 per channel so grouped conv sees
        [d1_ch0, d2_ch0, d1_ch1, d2_ch1, ...] → groups=D keeps channels separate.
        """
        batch_size, seq_len, num_channels = gate_z.shape
        d1_scale = d1.abs().mean(dim=1, keepdim=True).detach().clamp_min(self.eps)
        d2_scale = d2.abs().mean(dim=1, keepdim=True).detach().clamp_min(self.eps)
        d1_norm = d1 / d1_scale   # [B, L, D]
        d2_norm = d2 / d2_scale   # [B, L, D]

        # interleave: [B, 2*D, L]  (채널 c → index 2c, 2c+1)
        d_interleaved = torch.stack([d1_norm, d2_norm], dim=2)   # [B, L, 2, D]
        d_interleaved = d_interleaved.permute(0, 3, 2, 1)        # [B, D, 2, L]
        heatmap = d_interleaved.reshape(batch_size, 2 * num_channels, seq_len)  # [B, 2D, L]

        x = F.gelu(self.sharp_residual_in(heatmap))   # [B, H*D, L]
        x = F.gelu(self.sharp_residual_mid(x))        # [B, H*D, L]
        logits = self.sharp_residual_out(x)            # [B, D, L]

        residual = logits.permute(0, 2, 1)             # [B, L, D]
        residual = torch.tanh(residual) * sharpness
        residual = residual - residual.mean(dim=1, keepdim=True)
        return self.sharp_residual_scale * residual

    def _compute_context_spline_modulation(self, x_context, z):
        batch_size, pred_len, num_channels = z.shape
        context_z, _, _ = self._normalize(x_context)
        ctx_d1, ctx_d2 = finite_differences(context_z)
        pred_d1, pred_d2 = finite_differences(z)

        context_features = torch.cat([context_z, ctx_d1, ctx_d2], dim=-1)
        query_features = torch.cat([z, pred_d1, pred_d2], dim=-1)

        context_tokens = self.context_spline_key_value_in(context_features)
        query_tokens = self.context_spline_query_in(query_features)

        attended, attention_weights = self.context_spline_attention(
            query=query_tokens,
            key=context_tokens,
            value=context_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        attended = self.context_spline_norm1(query_tokens + attended)
        attended = self.context_spline_norm2(attended + self.context_spline_ff(attended))

        weight_delta = self.context_spline_weight_out(attended)
        weight_delta = weight_delta.view(batch_size, pred_len, num_channels, self.spline.n_basis)

        range_delta = self.context_spline_range_out(attended)
        range_delta = range_delta.view(batch_size, pred_len, num_channels)

        knot_offset_delta = None
        if self.context_spline_knot_out is not None:
            knot_offset_delta = self.context_spline_knot_out(attended)
            knot_offset_delta = knot_offset_delta.view(batch_size, pred_len, num_channels, self.spline.n_knots - 1)

        return {
            "weight_delta": weight_delta,
            "range_delta": range_delta,
            "knot_offset_delta": knot_offset_delta,
            "context_z": context_z,
            "context_d1": ctx_d1,
            "context_d2": ctx_d2,
            "attention_map": attention_weights.mean(dim=1),
        }

    @staticmethod
    def _match_seq_len(tensor, target_len):
        """Interpolate tensor from its current seq-len to target_len if needed."""
        if tensor.size(1) == target_len:
            return tensor
        return F.interpolate(
            tensor.permute(0, 2, 1),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, y_hat, x_context=None, return_params=False):
        """
        Args:
            y_hat      : model prediction  [B, L_pred, D]
            x_context  : encoder input     [B, L_enc,  D]  (optional)
                         When provided, normalisation statistics and sharpness
                         are derived from x_context instead of y_hat.
            return_params: if True, also return an aux dict with internals.

        Returns:
            calibrated [B, L_pred, D]  (+ aux dict if return_params=True)
        """
        if y_hat.dim() != 3:
            raise ValueError("Expected y_hat shape [B, L, D].")

        # ------------------------------------------------------------------
        # 1. Normalise y_hat using its own statistics → z used for spline
        # ------------------------------------------------------------------
        z, means, stdev = self._normalize(y_hat)

        # ------------------------------------------------------------------
        # 2. Spline correction (operates on z, i.e. normalised y_hat)
        #    Optional: modulate spline parameters from x_context patterns.
        # ------------------------------------------------------------------
        context_spline = None
        use_context = x_context is not None and x_context.dim() == 3
        if self.use_context_spline and use_context:
            context_spline = self._compute_context_spline_modulation(x_context, z)
            spline_res = self.spline(
                z,
                weight_delta=context_spline["weight_delta"],
                range_delta=context_spline["range_delta"],
                knot_offset_delta=context_spline["knot_offset_delta"],
                modulation_scale=self.context_spline_modulation_scale,
            )
        else:
            spline_res = self.spline(z)
        spline_res = spline_res - spline_res.mean(dim=1, keepdim=True)
        # [개선 3] temporal mixing: [B, L, D] → [B, D, L] → conv → [B, L, D]
        spline_res = self.temporal_mix(spline_res.permute(0, 2, 1)).permute(0, 2, 1)

        # ------------------------------------------------------------------
        # 3. Gate / residual source: x_context if available, else y_hat
        #    Normalised separately so sharpness is scale-invariant.
        # ------------------------------------------------------------------
        gate_source = x_context if use_context else y_hat
        gate_z, _, _ = self._normalize(gate_source)   # independent normalisation

        sharpness = None
        sharpness_gate = None
        error_gate = None
        sharp_residual = None
        x_d1 = None
        x_d2 = None

        if self.use_sharpness_gate or self.use_sharp_residual or self.use_error_aware_gate:
            sharpness, x_d1, x_d2 = self._compute_sharpness(gate_z)

        # ------------------------------------------------------------------
        # 4. Accumulate delta = spline_res [+ sharp_residual]
        # ------------------------------------------------------------------
        delta = spline_res

        if self.use_sharp_residual and sharpness is not None:
            # Reuse d1/d2 already computed above — no redundant finite_differences call
            sharp_residual = self._compute_sharp_residual(gate_z, x_d1, x_d2, sharpness)
            sharp_residual = self._match_seq_len(sharp_residual, delta.size(1))
            delta = delta + sharp_residual

        if self.use_error_aware_gate and sharpness is not None:
            error_gate = self._compute_error_aware_gate(gate_z, x_d1, x_d2)
            error_gate = self._match_seq_len(error_gate, delta.size(1))
            delta = delta * error_gate

        # ------------------------------------------------------------------
        # 5. Re-scale correction back to original y_hat scale and add
        #    [개선 2] tanh으로 delta 크기 제한 + learnable scale
        # ------------------------------------------------------------------
        correction = torch.tanh(delta) * F.softplus(self.correction_scale) * stdev
        if self.use_sharpness_gate and sharpness is not None:
            sharpness_gate = self._compute_sharpness_gate(sharpness)
            sharpness_gate = self._match_seq_len(sharpness_gate, correction.size(1))
            correction = sharpness_gate * correction
        calibrated = y_hat + correction

        if not return_params:
            return calibrated

        aux = {
            "z": z,
            "gate_z": gate_z,
            "spline_res": spline_res,
            "correction": correction,
            "context_mean": means,
            "context_std": stdev,
            "spline_range": F.softplus(self.spline.range_param).detach(),
            "spline_weight": self.spline.weight,
            "spline_type": self.spline_type,
            "correction_scale": F.softplus(self.correction_scale).detach(),
            "knot_offsets": self.spline.knot_offsets.detach() if getattr(self.spline, "knot_offsets", None) is not None else None,
            "sharpness": sharpness,
            "sharpness_gate": sharpness_gate,
            "error_gate": error_gate,
            "sharp_residual": sharp_residual,
            "x_d1": x_d1,
            "x_d2": x_d2,
            "context_spline_weight_delta": None if context_spline is None else context_spline["weight_delta"],
            "context_spline_range_delta": None if context_spline is None else context_spline["range_delta"],
            "context_spline_knot_delta": None if context_spline is None else context_spline["knot_offset_delta"],
            "context_spline_z": None if context_spline is None else context_spline["context_z"],
            "context_spline_d1": None if context_spline is None else context_spline["context_d1"],
            "context_spline_d2": None if context_spline is None else context_spline["context_d2"],
            "context_spline_attention_map": None if context_spline is None else context_spline["attention_map"],
        }
        return calibrated, aux


# Backward-compatible alias
SharpFluctuationCalibrator = ValueSplineCalibrator


class PostHocCalibration(nn.Module):
    """Plug-in calibration wrapper for the SOM (SCINet) path.

    Typical integration with Model in scinet.py:

        # Inside Model.__init__:
        self.calibrator = PostHocCalibration(num_channels=configs.enc_in, ...)

        # Inside Model.forecast:
        raw = self._forecast_raw(x_enc)          # existing SCINet output
        return self.calibrator(raw, x_context=x_enc)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.calibrator = ValueSplineCalibrator(**kwargs)

    def forward(self, y_hat, x_context=None, return_params=False):
        return self.calibrator(y_hat, x_context=x_context, return_params=return_params)


class Model(PostHocCalibration):
    pass