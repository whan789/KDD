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


class SharpFluctuationCalibrator(nn.Module):
    """
    Post-hoc calibration module for over-smoothed time-series forecasts.

    The module receives a backbone prediction y_hat with shape [B, L, D] and
    returns a calibrated prediction with the same shape. It uses first and
    second finite differences to detect local dynamic context, then maps the
    resulting sharpness score through knot-wise triangular basis functions.

    Normal regions can learn small gamma/beta values, while sharp regions can
    learn larger local correction parameters.
    """

    def __init__(
        self,
        num_knots=8,
        gamma_bound=0.5,
        beta_bound=0.25,
        gamma_init=-4.0,
        beta_init=0.0,
        grid_width=0.25,
        learnable_grid=True,
        moving_avg_kernel=3,
        d2_weight=1.0,
        sharpness_temperature=1.0,
        adaptive_grid=True,
        adaptive_grid_sharpness=1.0,
        adaptive_grid_min_scale=0.5,
        adaptive_grid_max_scale=1.0,
        use_input_context=True,
        input_context_weight=0.5,
        use_mlp_head=True,
        residual_head_type="mlp",
        mlp_hidden_dim=64,
        mlp_num_layers=2,
        mlp_dropout=0.0,
        mlp_kernel_size=5,
        residual_sharp_boost=0.0,
        mlp_width_scale_bound=1.0,
        horizon_decay_floor=1.0,
        horizon_decay_power=1.0,
        use_dynamics_gate=False,
        dynamics_gate_init=0.5,
        dynamics_gate_floor=0.0,
        channel_wise=False,
        num_channels=None,
        eps=1e-6,
    ):
        super(SharpFluctuationCalibrator, self).__init__()

        if num_knots < 2:
            raise ValueError("num_knots must be >= 2.")
        if grid_width <= 0:
            raise ValueError("grid_width must be positive.")
        if moving_avg_kernel < 1 or moving_avg_kernel % 2 == 0:
            raise ValueError("moving_avg_kernel must be a positive odd integer.")
        if adaptive_grid_sharpness < 0:
            raise ValueError("adaptive_grid_sharpness must be >= 0.")
        if adaptive_grid_min_scale <= 0 or adaptive_grid_max_scale <= 0:
            raise ValueError("adaptive_grid_min_scale and adaptive_grid_max_scale must be positive.")
        if adaptive_grid_min_scale > adaptive_grid_max_scale:
            raise ValueError("adaptive_grid_min_scale must be <= adaptive_grid_max_scale.")
        if input_context_weight < 0 or input_context_weight > 1:
            raise ValueError("input_context_weight must be in [0, 1].")
        if residual_head_type not in ("mlp", "conv"):
            raise ValueError("residual_head_type must be either mlp or conv.")
        if mlp_num_layers < 1:
            raise ValueError("mlp_num_layers must be >= 1.")
        if mlp_hidden_dim < 1:
            raise ValueError("mlp_hidden_dim must be >= 1.")
        if mlp_kernel_size < 1 or mlp_kernel_size % 2 == 0:
            raise ValueError("mlp_kernel_size must be a positive odd integer.")
        if residual_sharp_boost < 0:
            raise ValueError("residual_sharp_boost must be >= 0.")
        if mlp_width_scale_bound <= 0:
            raise ValueError("mlp_width_scale_bound must be positive.")
        if horizon_decay_floor <= 0 or horizon_decay_floor > 1:
            raise ValueError("horizon_decay_floor must be in (0, 1].")
        if horizon_decay_power <= 0:
            raise ValueError("horizon_decay_power must be positive.")
        if dynamics_gate_init <= 0 or dynamics_gate_init >= 1:
            raise ValueError("dynamics_gate_init must be in (0, 1).")
        if dynamics_gate_floor < 0 or dynamics_gate_floor >= 1:
            raise ValueError("dynamics_gate_floor must be in [0, 1).")
        if channel_wise and num_channels is None:
            raise ValueError("num_channels is required when channel_wise=True.")

        self.num_knots = num_knots
        self.gamma_bound = gamma_bound
        self.beta_bound = beta_bound
        self.moving_avg_kernel = moving_avg_kernel
        self.d2_weight = d2_weight
        self.sharpness_temperature = sharpness_temperature
        self.adaptive_grid = adaptive_grid
        self.adaptive_grid_sharpness = adaptive_grid_sharpness
        self.adaptive_grid_min_scale = adaptive_grid_min_scale
        self.adaptive_grid_max_scale = adaptive_grid_max_scale
        self.use_input_context = use_input_context
        self.input_context_weight = input_context_weight
        self.use_mlp_head = use_mlp_head
        self.residual_head_type = residual_head_type
        self.mlp_kernel_size = mlp_kernel_size
        self.residual_sharp_boost = residual_sharp_boost
        self.mlp_residual_bound = mlp_width_scale_bound
        self.horizon_decay_floor = horizon_decay_floor
        self.horizon_decay_power = horizon_decay_power
        self.horizon_decay_ref_len = 720
        self.horizon_decay_length_power = 1.5
        self.use_dynamics_gate = use_dynamics_gate
        self.dynamics_gate_floor = dynamics_gate_floor
        self.channel_wise = channel_wise
        self.num_channels = num_channels
        self.eps = eps

        centers = torch.linspace(0.0, 1.0, num_knots)
        self.register_buffer("knot_centers", centers)

        width = torch.full((num_knots,), float(grid_width))
        if learnable_grid:
            self.log_grid_width = nn.Parameter(torch.log(width))
        else:
            self.register_buffer("log_grid_width", torch.log(width))

        param_shape = (num_knots, num_channels) if channel_wise else (num_knots, 1)
        self.gamma_logits = nn.Parameter(torch.full(param_shape, float(gamma_init)))
        self.beta_logits = nn.Parameter(torch.full(param_shape, float(beta_init)))

        gate_shape = (num_channels,) if (channel_wise and num_channels is not None) else (1,)
        gate_bias = torch.full(gate_shape, torch.logit(torch.tensor(float(dynamics_gate_init))))
        self.dynamics_gate_bias = nn.Parameter(gate_bias)
        self.dynamics_gate_scale = nn.Parameter(torch.ones(gate_shape))

        # Residual head. The mlp mode reproduces the best point-wise residual setup.
        self.mlp_in_dim = 9
        mlp_layers = []
        in_dim = self.mlp_in_dim
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

    def _temporal_moving_average(self, x):
        if self.moving_avg_kernel == 1:
            return x

        padding = self.moving_avg_kernel // 2
        x_t = x.permute(0, 2, 1)
        x_t = F.pad(x_t, (padding, padding), mode="replicate")
        trend = F.avg_pool1d(x_t, kernel_size=self.moving_avg_kernel, stride=1)
        return trend.permute(0, 2, 1)

    def _sharpness_score(self, d1, d2):
        d1_scale = d1.abs().mean(dim=1, keepdim=True).detach() + self.eps
        d2_scale = d2.abs().mean(dim=1, keepdim=True).detach() + self.eps

        d1_norm = d1.abs() / d1_scale
        d2_norm = d2.abs() / d2_scale
        score = torch.sqrt(d1_norm.pow(2) + self.d2_weight * d2_norm.pow(2) + self.eps)

        temperature = max(float(self.sharpness_temperature), self.eps)
        return 1.0 - torch.exp(-score / temperature)

    def _resize_time_length(self, x, target_len):
        if x.size(1) == target_len:
            return x
        x_t = x.permute(0, 2, 1)
        x_t = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
        return x_t.permute(0, 2, 1)

    def _input_context_dynamics(self, target_len, x_context):
        if (not self.use_input_context) or x_context is None or x_context.dim() != 3:
            return None

        x_tail = x_context[:, -target_len:, :]
        x_tail = self._resize_time_length(x_tail, target_len)
        x_d1, x_d2 = finite_differences(x_tail)
        x_trend = self._temporal_moving_average(x_tail)
        x_local_deviation = x_tail - x_trend
        x_sharpness = self._sharpness_score(x_d1, x_d2)
        return {
            "x_sharpness": x_sharpness,
            "x_d1": x_d1,
            "x_d2": x_d2,
            "x_local_deviation": x_local_deviation,
        }

    def _merge_input_context_sharpness(self, y_sharpness, x_context, x_dynamics=None):
        if x_dynamics is None:
            x_dynamics = self._input_context_dynamics(y_sharpness.size(1), x_context)
        if x_dynamics is None:
            return y_sharpness, None
        x_sharpness = x_dynamics["x_sharpness"]

        weight = float(self.input_context_weight)
        merged = (1.0 - weight) * y_sharpness + weight * x_sharpness
        return merged, x_sharpness

    def _knot_basis(self, sharpness):
        centers = self.knot_centers.view(1, 1, 1, self.num_knots)
        base_widths = self.log_grid_width.exp().clamp_min(self.eps).view(1, 1, 1, self.num_knots)

        widths = base_widths
        if self.adaptive_grid:
            # Sharper regions use narrower local widths to form steeper local basis curves.
            width_scale = 1.0 / (1.0 + self.adaptive_grid_sharpness * sharpness.unsqueeze(-1))
            width_scale = width_scale.clamp(self.adaptive_grid_min_scale, self.adaptive_grid_max_scale)
            widths = (base_widths * width_scale).clamp_min(self.eps)

        basis = F.relu(1.0 - (sharpness.unsqueeze(-1) - centers).abs() / widths)
        basis_sum = basis.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return basis / basis_sum, widths

    def _compute_dynamics_gate(self, sharpness):
        gate_bias = self.dynamics_gate_bias.view(1, 1, -1)
        gate_scale = self.dynamics_gate_scale.view(1, 1, -1)
        gate_logits = gate_scale * sharpness + gate_bias
        gate = torch.sigmoid(gate_logits)
        if self.dynamics_gate_floor > 0.0:
            gate = self.dynamics_gate_floor + (1.0 - self.dynamics_gate_floor) * gate
        return gate

    def _predict_residual_with_mlp(self, y_hat, d1, d2, local_deviation, sharpness, x_dynamics):
        if x_dynamics is None:
            x_sharpness = torch.zeros_like(sharpness)
            x_d1 = torch.zeros_like(d1)
            x_d2 = torch.zeros_like(d2)
            x_local_deviation = torch.zeros_like(local_deviation)
        else:
            x_sharpness = x_dynamics["x_sharpness"]
            x_d1 = x_dynamics["x_d1"]
            x_d2 = x_dynamics["x_d2"]
            x_local_deviation = x_dynamics["x_local_deviation"]
        feats = torch.stack([
            y_hat, d1, d2, local_deviation, sharpness, x_sharpness,
            x_d1, x_d2, x_local_deviation,
        ], dim=-1)
        b, l, c, f = feats.shape
        if self.residual_head_type == "conv":
            conv_in = feats.permute(0, 2, 3, 1).reshape(b * c, f, l)
            residual_logits = self.mlp_head(conv_in).reshape(b, c, l).permute(0, 2, 1)
        else:
            residual_logits = self.mlp_head(feats.reshape(-1, f)).reshape(b, l, c)

        local_scale = (
            local_deviation.abs().detach()
            + d1.abs().detach()
            + d2.abs().detach()
        ) / 3.0
        global_scale = y_hat.detach().abs().mean(dim=1, keepdim=True).clamp_min(self.eps)
        sharp_gate = sharpness.detach()
        residual_scale = (local_scale + global_scale) * sharp_gate
        residual_scale = residual_scale * (1.0 + self.residual_sharp_boost * sharp_gate)
        return self.mlp_residual_bound * torch.tanh(residual_logits) * residual_scale

    def _project_params(self, basis, channels):
        if self.channel_wise and channels != self.num_channels:
            raise ValueError(
                "Input channel size does not match num_channels: "
                f"{channels} vs {self.num_channels}."
            )

        gamma_table = self.gamma_bound * torch.sigmoid(self.gamma_logits)
        beta_table = self.beta_bound * torch.tanh(self.beta_logits)

        if self.channel_wise:
            gamma = (basis * gamma_table.t().view(1, 1, channels, self.num_knots)).sum(dim=-1)
            beta = (basis * beta_table.t().view(1, 1, channels, self.num_knots)).sum(dim=-1)
        else:
            gamma = torch.einsum("blck,kd->blcd", basis, gamma_table)
            beta = torch.einsum("blck,kd->blcd", basis, beta_table)
            gamma = gamma.squeeze(-1)
            beta = beta.squeeze(-1)
        return gamma, beta

    def forward(self, y_hat, x_context=None, return_params=False):
        """
        Args:
            y_hat: Backbone prediction, shape [B, L, D].
            return_params: If True, also return gamma, beta, sharpness, d1, d2.

        Returns:
            calibrated: Calibrated prediction, shape [B, L, D].
            aux: Optional dictionary for analysis/debugging.
        """
        if y_hat.dim() != 3:
            raise ValueError("Expected y_hat shape [B, L, D].")

        _, _, channels = y_hat.shape
        d1, d2 = finite_differences(y_hat)
        y_sharpness = self._sharpness_score(d1, d2)
        x_dynamics = self._input_context_dynamics(y_sharpness.size(1), x_context)
        x_sharpness = x_dynamics["x_sharpness"] if x_dynamics is not None else None
        sharpness, _ = self._merge_input_context_sharpness(y_sharpness, x_context, x_dynamics=x_dynamics)

        trend = self._temporal_moving_average(y_hat)
        local_deviation = y_hat - trend

        mlp_residual = None
        if self.use_mlp_head:
            mlp_residual = self._predict_residual_with_mlp(
                y_hat, d1, d2, local_deviation, sharpness, x_dynamics
            )

        basis, widths = self._knot_basis(sharpness)
        gamma, beta = self._project_params(basis, channels)
        correction = gamma * local_deviation + beta * d2
        if mlp_residual is not None:
            correction = correction + mlp_residual

        dynamics_gate = None
        if self.use_dynamics_gate:
            dynamics_gate = self._compute_dynamics_gate(sharpness)
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
        calibrated = y_hat + correction

        if not return_params:
            return calibrated

        aux = {
            "gamma": gamma,
            "beta": beta,
            "sharpness": sharpness,
            "y_sharpness": y_sharpness,
            "x_sharpness": x_sharpness,
            "x_d1": x_dynamics["x_d1"] if x_dynamics is not None else None,
            "x_d2": x_dynamics["x_d2"] if x_dynamics is not None else None,
            "x_local_deviation": x_dynamics["x_local_deviation"] if x_dynamics is not None else None,
            "d1": d1,
            "d2": d2,
            "knot_basis": basis,
            "adaptive_widths": widths,
            "mlp_residual": mlp_residual,
            "dynamics_gate": dynamics_gate,
            "horizon_decay": horizon_decay,
        }
        return calibrated, aux


class PostHocCalibration(nn.Module):
    """
    Thin wrapper name for plug-in use after any forecasting backbone.
    """

    def __init__(self, **kwargs):
        super(PostHocCalibration, self).__init__()
        self.calibrator = SharpFluctuationCalibrator(**kwargs)

    def forward(self, y_hat, x_context=None, return_params=False):
        return self.calibrator(y_hat, x_context=x_context, return_params=return_params)


class Model(PostHocCalibration):
    """Compatibility alias for code paths that expect a Model class."""

    pass
