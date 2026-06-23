import torch
import torch.nn as nn

from .sharp_calibration import SharpFluctuationCalibrator, finite_differences


class DerivativeGatedSplineCalibrator(SharpFluctuationCalibrator):
    """Spline SOM variant that suppresses value-only correction on sharp input regimes."""

    def __init__(
        self,
        sharpness_gate_floor=0.0,
        sharpness_gate_power=1.0,
        use_input_sharpness_gate=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sharpness_gate_floor < 0 or sharpness_gate_floor > 1:
            raise ValueError("sharpness_gate_floor must be in [0, 1].")
        if sharpness_gate_power <= 0:
            raise ValueError("sharpness_gate_power must be positive.")
        self.sharpness_gate_floor = float(sharpness_gate_floor)
        self.sharpness_gate_power = float(sharpness_gate_power)
        self.use_input_sharpness_gate = bool(use_input_sharpness_gate)

    def _compute_sharpness_gate(self, sharpness):
        gate = (1.0 - sharpness).clamp(0.0, 1.0)
        if self.sharpness_gate_power != 1.0:
            gate = gate.pow(self.sharpness_gate_power)
        if self.sharpness_gate_floor > 0.0:
            gate = self.sharpness_gate_floor + (1.0 - self.sharpness_gate_floor) * gate
        return gate

    def forward(self, y_hat, x_context=None, return_params=False):
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

        gate_source = x_sharpness if (self.use_input_sharpness_gate and x_sharpness is not None) else sharpness
        sharpness_gate = self._compute_sharpness_gate(gate_source)
        correction = correction * sharpness_gate

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
            "sharpness_gate": sharpness_gate,
            "correction": correction,
            "horizon_decay": horizon_decay,
        }
        return calibrated, aux


class DerivativeGatedSplineCalibration(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.calibrator = DerivativeGatedSplineCalibrator(**kwargs)

    def forward(self, y_hat, x_context=None, return_params=False):
        return self.calibrator(y_hat, x_context=x_context, return_params=return_params)


class Model(DerivativeGatedSplineCalibration):
    pass
