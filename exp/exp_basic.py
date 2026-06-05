import os
import torch
import torch.nn as nn
import importlib
import pkgutil  
from utils.sharp_calibration import PostHocCalibration
from utils.trend_calibration import PostHocTrendCalibration
from utils.signed_spline_calibration import PostHocSignedSplineCalibration


class BackboneWithCalibration(nn.Module):
    calibrator_cls = None
    debug_label = "CAL"

    def __init__(self, backbone, freeze_backbone=True, calibrator_kwargs=None, debug_grid=False):
        super(BackboneWithCalibration, self).__init__()
        if self.calibrator_cls is None:
            raise ValueError("calibrator_cls must be defined on the subclass.")
        self.backbone = backbone
        self.som = self.calibrator_cls(**(calibrator_kwargs or {}))
        self.freeze_backbone = freeze_backbone
        self.debug_grid = debug_grid
        self.last_debug_stats = None

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def train(self, mode=True):
        super(BackboneWithCalibration, self).train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _run_backbone(self, *args, **kwargs):
        if self.freeze_backbone:
            with torch.no_grad():
                return self.backbone(*args, **kwargs)
        return self.backbone(*args, **kwargs)

    def set_retrieval_memory(self, x_memory, y_memory):
        if hasattr(self.som, "set_retrieval_memory"):
            self.som.set_retrieval_memory(x_memory, y_memory)

    def forward_with_aux(self, *args, **kwargs):
        outputs = self._run_backbone(*args, **kwargs)
        if torch.is_tensor(outputs) and outputs.dim() == 3:
            x_context = args[0] if len(args) > 0 and torch.is_tensor(args[0]) and args[0].dim() == 3 else None
            calibrated, aux = self.som(outputs, x_context=x_context, return_params=True)
            return outputs, calibrated, aux
        return outputs, outputs, {}

    def forward(self, *args, **kwargs):
        outputs, calibrated, aux = self.forward_with_aux(*args, **kwargs)
        if self.debug_grid and aux:
            self._update_debug_stats(aux)
        return calibrated

    def _update_debug_stats(self, aux):
        with torch.no_grad():
            widths = aux.get("adaptive_widths")
            if widths is not None:
                stats = {
                    "width_mean": float(widths.mean().item()),
                    "width_std": float(widths.std().item()),
                    "width_min": float(widths.min().item()),
                    "width_max": float(widths.max().item()),
                }
                sharpness = aux.get("sharpness")
                if sharpness is not None:
                    stats["sharpness_mean"] = float(sharpness.mean().item())
                correction = aux.get("correction")
                if correction is not None:
                    stats["correction_abs_mean"] = float(correction.abs().mean().item())
                    stats["correction_abs_max"] = float(correction.abs().max().item())
                y_correction = aux.get("y_correction")
                if y_correction is not None:
                    stats["y_correction_abs_mean"] = float(y_correction.abs().mean().item())
                context_correction = aux.get("context_correction")
                if context_correction is not None:
                    stats["context_correction_abs_mean"] = float(context_correction.abs().mean().item())
                    stats["context_correction_abs_max"] = float(context_correction.abs().max().item())
                mlp_residual = aux.get("mlp_residual")
                if mlp_residual is not None:
                    stats["mlp_residual_mean"] = float(mlp_residual.mean().item())
                    stats["mlp_residual_abs_mean"] = float(mlp_residual.abs().mean().item())
                    stats["mlp_residual_abs_max"] = float(mlp_residual.abs().max().item())
                self.last_debug_stats = stats
                return

            y_global = aux.get("y_global")
            retrieval_beta = aux.get("retrieval_beta")
            stage2_residual = aux.get("stage2_residual")
            if y_global is not None or retrieval_beta is not None or stage2_residual is not None:
                stats = {}
                if y_global is not None:
                    stats["y_global_abs_mean"] = float(y_global.abs().mean().item())
                if retrieval_beta is not None:
                    stats["retrieval_beta_mean"] = float(retrieval_beta.mean().item())
                if stage2_residual is not None:
                    stats["stage2_residual_abs_mean"] = float(stage2_residual.abs().mean().item())
                    stats["stage2_residual_abs_max"] = float(stage2_residual.abs().max().item())
                correction = aux.get("correction")
                if correction is not None:
                    stats["correction_abs_mean"] = float(correction.abs().mean().item())
                    stats["correction_abs_max"] = float(correction.abs().max().item())
                self.last_debug_stats = stats
                return

            trend_delta = aux.get("trend_delta")
            future_trend = aux.get("future_trend")
            backbone_trend = aux.get("backbone_trend")
            trend_gain = aux.get("trend_gain")
            if trend_delta is None and future_trend is None and backbone_trend is None and trend_gain is None:
                return

            stats = {}
            if trend_gain is not None:
                stats["trend_gain"] = float(trend_gain.item() if trend_gain.numel() == 1 else trend_gain.mean().item())
            if trend_delta is not None:
                stats["trend_delta_abs_mean"] = float(trend_delta.abs().mean().item())
                stats["trend_delta_abs_max"] = float(trend_delta.abs().max().item())
            if future_trend is not None:
                stats["future_trend_mean"] = float(future_trend.mean().item())
            if backbone_trend is not None:
                stats["backbone_trend_mean"] = float(backbone_trend.mean().item())
            self.last_debug_stats = stats

    def pop_debug_stats(self):
        stats = self.last_debug_stats
        self.last_debug_stats = None
        return stats


class BackboneWithSOM(BackboneWithCalibration):
    calibrator_cls = PostHocCalibration
    debug_label = "SOM"


class BackboneWithTrend(BackboneWithCalibration):
    calibrator_cls = PostHocTrendCalibration
    debug_label = "TREND"


class BackboneWithSignedSpline(BackboneWithCalibration):
    calibrator_cls = PostHocSignedSplineCalibration
    debug_label = "SIGNED_SPLINE"

# Just put your model files under models/ folder
# e.g., models/Transformer.py, models/LSTM.py, etc.
# All models will be automatically detected and can be used by specifying their names.

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        
        # -------------------------------------------------------
        #  Automatically generate model map
        # -------------------------------------------------------
        model_map = self._scan_models_directory()

        # Use smart dictionary
        self.model_dict = LazyModelDict(model_map)

        self.device = self._acquire_device()
        model = self._build_model()
        if getattr(self.args, "use_som", False):
            model = self._load_pretrained_backbone(model)
            calibration_kwargs = self._build_som_kwargs()
            variant = getattr(self.args, "som_variant", "sharp")
            wrapper_map = {
                "sharp": BackboneWithSOM,
                "trend": BackboneWithTrend,
                "signed_spline": BackboneWithSignedSpline,
            }
            wrapper_cls = wrapper_map.get(variant, BackboneWithSOM)
            model = wrapper_cls(
                model,
                freeze_backbone=True,
                calibrator_kwargs=calibration_kwargs,
                debug_grid=getattr(self.args, "som_debug_grid", False),
            )
            print(f"{variant.upper()} calibration args: {calibration_kwargs}")
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            print(f"Using {variant.upper()} calibration: frozen backbone params={frozen}, trainable SOM params={trainable}")
        self.model = model.to(self.device)

    def _build_som_kwargs(self):
        features = getattr(self.args, "features", None)
        c_out = getattr(self.args, "c_out", None)
        num_channels = 1 if features == "MS" else c_out
        variant = getattr(self.args, "som_variant", "sharp")

        gamma_init_default = 0.0 if variant == "signed_spline" else -4.0
        gamma_init = getattr(self.args, "som_gamma_init", None)
        if gamma_init is None:
            gamma_init = gamma_init_default

        kwargs = {
            "num_knots": 8,
            "gamma_bound": getattr(self.args, "som_gamma_bound", 0.5),
            "beta_bound": getattr(self.args, "som_beta_bound", 0.25),
            "gamma_init": gamma_init,
            "beta_init": getattr(self.args, "som_beta_init", 0.0),
            "grid_width": 0.25,
            "learnable_grid": True,
            "moving_avg_kernel": getattr(self.args, "som_moving_avg_kernel", 3),
            "d2_weight": 1.0,
            "sharpness_temperature": 1.0,
            "adaptive_grid": True,
            "adaptive_grid_sharpness": 1.0,
            "adaptive_grid_min_scale": 0.5,
            "adaptive_grid_max_scale": 1.0,
            "use_input_context": True,
            "input_context_weight": 0.5,
            "use_mlp_head": getattr(self.args, "som_use_mlp_head", True),
            "residual_head_type": "mlp",
            "mlp_hidden_dim": 64,
            "mlp_num_layers": 2,
            "mlp_dropout": 0.0,
            "mlp_kernel_size": 5,
            "residual_sharp_boost": 0.0,
            "mlp_width_scale_bound": getattr(self.args, "som_mlp_width_scale_bound", 1.5),
            "horizon_decay_floor": getattr(self.args, "som_horizon_decay_floor", 1.0),
            "horizon_decay_power": getattr(self.args, "som_horizon_decay_power", 1.0),
            "channel_wise": True,
            "num_channels": num_channels,
            "eps": 1e-6,
        }

        if variant == "signed_spline":
            kwargs.update({
                "residual_head_type": "kan",
                "slope_bound": getattr(self.args, "som_slope_bound", 0.5),
                "bias_bound": getattr(self.args, "som_bias_bound", 0.1),
                "context_bound": getattr(self.args, "som_context_bound", 1.0),
                "slope_init": getattr(self.args, "som_slope_init", 0.0),
                "bias_init": getattr(self.args, "som_bias_init", 0.0),
                "retrieval_topk": getattr(self.args, "som_retrieval_topk", 10),
                "retrieval_temperature": getattr(self.args, "som_retrieval_temperature", 1.0),
                "retrieval_beta_init": getattr(self.args, "som_retrieval_beta_init", 0.1),
            })

        if variant == "trend":
            kwargs.update({
                "seq_len": getattr(self.args, "seq_len", None),
                "pred_len": getattr(self.args, "pred_len", None),
            })

        return kwargs

    def _load_pretrained_backbone(self, model):
        checkpoint_path = getattr(self.args, "current_backbone_checkpoint", None) or getattr(
            self.args, "backbone_checkpoint", None)
        if not checkpoint_path:
            raise ValueError("use_som=True requires a pretrained backbone checkpoint.")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Pretrained backbone checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained backbone checkpoint: {checkpoint_path}")
        return model

    def _scan_models_directory(self):
        """
        Automatically scan all .py files in the models folder
        """
        model_map = {}
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, 'models')

        # Iterate through all files in 'models' directory
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                # Ignore __init__.py and non-.py files
                if filename.endswith('.py') and filename != '__init__.py':
                    # Remove .py extension to get module name
                    module_name = filename[:-3]
                    
                    # Build full import path
                    full_path = f"models.{module_name}"
                    
                    # loading dict: {'Transformer': 'models.Transformer'}
                    model_map[module_name] = full_path
        
        return model_map

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda' and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps' and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            self.args.use_gpu = False
            self.args.use_multi_gpu = False
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


class LazyModelDict(dict):
    """
    Smart Lazy-Loading Dictionary
    """
    def __init__(self, model_map):
        self.model_map = model_map
        super().__init__()

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        
        if key not in self.model_map:
            raise NotImplementedError(f"Model [{key}] not found in 'models' directory.")
            
        module_path = self.model_map[key]
        try:
            print(f"🚀 Lazy Loading: {key} ...") 
            module = importlib.import_module(module_path)
        except ImportError as e:
            print(f"❌ Error: Failed to import model [{key}]. Dependencies missing?")
            raise e

        # Try to find the model class
        if hasattr(module, 'Model'):
            model_class = module.Model
        elif hasattr(module, key):
            model_class = getattr(module, key)
        else:
            raise AttributeError(f"Module {module_path} has no class 'Model' or '{key}'")

        self[key] = model_class
        return model_class

