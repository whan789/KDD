import os
import torch
import torch.nn as nn
import importlib
import pkgutil  
from utils.sharp_calibration import PostHocCalibration


class BackboneWithSOM(nn.Module):
    def __init__(self, backbone, freeze_backbone=True, som_kwargs=None, debug_grid=False):
        super(BackboneWithSOM, self).__init__()
        self.backbone = backbone
        self.som = PostHocCalibration(**(som_kwargs or {}))
        self.freeze_backbone = freeze_backbone
        self.debug_grid = debug_grid
        self.last_debug_stats = None

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def train(self, mode=True):
        super(BackboneWithSOM, self).train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _run_backbone(self, *args, **kwargs):
        if self.freeze_backbone:
            with torch.no_grad():
                return self.backbone(*args, **kwargs)
        return self.backbone(*args, **kwargs)

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
            if widths is None:
                return
            stats = {
                "width_mean": float(widths.mean().item()),
                "width_std": float(widths.std().item()),
                "width_min": float(widths.min().item()),
                "width_max": float(widths.max().item()),
            }
            sharpness = aux.get("sharpness")
            if sharpness is not None:
                stats["sharpness_mean"] = float(sharpness.mean().item())
            mlp_residual = aux.get("mlp_residual")
            if mlp_residual is not None:
                stats["mlp_residual_mean"] = float(mlp_residual.mean().item())
                stats["mlp_residual_abs_mean"] = float(mlp_residual.abs().mean().item())
                stats["mlp_residual_abs_max"] = float(mlp_residual.abs().max().item())
            self.last_debug_stats = stats

    def pop_debug_stats(self):
        stats = self.last_debug_stats
        self.last_debug_stats = None
        return stats

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
            som_kwargs = self._build_som_kwargs()
            model = BackboneWithSOM(model, freeze_backbone=True, som_kwargs=som_kwargs, debug_grid=getattr(self.args, "som_debug_grid", False))
            print(f"SOM calibration args: {som_kwargs}")
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            print(f"Using SOM calibration: frozen backbone params={frozen}, trainable SOM params={trainable}")
        self.model = model.to(self.device)

    def _build_som_kwargs(self):
        features = getattr(self.args, "features", None)
        c_out = getattr(self.args, "c_out", None)
        num_channels = 1 if features == "MS" else c_out

        return {
            "num_knots": 8,
            "gamma_bound": 0.5,
            "beta_bound": 0.25,
            "grid_width": 0.25,
            "learnable_grid": True,
            "moving_avg_kernel": 3,
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

