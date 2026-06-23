import argparse
import os
import torch
import torch.backends
from utils.print_args import print_args
import random
import numpy as np


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def set_random_seed(seed, use_cuda=False):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def resolve_workspace_path(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def resolve_default_root_path(data):
    dataset_roots = {
        'ETTh1': ('datasets', 'ETT-small'),
        'ETTh2': ('datasets', 'ETT-small'),
        'ETTm1': ('datasets', 'ETT-small'),
        'ETTm2': ('datasets', 'ETT-small'),
        'm4': ('datasets', 'm4'),
        'PSM': ('datasets', 'PSM'),
        'MSL': ('datasets', 'MSL'),
        'SMAP': ('datasets', 'SMAP'),
        'SMD': ('datasets', 'SMD'),
        'SWAT': ('datasets', 'SWAT'),
        'UEA': ('datasets',),
    }
    return resolve_workspace_path(*dataset_roots.get(data, ('datasets',)))


def resolve_dataset_tag(args):
    if getattr(args, 'data', None) != 'custom':
        return args.data

    data_path = getattr(args, 'data_path', '') or 'custom'
    data_name = os.path.splitext(os.path.basename(data_path))[0]
    sanitized = ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in data_name)
    return f'custom_{sanitized}'


def build_setting(args, ii, seed, use_som=None):
    som_flag = int(args.use_som if use_som is None else use_som)
    som_variant = getattr(args, 'som_variant', 'sharp')
    som_suffix = f'_{som_variant}' if som_flag else ''
    dataset_tag = resolve_dataset_tag(args)
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}_seed{}_som{}{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        dataset_tag,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        ii,
        seed,
        som_flag,
        som_suffix)

    if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
        setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{dataset_tag}_ft{args.features}'                 + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}'                 + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}'                 + f'_tvdt{int(args.tv_dt)}_tvB{int(args.tv_B)}_tvC{int(args.tv_C)}_useD{int(args.use_D)}_{args.des}_{ii}_seed{seed}_som{som_flag}{som_suffix}'
    return setting

def resolve_backbone_checkpoint(args, ii, seed):
    if args.backbone_checkpoint:
        return args.backbone_checkpoint.format(
            setting=build_setting(args, ii, seed, use_som=False),
            seed=seed,
            ii=ii)

    backbone_setting = build_setting(args, ii, seed, use_som=False)
    return os.path.join(args.checkpoints, backbone_setting, 'checkpoint.pth')


def empty_device_cache(args):
    if args.use_gpu:
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()


def print_metric_average(results, seeds):
    numeric_results = [result for result in results if isinstance(result, dict)]
    if not numeric_results:
        print('No numeric test metrics were returned, so seed averages could not be calculated.')
        return

    keys = [key for key, value in numeric_results[0].items() if isinstance(value, (int, float, np.floating))]
    print('Average metrics over seeds {}:'.format(','.join(map(str, seeds))))
    for key in keys:
        values = [float(result[key]) for result in numeric_results if key in result]
        print('{}:{:.4f}'.format(key, float(np.mean(values))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='DLinear',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='ETTm2', help='dataset type')
    parser.add_argument('--root_path', type=str, default=None, help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTm2.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default=None, help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--tv_dt', type=int, default=0, help='whether to use time variant dt for MambaSL')
    parser.add_argument('--tv_B', type=int, default=0, help='whether to use time variant B for MambaSL')
    parser.add_argument('--tv_C', type=int, default=0, help='whether to use time variant C for MambaSL')
    parser.add_argument('--use_D', type=int, default=0, help='whether to use D for MambaSL')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # SOM sharp fluctuation calibration
    parser.add_argument('--use_som', type=str2bool, nargs='?', const=True, default=False,
                        help='enable SOM calibration module; freezes backbone and trains only SOM')
    parser.add_argument('--som_variant', type=str, default='sharp', choices=['sharp', 'basic_spline', 'kan_lite_spline', 'patch_spline', 'trend', 'signed_spline', 'derivative', 'derivative_sharpening', 'prototype_derivative', 'directional_derivative', 'derivative_gated_spline'],
                        help='select the calibration variant used when --use_som is enabled')
    parser.add_argument('--som_lr', type=float, default=1e-3,
                        help='learning rate used when --use_som is enabled')
    parser.add_argument('--som_lradj', type=str, default='type1',
                        help='learning rate schedule used when --use_som is enabled')
    parser.add_argument('--som_moving_avg_kernel', type=int, default=3,
                        help='moving average kernel used by SOM/trend calibration; must be a positive odd integer')
    parser.add_argument('--som_gamma_bound', type=float, default=0.5,
                        help='upper bound for spline gamma correction')
    parser.add_argument('--som_beta_bound', type=float, default=0.25,
                        help='absolute bound for spline beta correction')
    parser.add_argument('--som_gamma_init', type=float, default=None,
                        help='initial raw value for spline gamma table; defaults to -4.0 for sharp and 0.0 for signed_spline')
    parser.add_argument('--som_beta_init', type=float, default=0.0,
                        help='initial raw value for spline beta table')
    parser.add_argument('--som_spline_init_range', type=float, default=0.3,
                        help='initial positive range scale used by the value-only spline SOM')
    parser.add_argument('--som_spline_max_range', type=float, default=5.0,
                        help='maximum channel-wise knot range used by the value-only spline SOM')
    parser.add_argument('--som_mixer_init_scale', type=float, default=0.1,
                        help='initial residual scaling used by the lightweight channel mixer in som_variant=kan_lite_spline')
    parser.add_argument('--som_patch_len', type=int, default=5,
                        help='temporal patch length used by som_variant=patch_spline')
    parser.add_argument('--som_patch_hidden_dim', type=int, default=64,
                        help='hidden width of the patch MLP used by som_variant=patch_spline')
    parser.add_argument('--som_patch_init_scale', type=float, default=0.1,
                        help='initial residual scaling used by the temporal patch branch in som_variant=patch_spline')
    parser.add_argument('--som_spline_type', type=str, default='linear', choices=['linear', 'quadratic', 'cubic'],
                        help='basis type for the sharp SOM spline; quadratic uses relu^2 and cubic uses relu^3 hinge bases')
    parser.add_argument('--som_learnable_knot_offsets', type=str2bool, nargs='?', const=True, default=False,
                        help='learn monotonic per-channel knot adjustments on top of the fixed base knot grid')
    parser.add_argument('--som_knot_offset_scale', type=float, default=1.0,
                        help='multiplier controlling the magnitude of learnable knot offsets')
    parser.add_argument('--som_use_sharpness_gate', type=str2bool, nargs='?', const=True, default=False,
                        help='suppress value-only spline correction in sharp regimes using a derivative-based gate')
    parser.add_argument('--som_use_error_aware_gate', type=str2bool, nargs='?', const=True, default=False,
                        help='learn an error-aware amplification gate from observed derivatives and supervise it with target errors')
    parser.add_argument('--som_error_gate_hidden_dim', type=int, default=16,
                        help='hidden width of the error-aware gate Conv1d branch in som_variant=sharp')
    parser.add_argument('--som_error_gate_kernel_size', type=int, default=3,
                        help='temporal kernel size of the error-aware gate Conv1d branch in som_variant=sharp')
    parser.add_argument('--som_error_gate_max_boost', type=float, default=1.0,
                        help='maximum multiplicative boost applied by the learned error-aware gate')
    parser.add_argument('--som_error_gate_temperature', type=float, default=1.0,
                        help='temperature used when mapping error-aware gate logits to a boost scale')
    parser.add_argument('--som_error_gate_loss_weight', type=float, default=0.0,
                        help='auxiliary loss weight for supervising the error-aware gate against target error patterns')
    parser.add_argument('--som_sharpness_gate_floor', type=float, default=0.0,
                        help='minimum spline residual fraction preserved after sharpness suppression in som_variant=sharp')
    parser.add_argument('--som_sharpness_gate_power', type=float, default=1.0,
                        help='power applied to the sharpness suppression gate in som_variant=sharp')
    parser.add_argument('--som_sharpness_d2_weight', type=float, default=1.0,
                        help='relative second-derivative weight when computing the sharpness gate in som_variant=sharp')
    parser.add_argument('--som_use_sharp_residual', type=str2bool, nargs='?', const=True, default=False,
                        help='add a sharp-region residual branch on top of the value-only spline SOM')
    parser.add_argument('--som_sharp_residual_hidden_dim', type=int, default=16,
                        help='hidden width of the sharp-region residual Conv1d branch in som_variant=sharp')
    parser.add_argument('--som_sharp_residual_kernel_size', type=int, default=3,
                        help='temporal kernel size of the sharp-region residual Conv1d branch in som_variant=sharp')
    parser.add_argument('--som_sharp_residual_scale', type=float, default=0.25,
                        help='output scale applied to the sharp-region residual branch in som_variant=sharp')
    parser.add_argument('--som_use_context_spline', type=str2bool, nargs='?', const=True, default=False,
                        help='condition sharp SOM spline parameters on x_enc so context patterns can steer correction shape')
    parser.add_argument('--som_context_spline_attention_dim', type=int, default=32,
                        help='attention width used by the x_enc-conditioned spline cross-attention encoder')
    parser.add_argument('--som_context_spline_num_heads', type=int, default=4,
                        help='number of attention heads used by the x_enc-conditioned spline cross-attention encoder')
    parser.add_argument('--som_context_spline_ff_hidden_dim', type=int, default=64,
                        help='feed-forward hidden width used after cross-attention in the x_enc-conditioned spline encoder')
    parser.add_argument('--som_context_spline_modulation_scale', type=float, default=0.5,
                        help='maximum tanh-scaled modulation strength applied to spline weights/ranges from x_enc context')
    parser.add_argument('--som_slope_bound', type=float, default=0.5,
                        help='absolute bound for signed_spline d1/slope correction')
    parser.add_argument('--som_bias_bound', type=float, default=0.1,
                        help='absolute bound for signed_spline level/bias correction')
    parser.add_argument('--som_context_bound', type=float, default=1.0,
                        help='multiplier for signed_spline input-context dynamics correction')
    parser.add_argument('--som_slope_init', type=float, default=0.0,
                        help='initial raw value for signed_spline d1/slope table')
    parser.add_argument('--som_bias_init', type=float, default=0.0,
                        help='initial raw value for signed_spline level/bias table')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44], help='random seeds to run and average')
    parser.add_argument('--backbone_checkpoint', type=str, default=None,
                        help='pretrained backbone checkpoint path for --use_som; defaults to the matching som0 setting')
    parser.add_argument('--som_use_mlp_head', type=str2bool, nargs='?', const=True, default=True,
                        help='use residual prediction head')
    parser.add_argument('--som_mlp_width_scale_bound', type=float, default=1.5,
                        help='bound multiplier for SOM residual head output')
    parser.add_argument('--som_retrieval_topk', type=int, default=10,
                        help='number of train instances retrieved for signed_spline global revision')
    parser.add_argument('--som_retrieval_temperature', type=float, default=1.0,
                        help='softmax temperature for signed_spline retrieval weights')
    parser.add_argument('--som_retrieval_beta_init', type=float, default=0.1,
                        help='initial gate value for signed_spline retrieval global revision')
    parser.add_argument('--som_retrieval_max_samples', type=int, default=4096,
                        help='maximum train windows stored in signed_spline retrieval memory')
    parser.add_argument('--som_debug_grid', type=str2bool, nargs='?', const=True, default=False,
                        help='print SOM grid and correction diagnostics during training')
    parser.add_argument('--som_horizon_decay_floor', type=float, default=1.0,
                        help='final-step multiplier for SOM correction; 1.0 disables horizon decay')
    parser.add_argument('--som_horizon_decay_power', type=float, default=1.0,
                        help='base shape of SOM horizon decay; larger values decay later steps more')
    parser.add_argument('--som_use_dynamics_gate', type=str2bool, nargs='?', const=True, default=False,
                        help='scale SOM correction with a d1/d2 sharpness gate')
    parser.add_argument('--som_dynamics_gate_init', type=float, default=0.5,
                        help='initial sigmoid output of the d1/d2 correction gate at zero sharpness')
    parser.add_argument('--som_dynamics_gate_floor', type=float, default=0.0,
                        help='minimum residual fraction preserved by the d1/d2 correction gate')
    parser.add_argument('--som_use_input_sharpness_gate', type=str2bool, nargs='?', const=True, default=True,
                        help='build the derivative suppression gate from input sharpness instead of output sharpness')
    parser.add_argument('--som_recent_len', type=int, default=12,
                        help='sliding-window length for prototype_derivative d1/d2 tokens')
    parser.add_argument('--som_segment_stride', type=int, default=None,
                        help='sliding-window stride for prototype_derivative; defaults to window length')
    parser.add_argument('--som_correction_scale', type=float, default=0.2,
                        help='scale multiplier for derivative-aware SOM correction')
    parser.add_argument('--som_num_knots', type=int, default=8,
                        help='number of spline knots used by SOM-style modulation')
    parser.add_argument('--som_d2_weight', type=float, default=1.0,
                        help='relative weight of second derivative when computing SOM sharpness')
    parser.add_argument('--som_sharpness_temperature', type=float, default=1.0,
                        help='temperature used to squash SOM sharpness scores')
    parser.add_argument('--som_sharpness_mode', type=str, default='heuristic', choices=['heuristic', 'heatmap_cnn'],
                        help='sharpness scorer used by derivative SOM modulation')
    parser.add_argument('--som_sharpness_hidden_dim', type=int, default=16,
                        help='hidden width of the learned heatmap CNN sharpness scorer')
    parser.add_argument('--som_sharpness_kernel_size', type=int, default=3,
                        help='temporal kernel size of the learned heatmap CNN sharpness scorer')
    parser.add_argument('--som_use_dilate_loss', type=str2bool, nargs='?', const=True, default=False,
                        help='add a DILATE-style shape/time auxiliary loss when training derivative SOM')
    parser.add_argument('--som_dilate_loss_weight', type=float, default=0.1,
                        help='global weight applied to the DILATE-style auxiliary loss')
    parser.add_argument('--som_dilate_shape_weight', type=float, default=1.0,
                        help='relative weight of the DILATE-style shape term')
    parser.add_argument('--som_dilate_time_weight', type=float, default=1.0,
                        help='relative weight of the DILATE-style time-distortion term')
    parser.add_argument('--som_dilate_d1_weight', type=float, default=0.5,
                        help='shape-term weight for first-derivative alignment inside the DILATE-style loss')
    parser.add_argument('--som_dilate_d2_weight', type=float, default=1.0,
                        help='shape-term weight for second-derivative alignment inside the DILATE-style loss')
    parser.add_argument('--som_dilate_time_temperature', type=float, default=0.25,
                        help='softmax temperature used to estimate event timing in the DILATE-style loss')
    parser.add_argument('--som_use_times2d_modulation', type=str2bool, nargs='?', const=True, default=False,
                        help='modulate Times2D derivative correction with a spline-style SOM map')
    parser.add_argument('--som_times2d_modulation_bound', type=float, default=0.5,
                        help='maximum residual scaling offset applied on top of Times2D correction')
    parser.add_argument('--som_times2d_modulation_init', type=float, default=0.0,
                        help='initial raw value for the Times2D modulation spline table')
    parser.add_argument('--som_correction_head_type', type=str, default='conv', choices=['conv', 'mlp'],
                        help='projection head used to map derivative features into prediction-horizon corrections')
    parser.add_argument('--som_correction_mlp_hidden_dim', type=int, default=128,
                        help='hidden width used when som_correction_head_type=mlp')
    parser.add_argument('--som_use_prediction_features', type=str2bool, nargs='?', const=True, default=False,
                        help='add a small y_hat-shape feature branch on top of the derivative correction head')
    parser.add_argument('--som_prediction_feature_hidden_dim', type=int, default=64,
                        help='hidden width used by the y_hat-shape feature branch when enabled')
    parser.add_argument('--som_num_prototypes', type=int, default=8,
                        help='number of local d1/d2 prototypes for prototype_derivative')
    parser.add_argument('--som_prototype_temperature', type=float, default=1.0,
                        help='soft assignment temperature for prototype_derivative prototypes')
    parser.add_argument('--som_prototype_mode', type=str, default='cosine', choices=['cosine', 'vq_vae'],
                        help='prototype assignment mode for prototype_derivative')
    parser.add_argument('--som_prototype_distance', type=str, default='l2', choices=['l2', 'cosine'],
                        help='distance function used to match derivative latents with prototypes')
    parser.add_argument('--som_codebook_loss_weight', type=float, default=0.01,
                        help='VQ auxiliary loss weight used only when som_prototype_mode=vq_vae')
    parser.add_argument('--som_reconstruction_loss_weight', type=float, default=0.01,
                        help='derivative-token reconstruction loss weight for prototype_derivative')
    parser.add_argument('--som_commitment_loss_weight', type=float, default=0.25,
                        help='commitment term weight inside the VQ auxiliary loss')
    parser.add_argument('--som_vq_ema_update', type=str2bool, nargs='?', const=True, default=False,
                        help='update prototype_derivative VQ codebook with EMA instead of codebook gradients')
    parser.add_argument('--som_vq_ema_decay', type=float, default=0.99,
                        help='EMA decay used for prototype_derivative VQ codebook updates')
    parser.add_argument('--som_vq_ema_eps', type=float, default=1e-5,
                        help='numerical epsilon used for prototype_derivative VQ EMA normalization')
    parser.add_argument('--som_prototype_latent_dim', type=int, default=None,
                        help='latent dimension for prototype_derivative token encoder; defaults to 2 * som_recent_len')
    parser.add_argument('--som_kmeans_init', type=str2bool, nargs='?', const=True, default=False,
                        help='initialize prototype_derivative codebook with one-time k-means over sampled train latents')
    parser.add_argument('--som_kmeans_max_tokens', type=int, default=20000,
                        help='maximum derivative latent tokens used for prototype_derivative k-means initialization')
    parser.add_argument('--som_kmeans_iters', type=int, default=50,
                        help='number of iterations for prototype_derivative k-means initialization')
    parser.add_argument('--som_gate_init', type=float, default=0.1,
                        help='initial sigmoid gate value for derivative-aware SOM')
    parser.add_argument('--som_derivative_hidden_dim', type=int, default=16,
                        help='hidden width per channel for som_variant=derivative_sharpening')
    parser.add_argument('--som_derivative_kernel_size', type=int, default=3,
                        help='temporal kernel size for som_variant=derivative_sharpening')
    parser.add_argument('--som_need_gate_init', type=float, default=0.1,
                        help='initial sigmoid value for the correction-need gate in derivative_sharpening')
    parser.add_argument('--som_need_gate_loss_weight', type=float, default=0.5,
                        help='auxiliary loss weight for matching predicted need gate to target-based correction need')
    parser.add_argument('--som_derivative_aux_d1_weight', type=float, default=0.1,
                        help='auxiliary loss weight for matching predicted d1(y_true) in som_variant=derivative_sharpening')
    parser.add_argument('--som_derivative_aux_d2_weight', type=float, default=0.1,
                        help='auxiliary loss weight for matching predicted d2(y_true) in som_variant=derivative_sharpening')
    parser.add_argument('--som_derivative_consistency_weight', type=float, default=0.05,
                        help='consistency loss weight between corrected output derivatives and derivative predictions')
    parser.add_argument('--som_output_d1_loss_weight', type=float, default=0.1,
                        help='sharpness-weighted loss weight for matching d1(corrected output) to d1(y_true)')
    parser.add_argument('--som_output_d2_loss_weight', type=float, default=0.05,
                        help='sharpness-weighted loss weight for matching d2(corrected output) to d2(y_true)')
    parser.add_argument('--som_output_d1_fft_loss_weight', type=float, default=0.05,
                        help='band-limited FFT magnitude loss weight for matching d1(corrected output) to d1(y_true)')
    parser.add_argument('--som_output_d1_fft_low_ratio', type=float, default=0.1,
                        help='lower FFT bin ratio used for band-limited d1 FFT loss in derivative_sharpening')
    parser.add_argument('--som_output_d1_fft_high_ratio', type=float, default=0.4,
                        help='upper FFT bin ratio used for band-limited d1 FFT loss in derivative_sharpening')
    parser.add_argument('--som_need_window', type=int, default=9,
                        help='odd local window length for target-based correction-need scoring in derivative_sharpening')
    parser.add_argument('--som_need_excursion_weight', type=float, default=1.0,
                        help='weight for local target range when computing correction-need score')
    parser.add_argument('--som_need_derivative_weight', type=float, default=0.5,
                        help='weight for local d1/d2 target energy when computing correction-need score')
    parser.add_argument('--som_need_d2_weight', type=float, default=0.5,
                        help='relative d2 contribution inside local derivative energy for correction-need score')
    parser.add_argument('--som_need_loss_alpha', type=float, default=0.5,
                        help='strength of correction-need weighting in derivative_sharpening loss; 0 disables it')
    parser.add_argument('--som_need_loss_max_weight', type=float, default=5.0,
                        help='maximum per-step loss weight from correction-need scoring')
    parser.add_argument('--som_need_output_loss_weight', type=float, default=1.0,
                        help='base output MSE weight in derivative_sharpening residual-target training')
    parser.add_argument('--som_residual_target_loss_weight', type=float, default=0.0,
                        help='weight for directly matching correction to y_true - backbone forecast')
    parser.add_argument('--som_residual_detail_loss_weight', type=float, default=0.5,
                        help='weight for matching high-frequency correction detail to high-frequency target residual')
    parser.add_argument('--som_residual_detail_window', type=int, default=25,
                        help='odd moving average window used to remove low-frequency residual detail')
    parser.add_argument('--som_correction_mean_loss_weight', type=float, default=0.1,
                        help='penalty weight for non-zero mean correction over the prediction horizon')
    parser.add_argument('--som_flat_correction_loss_weight', type=float, default=0.01,
                        help='regularization weight that keeps correction small in low-need regions')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    args = parser.parse_args()

    # Fixed SOM defaults used across experiments.
    args.som_mlp_width_scale_bound = getattr(args, 'som_mlp_width_scale_bound', 1.5)
    args.som_sharp_loss_alpha = 0.5
    args.som_sharp_loss_max_weight = 5.0
    args.som_sharp_eval_ratio = 0.2
    args.som_loss_mae_weight = getattr(args, 'som_loss_mae_weight', 0.0)

    if args.root_path is None:
        args.root_path = resolve_default_root_path(args.data)
    if args.checkpoints is None:
        args.checkpoints = resolve_workspace_path('checkpoints')
    if args.use_som:
        args.learning_rate = args.som_lr
        args.lradj = args.som_lradj

    set_random_seed(args.seeds[0], args.use_gpu and args.gpu_type == 'cuda')
    args.seed = args.seeds[0]

    if args.use_gpu and args.gpu_type == 'cuda' and torch.cuda.is_available():
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    elif args.use_gpu and args.gpu_type == 'mps' and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        args.device = torch.device("mps")
        print('Using mps')
    else:
        args.use_gpu = False
        args.use_multi_gpu = False
        args.device = torch.device("cpu")
        print('Using CPU')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        from exp.exp_imputation import Exp_Imputation
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        from exp.exp_classification import Exp_Classification
        Exp = Exp_Classification
    elif args.task_name == 'zero_shot_forecast':
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
        Exp = Exp_Zero_Shot_Forecast
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast

    test_results = []
    if args.is_training:
        for seed in args.seeds:
            args.seed = seed
            set_random_seed(seed, args.use_gpu and args.gpu_type == 'cuda')
            for ii in range(args.itr):
                setting = build_setting(args, ii, seed)
                if args.use_som:
                    args.current_backbone_checkpoint = resolve_backbone_checkpoint(args, ii, seed)
                    print('Loading pretrained backbone checkpoint for SOM: {}'.format(args.current_backbone_checkpoint))
                else:
                    args.current_backbone_checkpoint = None
                exp = Exp(args)

                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                test_results.append(exp.test(setting))
                empty_device_cache(args)
    else:
        for seed in args.seeds:
            args.seed = seed
            set_random_seed(seed, args.use_gpu and args.gpu_type == 'cuda')
            for ii in range(args.itr):
                setting = build_setting(args, ii, seed)
                if args.use_som:
                    args.current_backbone_checkpoint = resolve_backbone_checkpoint(args, ii, seed)
                    print('Loading pretrained backbone checkpoint for SOM: {}'.format(args.current_backbone_checkpoint))
                else:
                    args.current_backbone_checkpoint = None
                exp = Exp(args)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                test_results.append(exp.test(setting, test=1))
                empty_device_cache(args)

    print_metric_average(test_results, args.seeds)
