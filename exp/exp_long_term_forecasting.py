from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)


    def _sharp_subset_metric(self, preds, trues):
        ratio = float(getattr(self.args, "som_sharp_eval_ratio", 0.2))
        ratio = min(max(ratio, 0.0), 1.0)
        if ratio <= 0.0:
            return None, None

        true_t = torch.from_numpy(trues).float()
        pred_t = torch.from_numpy(preds).float()

        d1 = torch.zeros_like(true_t)
        d1[:, 1:, :] = true_t[:, 1:, :] - true_t[:, :-1, :]
        d2 = torch.zeros_like(true_t)
        d2[:, 1:, :] = d1[:, 1:, :] - d1[:, :-1, :]

        sharp = d2.abs().reshape(-1)
        if sharp.numel() == 0:
            return None, None

        k = max(1, int(sharp.numel() * ratio))
        threshold = torch.topk(sharp, k, largest=True).values.min()
        mask = d2.abs() >= threshold

        abs_err = (pred_t - true_t).abs()
        sq_err = (pred_t - true_t).pow(2)

        sharp_mae = abs_err[mask].mean().item() if mask.any() else None
        sharp_mse = sq_err[mask].mean().item() if mask.any() else None
        return sharp_mae, sharp_mse

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def _build_som_retrieval_memory(self, train_loader):
        if not getattr(self.args, "use_som", False):
            return
        if getattr(self.args, "som_variant", "sharp") != "signed_spline":
            return
        if not hasattr(self.model, "set_retrieval_memory"):
            return

        max_samples = int(getattr(self.args, "som_retrieval_max_samples", 4096))
        xs = []
        ys = []
        total = 0
        f_dim = -1 if self.args.features == 'MS' else 0
        for batch_x, batch_y, _, _ in train_loader:
            x_mem = batch_x.float()
            y_mem = batch_y.float()[:, -self.args.pred_len:, f_dim:]
            remaining = max_samples - total
            if remaining <= 0:
                break
            if x_mem.size(0) > remaining:
                x_mem = x_mem[:remaining]
                y_mem = y_mem[:remaining]
            xs.append(x_mem)
            ys.append(y_mem)
            total += x_mem.size(0)

        if not xs:
            print("SIGNED_SPLINE retrieval memory skipped: no train samples were collected.")
            return

        x_memory = torch.cat(xs, dim=0).to(self.device)
        y_memory = torch.cat(ys, dim=0).to(self.device)
        self.model.set_retrieval_memory(x_memory, y_memory)
        print("SIGNED_SPLINE retrieval memory: x={} y={}".format(tuple(x_memory.shape), tuple(y_memory.shape)))



    def _initialize_som_kmeans_codebook(self, train_loader):
        if not getattr(self.args, "use_som", False):
            return
        if getattr(self.args, "som_variant", "sharp") != "prototype_derivative":
            return
        if not getattr(self.args, "som_kmeans_init", False):
            return

        som_model = self.model.module if hasattr(self.model, "module") else self.model
        som = getattr(som_model, "som", None)
        if som is None or not hasattr(som, "encode_derivative_latents"):
            print("PROTOTYPE_DERIVATIVE k-means init skipped: calibrator does not expose latent encoder.")
            return
        if not hasattr(som, "initialize_prototypes_kmeans"):
            print("PROTOTYPE_DERIVATIVE k-means init skipped: calibrator does not expose k-means initializer.")
            return

        max_tokens = int(getattr(self.args, "som_kmeans_max_tokens", 20000))
        num_iters = int(getattr(self.args, "som_kmeans_iters", 50))
        if max_tokens <= 0:
            print("PROTOTYPE_DERIVATIVE k-means init skipped: som_kmeans_max_tokens <= 0.")
            return

        was_training = self.model.training
        self.model.eval()
        latents = []
        total = 0
        with torch.no_grad():
            for batch_x, _, _, _ in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_latents = som.encode_derivative_latents(batch_x)
                remaining = max_tokens - total
                if remaining <= 0:
                    break
                if batch_latents.size(0) > remaining:
                    indices = torch.randperm(batch_latents.size(0), device=batch_latents.device)[:remaining]
                    batch_latents = batch_latents[indices]
                latents.append(batch_latents.detach().cpu())
                total += batch_latents.size(0)
                if total >= max_tokens:
                    break
        if was_training:
            self.model.train()

        if not latents:
            print("PROTOTYPE_DERIVATIVE k-means init skipped: no derivative latents were collected.")
            return

        latents = torch.cat(latents, dim=0).to(self.device)
        stats = som.initialize_prototypes_kmeans(latents, num_iters=num_iters)
        if stats.get("initialized", False):
            print(
                "PROTOTYPE_DERIVATIVE k-means init: tokens={} iters={} usage_min={:.4f} usage_max={:.4f}".format(
                    stats["num_latents"], stats["num_iters"], stats["usage_min"], stats["usage_max"]
                )
            )
        else:
            print("PROTOTYPE_DERIVATIVE k-means init skipped: {}".format(stats.get("reason", "unknown reason")))

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = self._select_loss(pred, true, criterion)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def _select_som_aux_loss(self):
        if not getattr(self.args, "use_som", False):
            return None
        if getattr(self.args, "som_variant", "sharp") != "prototype_derivative":
            return None

        som_model = self.model.module if hasattr(self.model, "module") else self.model
        aux = getattr(som_model, "last_aux", None)
        if not aux:
            return None

        loss = None
        if getattr(self.args, "som_prototype_mode", "cosine") == "vq_vae":
            vq_weight = float(getattr(self.args, "som_codebook_loss_weight", 0.01))
            vq_loss = aux.get("vq_loss")
            if vq_weight > 0.0 and vq_loss is not None:
                loss = vq_weight * vq_loss

        recon_weight = float(getattr(self.args, "som_reconstruction_loss_weight", 0.01))
        reconstruction_loss = aux.get("reconstruction_loss")
        if recon_weight > 0.0 and reconstruction_loss is not None:
            recon_term = recon_weight * reconstruction_loss
            loss = recon_term if loss is None else loss + recon_term

        return loss

    def _select_loss(self, outputs, targets, criterion):
        alpha = float(getattr(self.args, "som_sharp_loss_alpha", 0.5))
        mae_weight = float(getattr(self.args, "som_loss_mae_weight", 0.0))
        if not getattr(self.args, "use_som", False):
            return criterion(outputs, targets)
        if alpha <= 0.0 and mae_weight <= 0.0:
            return criterion(outputs, targets)

        d1 = torch.zeros_like(targets)
        d1[:, 1:, :] = targets[:, 1:, :] - targets[:, :-1, :]
        d2 = torch.zeros_like(targets)
        d2[:, 1:, :] = d1[:, 1:, :] - d1[:, :-1, :]

        sharp = d2.abs().detach()
        sharp_scale = sharp.mean(dim=1, keepdim=True).clamp_min(1e-6)
        sharp_score = sharp / sharp_scale
        max_weight = float(getattr(self.args, "som_sharp_loss_max_weight", 5.0))
        weights = (1.0 + alpha * sharp_score).clamp(max=max_weight)

        err = outputs - targets
        weighted_mse = (err.pow(2) * weights).sum() / weights.sum().clamp_min(1e-6)
        loss = weighted_mse
        if mae_weight > 0.0:
            weighted_mae = (err.abs() * weights).sum() / weights.sum().clamp_min(1e-6)
            loss = loss + mae_weight * weighted_mae

        return loss

    def _collect_prototype_epoch_stats(self, epoch_stats):
        if not getattr(self.args, "use_som", False):
            return
        if getattr(self.args, "som_variant", "sharp") != "prototype_derivative":
            return

        som_model = self.model.module if hasattr(self.model, "module") else self.model
        aux = getattr(som_model, "last_aux", None)
        if not aux:
            return

        scalar_keys = (
            "reconstruction_loss",
            "vq_loss",
            "codebook_loss",
            "commitment_loss",
        )
        for key in scalar_keys:
            value = aux.get(key)
            if value is not None:
                epoch_stats[key].append(float(value.detach().item()))

        usage = aux.get("prototype_usage")
        if usage is not None:
            usage = usage.detach().float().reshape(-1)
            usage = usage / usage.sum().clamp_min(1e-8)
            entropy = -(usage * torch.log(usage.clamp_min(1e-8))).sum()
            epoch_stats["prototype_usage_entropy"].append(float(entropy.item()))
            epoch_stats["prototype_usage_perplexity"].append(float(torch.exp(entropy).item()))
            epoch_stats["prototype_usage_active"].append(float((usage > 1e-3).sum().item()))
            epoch_stats["prototype_usage_max"].append(float(usage.max().item()))

    def _summarize_prototype_epoch_stats(self, epoch_stats):
        summary = {}
        for key, values in epoch_stats.items():
            if values:
                summary[key] = float(np.mean(values))
        return summary

    def _format_epoch_delta(self, current, previous, digits=6):
        if current is None:
            return "n/a"
        if previous is None:
            return f"{current:.{digits}f}"
        delta = current - previous
        return f"{current:.{digits}f} ({delta:+.{digits}f})"

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        self._build_som_retrieval_memory(train_loader)
        self._initialize_som_kmeans_codebook(train_loader)
        time_now = time.time()

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        prev_prototype_summary = None
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            prototype_epoch_stats = {
                "reconstruction_loss": [],
                "vq_loss": [],
                "codebook_loss": [],
                "commitment_loss": [],
                "prototype_usage_entropy": [],
                "prototype_usage_perplexity": [],
                "prototype_usage_active": [],
                "prototype_usage_max": [],
            }

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = self._select_loss(outputs, batch_y, criterion)
                        aux_loss = self._select_som_aux_loss()
                        if aux_loss is not None:
                            loss = loss + aux_loss
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = self._select_loss(outputs, batch_y, criterion)
                    aux_loss = self._select_som_aux_loss()
                    if aux_loss is not None:
                        loss = loss + aux_loss
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                self._collect_prototype_epoch_stats(prototype_epoch_stats)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            if getattr(self.args, "use_som", False) and getattr(self.args, "som_debug_grid", False):
                if getattr(self.args, "som_variant", "sharp") == "prototype_derivative":
                    prototype_summary = self._summarize_prototype_epoch_stats(prototype_epoch_stats)
                    if prototype_summary:
                        print("Prototype epoch avg | recon:{} vq:{} codebook:{} commit:{}".format(
                            self._format_epoch_delta(prototype_summary.get("reconstruction_loss"), None if prev_prototype_summary is None else prev_prototype_summary.get("reconstruction_loss")),
                            self._format_epoch_delta(prototype_summary.get("vq_loss"), None if prev_prototype_summary is None else prev_prototype_summary.get("vq_loss")),
                            self._format_epoch_delta(prototype_summary.get("codebook_loss"), None if prev_prototype_summary is None else prev_prototype_summary.get("codebook_loss")),
                            self._format_epoch_delta(prototype_summary.get("commitment_loss"), None if prev_prototype_summary is None else prev_prototype_summary.get("commitment_loss")),
                        ))
                        if "prototype_usage_entropy" in prototype_summary:
                            print("Prototype usage avg | entropy:{} perplexity:{} active:{:.2f} top1:{}".format(
                                self._format_epoch_delta(prototype_summary.get("prototype_usage_entropy"), None if prev_prototype_summary is None else prev_prototype_summary.get("prototype_usage_entropy")),
                                self._format_epoch_delta(prototype_summary.get("prototype_usage_perplexity"), None if prev_prototype_summary is None else prev_prototype_summary.get("prototype_usage_perplexity"), digits=3),
                                prototype_summary.get("prototype_usage_active", 0.0),
                                self._format_epoch_delta(prototype_summary.get("prototype_usage_max"), None if prev_prototype_summary is None else prev_prototype_summary.get("prototype_usage_max")),
                            ))
                        prev_prototype_summary = prototype_summary
                elif hasattr(self.model, "pop_debug_stats"):
                    debug_stats = self.model.pop_debug_stats()
                    if debug_stats and "prototype_usage_entropy" in debug_stats:
                        print("Prototype usage | min:{:.6f} max:{:.6f} entropy:{:.6f} perplexity:{:.3f} active:{} top:{}".format(
                            debug_stats.get("prototype_usage_min", 0.0),
                            debug_stats.get("prototype_usage_max", 0.0),
                            debug_stats.get("prototype_usage_entropy", 0.0),
                            debug_stats.get("prototype_usage_perplexity", 0.0),
                            debug_stats.get("prototype_usage_active", 0),
                            debug_stats.get("prototype_usage_top", []),
                        ))
                        print("Prototype aux | recon:{:.6f} vq:{:.6f} codebook:{:.6f} commit:{:.6f}".format(
                            debug_stats.get("reconstruction_loss", 0.0),
                            debug_stats.get("vq_loss", 0.0),
                            debug_stats.get("codebook_loss", 0.0),
                            debug_stats.get("commitment_loss", 0.0),
                        ))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                if getattr(self.args, "use_som", False) and getattr(self.args, "som_plot_diagnostics", False) and i % 20 == 0:
                    som_model = self.model.module if hasattr(self.model, "module") else self.model
                    if hasattr(som_model, "forward_with_aux"):
                        backbone_raw, som_raw, aux = som_model.forward_with_aux(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        backbone_pred = backbone_raw[:, -self.args.pred_len:, :].detach().cpu().numpy()
                        som_pred = som_raw[:, -self.args.pred_len:, :].detach().cpu().numpy()

                        if test_data.scale and self.args.inverse:
                            shape = backbone_pred.shape
                            backbone_pred = test_data.inverse_transform(backbone_pred.reshape(shape[0] * shape[1], -1)).reshape(shape)
                            som_pred = test_data.inverse_transform(som_pred.reshape(shape[0] * shape[1], -1)).reshape(shape)

                        backbone_pred = backbone_pred[:, :, f_dim:]
                        som_pred = som_pred[:, :, f_dim:]
                        true_np = true

                        backbone_line = backbone_pred[0, :, -1]
                        som_line = som_pred[0, :, -1]
                        true_line = true_np[0, :, -1]

                        sharpness = aux.get("sharpness")
                        widths = aux.get("adaptive_widths")
                        mlp_residual = aux.get("mlp_residual")

                        sharp_line = None
                        width_line = None
                        mlp_line = None
                        if sharpness is not None:
                            sharp_line = sharpness[0].detach().cpu().mean(dim=-1).numpy()
                        if widths is not None:
                            width_line = widths[0].detach().cpu().mean(dim=(-1, -2)).numpy()
                        if mlp_residual is not None:
                            mlp_line = mlp_residual[0].detach().cpu().mean(dim=-1).numpy()

                        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                        axes[0].plot(true_line, label='GT', linewidth=2.0)
                        axes[0].plot(backbone_line, label='Backbone', linewidth=1.4)
                        axes[0].plot(som_line, label='SOM', linewidth=1.4)
                        axes[0].legend(loc='best')
                        axes[0].set_title('Prediction Comparison')

                        if sharp_line is not None:
                            axes[1].plot(sharp_line, label='Sharpness(mean over channels)', linewidth=1.2)
                        if width_line is not None:
                            axes[1].plot(width_line, label='Adaptive width(mean ch,knots)', linewidth=1.2)
                        if mlp_line is not None:
                            axes[1].plot(mlp_line, label='MLP residual(mean ch)', linewidth=1.2)
                        axes[1].legend(loc='best')
                        axes[1].set_title('SOM Internal Dynamics')
                        axes[1].set_xlabel('Prediction Time Index')

                        fig.tight_layout()
                        fig.savefig(os.path.join(folder_path, str(i) + '_som_diag.png'))
                        plt.close(fig)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        sharp_mae, sharp_mse = self._sharp_subset_metric(preds, trues)
        if sharp_mae is not None and sharp_mse is not None:
            print('mse:{}, mae:{}, sharp_mse:{}, sharp_mae:{}, dtw:{}'.format(mse, mae, sharp_mse, sharp_mae, dtw))
        else:
            print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        if sharp_mae is not None and sharp_mse is not None:
            f.write('mse:{}, mae:{}, sharp_mse:{}, sharp_mae:{}, dtw:{}'.format(mse, mae, sharp_mse, sharp_mae, dtw))
        else:
            f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        metric_values = [mse, mae]
        if sharp_mae is not None and sharp_mse is not None:
            metric_values.extend([sharp_mse, sharp_mae])
            np.save(folder_path + 'sharp_metrics.npy', np.array([sharp_mse, sharp_mae]))
        np.save(folder_path + 'metrics.npy', np.array(metric_values))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        results = {'mse': mse, 'mae': mae}
        if sharp_mae is not None and sharp_mse is not None:
            results['sharp_mse'] = sharp_mse
            results['sharp_mae'] = sharp_mae
        return results
