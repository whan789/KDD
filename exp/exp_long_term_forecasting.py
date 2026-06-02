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
        if mae_weight <= 0.0:
            return weighted_mse
        weighted_mae = (err.abs() * weights).sum() / weights.sum().clamp_min(1e-6)
        return weighted_mse + mae_weight * weighted_mae

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

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

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
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = self._select_loss(outputs, batch_y, criterion)
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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            if getattr(self.args, "use_som", False) and getattr(self.args, "som_debug_grid", False) and hasattr(self.model, "pop_debug_stats"):
                debug_stats = self.model.pop_debug_stats()
                if debug_stats:
                    print("SOM grid stats | width mean:{:.6f} std:{:.6f} min:{:.6f} max:{:.6f} sharp mean:{:.6f}".format(
                        debug_stats.get("width_mean", 0.0),
                        debug_stats.get("width_std", 0.0),
                        debug_stats.get("width_min", 0.0),
                        debug_stats.get("width_max", 0.0),
                        debug_stats.get("sharpness_mean", 0.0),
                    ))
                    if "mlp_residual_abs_mean" in debug_stats:
                        print("SOM mlp residual | mean:{:.6f} abs_mean:{:.6f} abs_max:{:.6f}".format(
                            debug_stats.get("mlp_residual_mean", 0.0),
                            debug_stats.get("mlp_residual_abs_mean", 0.0),
                            debug_stats.get("mlp_residual_abs_max", 0.0),
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

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if sharp_mae is not None and sharp_mse is not None:
            np.save(folder_path + 'sharp_metrics.npy', np.array([sharp_mae, sharp_mse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        results = {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'mspe': mspe}
        if sharp_mae is not None and sharp_mse is not None:
            results['sharp_mae'] = sharp_mae
            results['sharp_mse'] = sharp_mse
        return results
