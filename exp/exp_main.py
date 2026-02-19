from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import AdaPatch
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import math

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self._build_arctan_weights()

    def _build_model(self):
        model = AdaPatch.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _build_arctan_weights(self):
        """Precompute arctangent loss scaling weights (xPatch innovation)."""
        ratio = np.array([
            -1 * math.atan(i + 1) + math.pi / 4 + 1
            for i in range(self.args.pred_len)
        ])
        self.ratio = torch.tensor(ratio, dtype=torch.float32).unsqueeze(-1)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # Separate param groups: alpha_raw gets a much higher LR
        # so the decomposition can actually adapt per-variable.
        # Default base_lr=0.0005 → alpha gets 0.0005*50 = 0.025
        alpha_lr_mult = getattr(self.args, 'alpha_lr_mult', 50)
        
        # Find alpha_raw parameters
        alpha_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'alpha_raw' in name:
                alpha_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': other_params, 'lr': self.args.learning_rate, 'lr_scale': 1.0},
            {'params': alpha_params, 'lr': self.args.learning_rate * alpha_lr_mult,
             'lr_scale': alpha_lr_mult, 'weight_decay': 0.0},
        ]
        
        n_alpha = sum(p.numel() for p in alpha_params)
        n_other = sum(p.numel() for p in other_params)
        print(f"Optimizer: {n_other:,} params @ lr={self.args.learning_rate}, "
              f"{n_alpha} alpha params @ lr={self.args.learning_rate * alpha_lr_mult}")
        
        return optim.AdamW(param_groups)

    def _select_criterion(self):
        return nn.MSELoss(), nn.L1Loss()

    def _get_ratio(self):
        return self.ratio.to(self.device)

    def _get_model_attr(self, attr):
        """Helper for DataParallel compatibility."""
        if hasattr(self.model, attr):
            return getattr(self.model, attr)
        elif hasattr(self.model, 'module'):
            return getattr(self.model.module, attr)
        return None

    def vali(self, vali_data, vali_loader, criterion, is_test=True):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if not is_test:
                    ratio = self._get_ratio()
                    pred = outputs * ratio
                    true = batch_y * ratio
                else:
                    pred = outputs
                    true = batch_y

                loss = criterion(pred, true)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

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
        mse_criterion, mae_criterion = self._select_criterion()

        alpha_history = []

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

                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                # Arctangent-scaled MAE loss
                ratio = self._get_ratio()
                loss = mae_criterion(outputs * ratio, batch_y * ratio)

                # EMA alpha regularization
                get_reg = self._get_model_attr('get_ema_reg_loss')
                if get_reg is not None:
                    loss = loss + get_reg()

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, mae_criterion, is_test=False)
            test_loss = self.vali(test_data, test_loader, mse_criterion)

            # Log learned alpha values
            get_alpha = self._get_model_attr('get_alpha_values')
            if get_alpha is not None:
                alphas = get_alpha()
                alpha_history.append(alphas.copy())
                alpha_str = ', '.join([f'{a:.4f}' for a in alphas[:5]])
                if len(alphas) > 5:
                    alpha_str += f', ... ({len(alphas)} total)'
                print(f"  Learned alphas: [{alpha_str}]")

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        if alpha_history:
            np.save(os.path.join(path, 'alpha_history.npy'), np.array(alpha_history))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        os.remove(best_model_path)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

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

                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)

                if i % 20 == 0:
                    inp = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((inp[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((inp[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n\n')
        f.close()

        # Save final alphas
        get_alpha = self._get_model_attr('get_alpha_values')
        if get_alpha is not None:
            alphas = get_alpha()
            print(f"Final learned alphas: {alphas}")
            np.save(os.path.join(folder_path, 'final_alphas.npy'), alphas)

        return
