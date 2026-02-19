import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='AdaPatch')

# ─── Basic config ────────────────────────────────────────────────
parser.add_argument('--is_training', type=int, required=True, default=1)
parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--model', type=str, required=True, default='AdaPatch')

# ─── Data loader ─────────────────────────────────────────────────
parser.add_argument('--data', type=str, required=True, default='ETTh1')
parser.add_argument('--root_path', type=str, default='./dataset')
parser.add_argument('--data_path', type=str, default='ETTh1.csv')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--train_only', type=bool, default=False)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
parser.add_argument('--embed', type=str, default='timeF')

# ─── Forecasting task ────────────────────────────────────────────
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--enc_in', type=int, default=7)

# ─── Seasonal stream (CNN) ───────────────────────────────────────
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--padding_patch', default='end')
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--n_blocks', type=int, default=1)
parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[3, 5, 7])

# ─── Trend stream (linear) ──────────────────────────────────────
parser.add_argument('--patch_len_trend', type=int, default=32)
parser.add_argument('--agg_kernel', type=int, default=5)

# ─── Learnable EMA decomposition ────────────────────────────────
parser.add_argument('--alpha', type=float, default=0.3)
parser.add_argument('--ema_reg_lambda', type=float, default=0.0)
parser.add_argument('--alpha_lr_mult', type=float, default=50.0,
    help='learning rate multiplier for EMA alpha params (default 50x base LR)')
parser.add_argument('--ema_backend', type=str, default='matrix')

# ─── Cross-variable gate ────────────────────────────────────────
parser.add_argument('--use_cross_variable', action='store_true', default=False)

# ─── Ablation flags ────────────────────────────────────────────
parser.add_argument('--no_multiscale', action='store_true', default=False,
    help='ablate: disable multiscale depthwise conv in seasonal stream')
parser.add_argument('--no_causal', action='store_true', default=False,
    help='ablate: disable dilated causal conv in seasonal stream')
parser.add_argument('--no_gated_fusion', action='store_true', default=False,
    help='ablate: replace gated fusion with concat+linear')
parser.add_argument('--no_agg_conv', action='store_true', default=False,
    help='ablate: disable aggregate conv in trend stream')

# ─── Optimization ───────────────────────────────────────────────
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--train_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--des', type=str, default='test')
parser.add_argument('--loss', type=str, default='arctan_mae')
parser.add_argument('--lradj', type=str, default='sigmoid')
parser.add_argument('--use_amp', action='store_true', default=False)
parser.add_argument('--revin', type=int, default=1)

# ─── GPU ─────────────────────────────────────────────────────────
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3')

args = parser.parse_args()
args.kernel_sizes = tuple(args.kernel_sizes)
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

# Convert ablation flags: --no_X → use_X = False
args.use_multiscale = not args.no_multiscale
args.use_causal = not args.no_causal
args.use_gated_fusion = not args.no_gated_fusion
args.use_agg_conv = not args.no_agg_conv

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nb{}_{}_{}'.format(
            args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_blocks, args.des, ii)

        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nb{}_{}_{}'.format(
        args.model_id, args.model, args.data, args.features,
        args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.n_blocks, args.des, ii)

    exp = Exp(args)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
