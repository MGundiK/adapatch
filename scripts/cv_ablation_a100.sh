#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Cross-Variable Mixing v2 — A100 80GB Colab Edition
# ═══════════════════════════════════════════════════════════════
#
# Batch size / LR scaling rationale:
# ───────────────────────────────────
# Sigmoid LR is epoch-based → larger batch = fewer gradient steps/epoch
# Adam sqrt-scaling rule: lr_new = lr_old * sqrt(bs_new / bs_old)
#
#   Dataset      ref_bs → a100_bs   ref_lr → a100_lr   steps/epoch
#   ──────────   ─────────────────   ────────────────   ───────────
#   Traffic      192  → 512         0.005  → 0.008     ~24
#   Electricity  512  → 1024        0.005  → 0.007     ~18
#   Weather      2048 → 4096        0.0005 → 0.0007    ~9
#   ETTh1        2048 → 4096        0.0005 → 0.0007    ~2 (tiny dataset)
#
# Traffic at bs=512: 862 vars × 512 batch × 96 seq → ~40GB fwd+bwd
# with CV mixing buffers (B,862,192): ~0.7GB extra — fits A100 80GB
# ═══════════════════════════════════════════════════════════════

set -e  # Exit on error

# ─── Setup ──────────────────────────────────────────────────
if [ ! -d "./dataset" ]; then
    echo "ERROR: ./dataset directory not found."
    echo "Please upload datasets first (see setup cell)."
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

if [ ! -d "./logs" ]; then mkdir -p ./logs; fi
if [ ! -d "./logs/cv_v2_a100" ]; then mkdir -p ./logs/cv_v2_a100; fi
if [ ! -d "./checkpoints" ]; then mkdir -p ./checkpoints; fi

model_name=AdaPatch

# ─── Shared config ──────────────────────────────────────────
COMMON="--is_training 1 --model $model_name --features M --seq_len 96 \
        --des CVv2_A100 --itr 1 --lradj sigmoid \
        --alpha 0.3 --alpha_lr_mult 50 \
        --train_epochs 100 --patience 10"

# ─── Run wrapper with timing ────────────────────────────────
run_one() {
    local dataset=$1 pred_len=$2 tag=$3
    shift 3
    local logfile="logs/cv_v2_a100/${tag}_${dataset}_96_${pred_len}.log"
    
    echo -n "  ${dataset} T=${pred_len} [${tag}] ... "
    local t0=$(date +%s)
    
    python -u run.py $COMMON \
        --pred_len $pred_len \
        --model_id "${dataset}_${pred_len}_${tag}" \
        "$@" > "$logfile" 2>&1
    
    local t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    local mse=$(grep -oP 'mse:\K[0-9.]+' "$logfile" | tail -1)
    local mae=$(grep -oP 'mae:\K[0-9.]+' "$logfile" | tail -1)
    echo "MSE=${mse:-FAIL}  MAE=${mae:-FAIL}  (${elapsed}s)"
}

# ═══════════════════════════════════════════════════════════════
# TRAFFIC (862 vars) — A100: bs=512, lr=0.008
# ═══════════════════════════════════════════════════════════════
TRAFFIC_ARGS="--root_path ./dataset/ --data_path traffic.csv \
              --data custom --enc_in 862 \
              --batch_size 512 --learning_rate 0.008 --num_workers 4"

run_traffic() {
    local tag=$1; shift
    for pl in 96 336; do
        run_one traffic $pl "$tag" $TRAFFIC_ARGS "$@"
    done
}

# ═══════════════════════════════════════════════════════════════
# ELECTRICITY (321 vars) — A100: bs=1024, lr=0.007
# ═══════════════════════════════════════════════════════════════
ELEC_ARGS="--root_path ./dataset/ --data_path electricity.csv \
           --data custom --enc_in 321 \
           --batch_size 1024 --learning_rate 0.007 --num_workers 4"

run_electricity() {
    local tag=$1; shift
    for pl in 96 336; do
        run_one electricity $pl "$tag" $ELEC_ARGS "$@"
    done
}

# ═══════════════════════════════════════════════════════════════
# WEATHER (21 vars) — A100: bs=4096, lr=0.0007
# ═══════════════════════════════════════════════════════════════
WEATHER_ARGS="--root_path ./dataset/ --data_path weather.csv \
              --data custom --enc_in 21 \
              --batch_size 4096 --learning_rate 0.0007 --num_workers 4"

run_weather() {
    local tag=$1; shift
    for pl in 96 336; do
        run_one weather $pl "$tag" $WEATHER_ARGS "$@"
    done
}

# ═══════════════════════════════════════════════════════════════
# ETTh1 (7 vars) — control — A100: bs=4096, lr=0.0007
# ═══════════════════════════════════════════════════════════════
ETTH1_ARGS="--root_path ./dataset/ --data_path ETTh1.csv \
            --data ETTh1 --enc_in 7 \
            --batch_size 4096 --learning_rate 0.0007 --num_workers 4"

run_etth1() {
    local tag=$1; shift
    for pl in 96 336; do
        run_one ETTh1 $pl "$tag" $ETTH1_ARGS "$@"
    done
}

# ═══════════════════════════════════════════════════════════════
# ABLATION
# ═══════════════════════════════════════════════════════════════
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  Cross-Variable Mixing v2 (on representations)       ║"
echo "║  A100 80GB — batch-optimized                          ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# ── Phase 1: Baseline (no mixing) ──────────────────────────
echo "━━━ [1/3] Baseline (no mixing) ━━━"
#run_traffic     "none"
#run_electricity "none"
#run_weather     "none"
run_etth1       "none"
echo ""

# ── Phase 2: MLP bottleneck mixing on representations ──────
echo "━━━ [2/3] MLP mixing on learned representations ━━━"
run_traffic     "mlp_r32" --cv_mixing mlp --cv_rank 32
run_electricity "mlp_r32" --cv_mixing mlp --cv_rank 32
run_weather     "mlp_r16" --cv_mixing mlp --cv_rank 16
run_etth1       "mlp_r8"  --cv_mixing mlp --cv_rank 8
echo ""

# ── Phase 3: Conv mixing on representations ────────────────
echo "━━━ [3/3] Conv mixing on learned representations ━━━"
run_traffic     "conv_k7" --cv_mixing conv --cv_kernel 7
run_electricity "conv_k7" --cv_mixing conv --cv_kernel 7
run_weather     "conv_k7" --cv_mixing conv --cv_kernel 7
run_etth1       "conv_k7" --cv_mixing conv --cv_kernel 7
echo ""

# ═══════════════════════════════════════════════════════════════
# RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  RESULTS                                              ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""
echo "Parsed MSE results:"
echo "─────────────────────────────────────────────────────────"

# Parse all results into a clean table
python3 -c "
import re, os, glob

results = {}
for f in sorted(glob.glob('logs/cv_v2_a100/*.log')):
    name = os.path.basename(f).replace('.log','')
    with open(f) as fh:
        content = fh.read()
    mse = re.findall(r'mse:([0-9.]+)', content)
    mae = re.findall(r'mae:([0-9.]+)', content)
    if mse:
        results[name] = {'mse': float(mse[-1]), 'mae': float(mae[-1])}

# Organize by dataset and horizon
datasets = ['traffic', 'electricity', 'weather', 'ETTh1']
horizons = [96, 336]
configs = ['none', 'mlp_r32', 'mlp_r16', 'mlp_r8', 'conv_k7']

# Header
print(f'{\"Config\":15s} | ', end='')
for ds in datasets:
    for h in horizons:
        print(f'{ds[:6]}_{h:3d} | ', end='')
print()
print('-' * 95)

# Rows
for cfg in configs:
    row = f'{cfg:15s} | '
    for ds in datasets:
        for h in horizons:
            key = f'{cfg}_{ds}_96_{h}'
            if key in results:
                mse = results[key]['mse']
                row += f'{mse:10.4f} | '
            else:
                row += f'{\"---\":>10s} | '
    print(row)

print()
print('TARGETS (from literature):')
print('  Traffic   T=96:  xPatch=0.471  iTransformer=0.395  (-16.1% gap)')
print('  Traffic   T=336: xPatch=0.501  iTransformer=0.433  (-13.6% gap)')
print('  Elec     T=96:  xPatch=0.159  CARD=0.141          (-11.3% gap)')
print('  Elec     T=336: xPatch=0.182  CARD=0.173          (-4.9% gap)')
print('  Weather  T=96:  xPatch=0.168  CARD=0.150          (-10.7% gap)')
print('  Weather  T=336: xPatch=0.236  CARD=0.260          (xPatch wins)')
print('  ETTh1    T=96:  xPatch=0.376  (CONTROL — no improvement expected)')
"

echo ""
echo "Done! Full logs in logs/cv_v2_a100/"
