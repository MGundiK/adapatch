#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Cross-Variable Mixing v3 — Depth / Width / Position Ablation
# A100 80GB Colab Edition
# ═══════════════════════════════════════════════════════════════
#
# Building on v2 results where MLP mixing on representations worked:
#   Traffic   T=96:  0.483 → 0.464 (−3.8%)
#   Elec      T=96:  0.163 → 0.152 (−6.6%)
#   Weather   T=96:  0.155 → 0.145 (−6.0%)
#
# Testing 3 axes:
#   1. Depth:   d=2 (stack 2 mixing layers, each with residual+LN)
#   2. Width:   r=64 (wider bottleneck, esp. for Traffic's 862 vars)
#   3. Position: post_pw (second mixer after pointwise conv)
#
# Skipping: baseline (none) and mlp_r32_d1 — already have these from v2.
# ═══════════════════════════════════════════════════════════════

if [ ! -d "./logs" ]; then mkdir -p ./logs; fi
if [ ! -d "./logs/cv_v3_a100" ]; then mkdir -p ./logs/cv_v3_a100; fi
if [ ! -d "./checkpoints" ]; then mkdir -p ./checkpoints; fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

model_name=AdaPatch

COMMON="--is_training 1 --model $model_name --features M --seq_len 96 \
        --des CVv3_A100 --itr 1 --lradj sigmoid \
        --alpha 0.3 --alpha_lr_mult 50 \
        --train_epochs 100 --patience 10"

run_one() {
    local dataset=$1 pred_len=$2 tag=$3
    shift 3
    local logfile="logs/cv_v3_a100/${tag}_${dataset}_96_${pred_len}.log"

    echo -n "  ${dataset} T=${pred_len} [${tag}] ... "
    local t0=$(date +%s)

    python -u run.py $COMMON \
        --pred_len $pred_len \
        --model_id "${dataset}_${pred_len}_${tag}" \
        "$@" > "$logfile" 2>&1

    local rc=$?
    local t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))

    if [ $rc -ne 0 ]; then
        echo "FAILED (exit code $rc, ${elapsed}s) — check $logfile"
        return
    fi

    local mse=$(grep -oP 'mse:\K[0-9.]+' "$logfile" | tail -1)
    local mae=$(grep -oP 'mae:\K[0-9.]+' "$logfile" | tail -1)
    echo "MSE=${mse:-FAIL}  MAE=${mae:-FAIL}  (${elapsed}s)"
}

# ═══════════════════════════════════════════════════════════════
# TRAFFIC (862 vars) — Main target, biggest gap to close
# ═══════════════════════════════════════════════════════════════
TRAFFIC="--root_path ./dataset/ --data_path traffic.csv \
         --data custom --enc_in 862 \
         --batch_size 512 --learning_rate 0.008 --num_workers 4"

run_traffic() {
    local tag=$1; shift
    for pl in 96 336; do
        run_one traffic $pl "$tag" $TRAFFIC "$@"
    done
}

# ═══════════════════════════════════════════════════════════════
# ELECTRICITY (321 vars)
# ═══════════════════════════════════════════════════════════════
ELEC="--root_path ./dataset/ --data_path electricity.csv \
      --data custom --enc_in 321 \
      --batch_size 1024 --learning_rate 0.007 --num_workers 4"

run_elec() {
    local tag=$1; shift
    for pl in 96 336; do
        run_one electricity $pl "$tag" $ELEC "$@"
    done
}

# ═══════════════════════════════════════════════════════════════
# WEATHER (21 vars) — Already beats CARD, light touch
# ═══════════════════════════════════════════════════════════════
WEATHER="--root_path ./dataset/ --data_path weather.csv \
         --data custom --enc_in 21 \
         --batch_size 4096 --learning_rate 0.0007 --num_workers 4"

run_weather() {
    local tag=$1; shift
    for pl in 96 336; do
        run_one weather $pl "$tag" $WEATHER "$@"
    done
}

# ═══════════════════════════════════════════════════════════════
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  CV Mixing v3: Depth / Width / Position Ablation      ║"
echo "║  A100 80GB                                             ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# ── Traffic: 5 configs × 2 horizons = 10 runs ──────────────
echo "━━━ TRAFFIC (862v) ━━━"

echo "  [1/5] Deeper: r=32, depth=2"
run_traffic "d2_r32" \
    --cv_mixing mlp --cv_rank 32 --cv_depth 2

echo "  [2/5] Wider: r=64, depth=1"
run_traffic "d1_r64" \
    --cv_mixing mlp --cv_rank 64

echo "  [3/5] Wider+Deeper: r=64, depth=2"
run_traffic "d2_r64" \
    --cv_mixing mlp --cv_rank 64 --cv_depth 2

echo "  [4/5] Post-PW: r=32, depth=1, +post_pw"
run_traffic "d1_r32_ppw" \
    --cv_mixing mlp --cv_rank 32 --cv_post_pw

echo "  [5/5] Wider+Post-PW: r=64, depth=1, +post_pw"
run_traffic "d1_r64_ppw" \
    --cv_mixing mlp --cv_rank 64 --cv_post_pw

echo ""

# ── Electricity: 3 configs × 2 horizons = 6 runs ───────────
echo "━━━ ELECTRICITY (321v) ━━━"

echo "  [1/3] Deeper: r=32, depth=2"
run_elec "d2_r32" \
    --cv_mixing mlp --cv_rank 32 --cv_depth 2

echo "  [2/3] Wider: r=64, depth=1"
run_elec "d1_r64" \
    --cv_mixing mlp --cv_rank 64

echo "  [3/3] Post-PW: r=32, depth=1, +post_pw"
run_elec "d1_r32_ppw" \
    --cv_mixing mlp --cv_rank 32 --cv_post_pw

echo ""

# ── Weather: 2 configs × 2 horizons = 4 runs ───────────────
echo "━━━ WEATHER (21v) ━━━"

echo "  [1/2] Deeper: r=16, depth=2"
run_weather "d2_r16" \
    --cv_mixing mlp --cv_rank 16 --cv_depth 2

echo "  [2/2] Post-PW: r=16, depth=1, +post_pw"
run_weather "d1_r16_ppw" \
    --cv_mixing mlp --cv_rank 16 --cv_post_pw

echo ""

# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  RESULTS                                              ║"
echo "╚═══════════════════════════════════════════════════════╝"

python3 -c "
import re, os, glob

# Collect v2 + v3 results
results = {}
for pattern in ['logs/cv_v2_a100/*.log', 'logs/cv_v3_a100/*.log']:
    for f in sorted(glob.glob(pattern)):
        name = os.path.basename(f).replace('.log','')
        with open(f) as fh:
            content = fh.read()
        mse = re.findall(r'mse:([0-9.]+)', content)
        mae = re.findall(r'mae:([0-9.]+)', content)
        if mse:
            results[name] = float(mse[-1])

# Print by dataset
for ds, horizons in [('traffic', [96,336]), ('electricity', [96,336]), ('weather', [96,336])]:
    print(f'\n  {ds.upper()}:')
    print(f'  {\"Config\":25s}  T=96       T=336')
    print(f'  ' + '-' * 50)
    # Find all configs for this dataset
    configs = set()
    for key in results:
        if f'_{ds}_' in key:
            parts = key.rsplit(f'_{ds}_96_', 1)
            if parts: configs.add(parts[0])
    for cfg in sorted(configs):
        vals = []
        for h in horizons:
            key = f'{cfg}_{ds}_96_{h}'
            vals.append(f'{results[key]:.4f}' if key in results else '  ---  ')
        print(f'  {cfg:25s}  {vals[0]:>9s}  {vals[1]:>9s}')

print()
print('  TARGETS:')
print('  Traffic   T=96: v2=0.464  xPatch=0.471  iTransformer=0.395')
print('  Traffic   T=336: v2=0.497  xPatch=0.501  iTransformer=0.433')
print('  Elec      T=96: v2=0.152  xPatch=0.159  CARD=0.141')
print('  Elec      T=336: v2=0.177  xPatch=0.182  CARD=0.173')
print('  Weather   T=96: v2=0.145  xPatch=0.168  CARD=0.150')
print('  Weather   T=336: v2=0.232  xPatch=0.236  CARD=0.260')
"

echo ""
echo "Done! Logs in logs/cv_v3_a100/"
