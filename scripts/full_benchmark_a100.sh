#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Full Benchmark — r32 d1 +post_pw (best balanced config)
# A100 80GB
# ═══════════════════════════════════════════════════════════════
#
# 10 datasets × 4 horizons = 40 settings
#
# Strategy:
#   High-var (Traffic 862v, Electricity 321v, Weather 21v, Solar 137v):
#     → cv_mixing=mlp, r=32, d=1, +post_pw
#   Low-var (ETTh1/h2/m1/m2 7v, Exchange 8v, ILI 7v):
#     → Run BOTH with and without mixing
#     → Report best, note threshold in paper
#
# Estimated runtime: ~18-24 hours on A100
# ═══════════════════════════════════════════════════════════════

if [ ! -d "./logs" ]; then mkdir -p ./logs; fi
if [ ! -d "./logs/full_bench" ]; then mkdir -p ./logs/full_bench; fi
if [ ! -d "./checkpoints" ]; then mkdir -p ./checkpoints; fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

model_name=AdaPatch

COMMON="--is_training 1 --model $model_name --features M --seq_len 96 \
        --des FullBench --itr 1 --lradj sigmoid \
        --alpha 0.3 --alpha_lr_mult 50 \
        --train_epochs 100 --patience 10 --num_workers 4"

# Cross-variable mixing args for high-var datasets
CV_ARGS="--cv_mixing mlp --cv_rank 32 --cv_post_pw"

run_one() {
    local dataset=$1 pred_len=$2 tag=$3
    shift 3
    local logfile="logs/full_bench/${tag}_${dataset}_${pred_len}.log"

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
        echo "FAILED (${elapsed}s) — check $logfile"
        return
    fi

    local mse=$(grep -oP 'mse:\K[0-9.]+' "$logfile" | tail -1)
    local mae=$(grep -oP 'mae:\K[0-9.]+' "$logfile" | tail -1)
    echo "MSE=${mse:-FAIL}  MAE=${mae:-FAIL}  (${elapsed}s)"
}

run_all_horizons() {
    local dataset=$1 tag=$2
    shift 2
    for pl in 96 192 336 720; do
        run_one "$dataset" $pl "$tag" "$@"
    done
}

run_ili_horizons() {
    local dataset=$1 tag=$2
    shift 2
    for pl in 24 36 48 60; do
        run_one "$dataset" $pl "$tag" "$@"
    done
}

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Full Benchmark: r=32, d=1, +post_pw                     ║"
echo "║  A100 80GB                                                ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════
# HIGH-VAR DATASETS — with cross-variable mixing
# ═══════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "HIGH-VAR DATASETS (with CV mixing r=32 d=1 +ppw)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "── Traffic (862v) ──"
run_all_horizons traffic cv \
    --root_path ./dataset/ --data_path traffic.csv \
    --data custom --enc_in 862 \
    --batch_size 512 --learning_rate 0.008 \
    $CV_ARGS
echo ""

echo "── Electricity (321v) ──"
run_all_horizons electricity cv \
    --root_path ./dataset/ --data_path electricity.csv \
    --data custom --enc_in 321 \
    --batch_size 1024 --learning_rate 0.007 \
    $CV_ARGS
echo ""

echo "── Weather (21v) ──"
run_all_horizons weather cv \
    --root_path ./dataset/ --data_path weather.csv \
    --data custom --enc_in 21 \
    --batch_size 4096 --learning_rate 0.0007 \
    $CV_ARGS
echo ""

echo "── Solar (137v) ──"
run_all_horizons solar cv \
    --root_path ./dataset/ --data_path solar_AL.csv \
    --data custom --enc_in 137 \
    --batch_size 1024 --learning_rate 0.005 \
    $CV_ARGS
echo ""

# ═══════════════════════════════════════════════════════════════
# LOW-VAR DATASETS — run BOTH with and without mixing
# ═══════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "LOW-VAR DATASETS (both configs: no mixing + cv mixing)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Low-var CV args: smaller rank for 7-8 vars
CV_LOW="--cv_mixing mlp --cv_rank 4 --cv_post_pw"

for ds_name in ETTh1 ETTh2 ETTm1 ETTm2; do
    # Determine data flag
    if [ "$ds_name" = "ETTh1" ] || [ "$ds_name" = "ETTh2" ]; then
        data_flag="$ds_name"
    else
        data_flag="$ds_name"
    fi

    echo "── ${ds_name} (7v) — no mixing ──"
    run_all_horizons "$ds_name" none \
        --root_path ./dataset/ --data_path "${ds_name}.csv" \
        --data "$data_flag" --enc_in 7 \
        --batch_size 2048 --learning_rate 0.0005
    echo ""

    echo "── ${ds_name} (7v) — cv mixing r=4 ──"
    run_all_horizons "$ds_name" cv \
        --root_path ./dataset/ --data_path "${ds_name}.csv" \
        --data "$data_flag" --enc_in 7 \
        --batch_size 2048 --learning_rate 0.0005 \
        $CV_LOW
    echo ""
done

echo "── Exchange (8v) — no mixing ──"
run_all_horizons exchange none \
    --root_path ./dataset/ --data_path exchange_rate.csv \
    --data custom --enc_in 8 \
    --batch_size 2048 --learning_rate 0.0005
echo ""

echo "── Exchange (8v) — cv mixing r=4 ──"
run_all_horizons exchange cv \
    --root_path ./dataset/ --data_path exchange_rate.csv \
    --data custom --enc_in 8 \
    --batch_size 2048 --learning_rate 0.0005 \
    $CV_LOW
echo ""

echo "── ILI (7v) — no mixing ──"
run_ili_horizons ili none \
    --root_path ./dataset/ --data_path national_illness.csv \
    --data custom --enc_in 7 --seq_len 36 \
    --batch_size 512 --learning_rate 0.01
echo ""

echo "── ILI (7v) — cv mixing r=4 ──"
run_ili_horizons ili cv \
    --root_path ./dataset/ --data_path national_illness.csv \
    --data custom --enc_in 7 --seq_len 36 \
    --batch_size 512 --learning_rate 0.01 \
    $CV_LOW
echo ""

# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  FULL RESULTS                                             ║"
echo "╚═══════════════════════════════════════════════════════════╝"

python3 -c "
import re, os, glob

results = {}
for f in sorted(glob.glob('logs/full_bench/*.log')):
    name = os.path.basename(f).replace('.log','')
    with open(f) as fh:
        content = fh.read()
    mse_vals = re.findall(r'mse:([0-9.]+)', content)
    mae_vals = re.findall(r'mae:([0-9.]+)', content)
    if mse_vals:
        results[name] = {'mse': float(mse_vals[-1]), 'mae': float(mae_vals[-1])}

# Parse keys: {tag}_{dataset}_{pred_len}
# Examples: cv_traffic_96, none_ETTh1_336, cv_ili_24

datasets_high = ['traffic', 'electricity', 'weather', 'solar']
datasets_low = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange']
datasets_ili = ['ili']

def print_dataset(ds, horizons, configs):
    print(f'\n  {ds.upper()}:')
    header = f\"  {'Config':8s}\"
    for h in horizons:
        header += f'  T={h:3d}  '
    print(header)
    print('  ' + '-' * (10 + 9 * len(horizons)))
    for cfg in configs:
        row = f'  {cfg:8s}'
        for h in horizons:
            key = f'{cfg}_{ds}_{h}'
            if key in results:
                row += f'  {results[key][\"mse\"]:7.4f}'
            else:
                row += f'    ---  '
        print(row)

print('\n═══ HIGH-VAR (cv mixing) ═══')
for ds in datasets_high:
    print_dataset(ds, [96, 192, 336, 720], ['cv'])

print('\n═══ LOW-VAR (none vs cv) ═══')
for ds in datasets_low:
    print_dataset(ds, [96, 192, 336, 720], ['none', 'cv'])

print_dataset('ili', [24, 36, 48, 60], ['none', 'cv'])

# Best per cell
print('\n\n═══ BEST MSE PER SETTING (for paper Table) ═══')
print(f'{\"Dataset\":12s}  {\"H1\":>7s}  {\"H2\":>7s}  {\"H3\":>7s}  {\"H4\":>7s}  {\"Avg\":>7s}')
print('-' * 55)
all_ds = datasets_high + datasets_low + datasets_ili
for ds in all_ds:
    horizons = [24,36,48,60] if ds == 'ili' else [96,192,336,720]
    row = f'{ds:12s}'
    mses = []
    for h in horizons:
        best_mse = 999
        for cfg in ['cv', 'none']:
            key = f'{cfg}_{ds}_{h}'
            if key in results and results[key]['mse'] < best_mse:
                best_mse = results[key]['mse']
        if best_mse < 999:
            row += f'  {best_mse:.4f}'
            mses.append(best_mse)
        else:
            row += f'    --- '
    if mses:
        row += f'  {sum(mses)/len(mses):.4f}'
    print(row)
"

echo ""
echo "Done! Logs in logs/full_bench/"
