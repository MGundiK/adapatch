#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Rerun FAILED datasets: Solar, Exchange, ILI
# Fixes:
#   Solar:    solar_AL.txt (not .csv), may need --data Solar
#   Exchange: batch_size 2048→512 (test set only 1326 samples)
#   ILI:      --label_len 18 (default 48 > seq_len 36)
# ═══════════════════════════════════════════════════════════════

if [ ! -d "./logs/full_bench" ]; then mkdir -p ./logs/full_bench; fi

model_name=AdaPatch

COMMON="--is_training 1 --model $model_name --features M --seq_len 96 \
        --des FullBench --itr 1 --lradj sigmoid \
        --alpha 0.3 --alpha_lr_mult 50 \
        --train_epochs 100 --patience 10 --num_workers 4"

CV_LOW="--cv_mixing mlp --cv_rank 4 --cv_post_pw"
CV_MID="--cv_mixing mlp --cv_rank 32 --cv_post_pw"

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

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Rerun: Solar, Exchange, ILI (with fixes)                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════
# SOLAR (137v) — Fix: use --data Solar (special loader for .txt)
# The file is solar.txt (symlinked from solar_AL.txt)
# xPatch's data_factory maps 'Solar' to Dataset_Solar which
# handles the headerless tab-separated format.
# ═══════════════════════════════════════════════════════════════
echo "── Checking Solar file ──"
if [ -f "./dataset/solar_AL.txt" ]; then
    SOLAR_FILE="solar_AL.txt"
    echo "  Found: solar_AL.txt"
elif [ -f "./dataset/solar.txt" ]; then
    SOLAR_FILE="solar.txt"
    echo "  Found: solar.txt"
elif [ -f "./dataset/solar_AL.csv" ]; then
    SOLAR_FILE="solar_AL.csv"
    echo "  Found: solar_AL.csv"
else
    echo "  Listing dataset dir for solar files:"
    ls -la ./dataset/solar* 2>/dev/null || echo "  No solar files found!"
    SOLAR_FILE=""
fi

if [ -n "$SOLAR_FILE" ]; then
    echo ""
    echo "── Solar (137v) — cv mixing ──"
    for pl in 96 192 336 720; do
        run_one solar $pl cv \
            --root_path ./dataset/ --data_path "$SOLAR_FILE" \
            --data Solar --enc_in 137 \
            --batch_size 1024 --learning_rate 0.005 \
            $CV_MID
    done
fi

echo ""

# ═══════════════════════════════════════════════════════════════
# EXCHANGE (8v) — Fix: batch_size 2048→512
# test set = 1326 samples; bs=2048 + drop_last → 0 batches
# ═══════════════════════════════════════════════════════════════
echo "── Exchange (8v) — no mixing (bs=512) ──"
for pl in 96 192 336 720; do
    run_one exchange $pl none \
        --root_path ./dataset/ --data_path exchange_rate.csv \
        --data custom --enc_in 8 \
        --batch_size 512 --learning_rate 0.0005
done
echo ""

echo "── Exchange (8v) — cv mixing r=4 (bs=512) ──"
for pl in 96 192 336 720; do
    run_one exchange $pl cv \
        --root_path ./dataset/ --data_path exchange_rate.csv \
        --data custom --enc_in 8 \
        --batch_size 512 --learning_rate 0.0005 \
        $CV_LOW
done
echo ""

# ═══════════════════════════════════════════════════════════════
# ILI (7v) — Fix: --label_len 18 (was 48, > seq_len=36)
# ═══════════════════════════════════════════════════════════════
echo "── ILI (7v) — no mixing (label_len=18) ──"
for pl in 24 36 48 60; do
    run_one ili $pl none \
        --root_path ./dataset/ --data_path national_illness.csv \
        --data custom --enc_in 7 --seq_len 36 --label_len 18 \
        --batch_size 32 --learning_rate 0.01
done
echo ""

echo "── ILI (7v) — cv mixing r=4 (label_len=18) ──"
for pl in 24 36 48 60; do
    run_one ili $pl cv \
        --root_path ./dataset/ --data_path national_illness.csv \
        --data custom --enc_in 7 --seq_len 36 --label_len 18 \
        --batch_size 32 --learning_rate 0.01 \
        $CV_LOW
done
echo ""

echo "Done! Check logs/full_bench/ for results."
