#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Targeted Reruns — Only datasets with wrong hyperparameters
# ~28 runs instead of 80
# ═══════════════════════════════════════════════════════════════
#
# MUST FIX:
#   Exchange:  bs=32, lr=1e-5   (was bs=2048, lr=5e-4 — 64x/50x wrong)
#   ILI:       bs=32, lr=0.01, lradj=type3, patch=6/3, label_len=18
#   ETTm2:     lr=0.0001        (was 0.0005 — 5x wrong)
#
# PARTIAL FIX:
#   Traffic:   cv only with xPatch ref bs=96, lr=0.005
#              (keep existing 'none' baseline, just rerun cv configs)
#
# SKIP (already correct or already winning):
#   ETTh1, ETTh2, ETTm1, Weather, Solar, Electricity
#
# ═══════════════════════════════════════════════════════════════

if [ ! -d "./logs/rerun_fixed" ]; then mkdir -p ./logs/rerun_fixed; fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo ""

model_name=AdaPatch

COMMON="--is_training 1 --model $model_name --features M \
        --des RerunFixed --itr 1 \
        --alpha 0.3 --alpha_lr_mult 50 \
        --train_epochs 100 --patience 10 --num_workers 4"

CV_HIGH="--cv_mixing mlp --cv_rank 32 --cv_post_pw"
CV_HIGH_NOPPW="--cv_mixing mlp --cv_rank 32"
CV_LOW="--cv_mixing mlp --cv_rank 4 --cv_post_pw"

run_one() {
    local dataset=$1 pred_len=$2 tag=$3
    shift 3
    local logfile="logs/rerun_fixed/${tag}_${dataset}_${pred_len}.log"

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
echo "║  Targeted Reruns — Fix HParam Mismatches (28 runs)       ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════
# FIX 1: EXCHANGE — bs=32, lr=1e-5 (was 64x/50x wrong!)
# ═══════════════════════════════════════════════════════════════
echo "━━━ FIX 1: Exchange (bs=32, lr=1e-5) ━━━"

EXCHANGE="--root_path ./dataset/ --data_path exchange_rate.csv \
          --data custom --enc_in 8 --seq_len 96 \
          --batch_size 32 --learning_rate 0.00001 --lradj sigmoid"

echo "  — no mixing —"
for pl in 96 192 336 720; do
    run_one exchange $pl none $EXCHANGE
done

echo "  — cv mixing r=4 —"
for pl in 96 192 336 720; do
    run_one exchange $pl cv $EXCHANGE $CV_LOW
done
echo ""

# ═══════════════════════════════════════════════════════════════
# FIX 2: ILI — bs=32, lr=0.01, lradj=type3, patch=6/3
# ═══════════════════════════════════════════════════════════════
echo "━━━ FIX 2: ILI (bs=32, lradj=type3, patch=6/3) ━━━"

ILI="--root_path ./dataset/ --data_path national_illness.csv \
     --data custom --enc_in 7 --seq_len 36 --label_len 18 \
     --batch_size 32 --learning_rate 0.01 --lradj type3 \
     --patch_len 6 --stride 3"

echo "  — no mixing —"
for pl in 24 36 48 60; do
    run_one ili $pl none $ILI
done

echo "  — cv mixing r=4 —"
for pl in 24 36 48 60; do
    run_one ili $pl cv $ILI $CV_LOW
done
echo ""

# ═══════════════════════════════════════════════════════════════
# FIX 3: ETTm2 — lr=0.0001 (was 5x too high)
# ═══════════════════════════════════════════════════════════════
echo "━━━ FIX 3: ETTm2 (lr=0.0001) ━━━"

ETTM2="--root_path ./dataset/ --data_path ETTm2.csv \
       --data ETTm2 --enc_in 7 --seq_len 96 \
       --batch_size 2048 --learning_rate 0.0001 --lradj sigmoid"

echo "  — no mixing —"
for pl in 96 192 336 720; do
    run_one ETTm2 $pl none $ETTM2
done

echo "  — cv mixing r=4 —"
for pl in 96 192 336 720; do
    run_one ETTm2 $pl cv $ETTM2 $CV_LOW
done
echo ""

# ═══════════════════════════════════════════════════════════════
# FIX 4: Traffic — cv only with xPatch ref hparams
# (keep existing 'none' baseline results)
# ═══════════════════════════════════════════════════════════════
echo "━━━ FIX 4: Traffic cv only (bs=96, lr=0.005) ━━━"

TRAFFIC="--root_path ./dataset/ --data_path traffic.csv \
         --data custom --enc_in 862 --seq_len 96 \
         --batch_size 96 --learning_rate 0.005 --lradj sigmoid"

echo "  — cv r=32 +ppw (T=96,192,336) —"
for pl in 96 192 336; do
    run_one traffic $pl cv_ppw $TRAFFIC $CV_HIGH
done

echo "  — cv r=32 NO ppw (T=720, avoid overfitting) —"
run_one traffic 720 cv_noppw $TRAFFIC $CV_HIGH_NOPPW
echo ""

# ═══════════════════════════════════════════════════════════════
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  DONE — 28 runs complete                                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Logs in logs/rerun_fixed/"
echo "Total: $(ls logs/rerun_fixed/*.log 2>/dev/null | wc -l) log files"
