#!/bin/bash
# AdaPatch Ablation Study
# Tests contribution of each innovation component.
#
# Configurations:
#   full          = all innovations ON (AdaPatch as designed)
#   no_multiscale = remove multiscale depthwise conv
#   no_causal     = remove dilated causal conv
#   no_gated      = replace gated fusion with concat+linear
#   no_agg        = remove aggregate conv in trend
#   no_alpha_lr   = freeze alpha (alpha_lr_mult=0, effectively xPatch decomposition)
#   xpatch_like   = ALL innovations OFF (closest to xPatch baseline)
#
# Datasets: ETTh1 (7 vars, small) + Weather (21 vars, heterogeneous)
# Horizons: 96 + 336

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ablation" ]; then
    mkdir ./logs/ablation
fi

model_name=AdaPatch

# Common args for each dataset
run_etth1() {
    local tag=$1
    shift
    for pred_len in 96 336; do
        python -u run.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path ETTh1.csv \
            --model_id ETTh1_${pred_len}_${tag} \
            --model $model_name \
            --data ETTh1 \
            --features M \
            --seq_len 96 \
            --pred_len $pred_len \
            --enc_in 7 \
            --des 'Ablation' \
            --itr 1 \
            --batch_size 2048 \
            --learning_rate 0.0005 \
            --lradj 'sigmoid' \
            --alpha 0.3 \
            "$@" > logs/ablation/${tag}_ETTh1_96_${pred_len}.log
    done
}

run_weather() {
    local tag=$1
    shift
    for pred_len in 96 336; do
        python -u run.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path weather.csv \
            --model_id weather_${pred_len}_${tag} \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len 96 \
            --pred_len $pred_len \
            --enc_in 21 \
            --des 'Ablation' \
            --itr 1 \
            --batch_size 2048 \
            --learning_rate 0.0005 \
            --lradj 'sigmoid' \
            --alpha 0.3 \
            "$@" > logs/ablation/${tag}_weather_96_${pred_len}.log
    done
}

echo "============================================"
echo "AdaPatch Ablation Study"
echo "============================================"

# 1. Full model (all innovations ON)
echo "[1/7] Full model"
run_etth1 "full" --alpha_lr_mult 50
run_weather "full" --alpha_lr_mult 50

# 2. Remove multiscale depthwise conv
echo "[2/7] No multiscale"
run_etth1 "no_multiscale" --no_multiscale --alpha_lr_mult 50
run_weather "no_multiscale" --no_multiscale --alpha_lr_mult 50

# 3. Remove dilated causal conv
echo "[3/7] No causal"
run_etth1 "no_causal" --no_causal --alpha_lr_mult 50
run_weather "no_causal" --no_causal --alpha_lr_mult 50

# 4. Replace gated fusion with concat+linear
echo "[4/7] No gated fusion"
run_etth1 "no_gated" --no_gated_fusion --alpha_lr_mult 50
run_weather "no_gated" --no_gated_fusion --alpha_lr_mult 50

# 5. Remove aggregate conv in trend
echo "[5/7] No aggregate conv"
run_etth1 "no_agg" --no_agg_conv --alpha_lr_mult 50
run_weather "no_agg" --no_agg_conv --alpha_lr_mult 50

# 6. Freeze alpha (no learnable decomposition)
echo "[6/7] No alpha learning"
run_etth1 "no_alpha_lr" --alpha_lr_mult 0
run_weather "no_alpha_lr" --alpha_lr_mult 0

# 7. All innovations OFF (closest to xPatch)
echo "[7/7] xPatch-like baseline"
run_etth1 "xpatch_like" --no_multiscale --no_causal --no_gated_fusion --no_agg_conv --alpha_lr_mult 0
run_weather "xpatch_like" --no_multiscale --no_causal --no_gated_fusion --no_agg_conv --alpha_lr_mult 0

echo "============================================"
echo "Ablation complete. Parse results with:"
echo "  grep -r 'mse:' logs/ablation/ | sort"
echo "============================================"
