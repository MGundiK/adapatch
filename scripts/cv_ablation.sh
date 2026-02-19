#!/bin/bash
# Cross-Variable Mixing Ablation Study
#
# Tests: none (baseline) vs mlp (bottleneck) vs conv (local)
# Datasets: Traffic (862v), Electricity (321v), Weather (21v)
# Horizons: 96, 336
#
# These are the datasets where xPatch loses to iTransformer/CARD
# due to channel independence. If CV mixing helps, it will show here.

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/cv_ablation" ]; then
    mkdir ./logs/cv_ablation
fi

model_name=AdaPatch

# ─── Traffic (862 vars) ─────────────────────────────────────
run_traffic() {
    local tag=$1
    shift
    for pred_len in 96 336; do
        echo "  Traffic pred_len=${pred_len} [${tag}]"
        python -u run.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path traffic.csv \
            --model_id traffic_${pred_len}_${tag} \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len 96 \
            --pred_len $pred_len \
            --enc_in 862 \
            --des 'CVAblation' \
            --itr 1 \
            --batch_size 192 \
            --learning_rate 0.005 \
            --lradj 'sigmoid' \
            --alpha 0.3 \
            --alpha_lr_mult 50 \
            "$@" > logs/cv_ablation/${tag}_traffic_96_${pred_len}.log
    done
}

# ─── Electricity (321 vars) ─────────────────────────────────
run_electricity() {
    local tag=$1
    shift
    for pred_len in 96 336; do
        echo "  Electricity pred_len=${pred_len} [${tag}]"
        python -u run.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path electricity.csv \
            --model_id electricity_${pred_len}_${tag} \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len 96 \
            --pred_len $pred_len \
            --enc_in 321 \
            --des 'CVAblation' \
            --itr 1 \
            --batch_size 512 \
            --learning_rate 0.005 \
            --lradj 'sigmoid' \
            --alpha 0.3 \
            --alpha_lr_mult 50 \
            "$@" > logs/cv_ablation/${tag}_electricity_96_${pred_len}.log
    done
}

# ─── Weather (21 vars) ──────────────────────────────────────
run_weather() {
    local tag=$1
    shift
    for pred_len in 96 336; do
        echo "  Weather pred_len=${pred_len} [${tag}]"
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
            --des 'CVAblation' \
            --itr 1 \
            --batch_size 2048 \
            --learning_rate 0.0005 \
            --lradj 'sigmoid' \
            --alpha 0.3 \
            --alpha_lr_mult 50 \
            "$@" > logs/cv_ablation/${tag}_weather_96_${pred_len}.log
    done
}

echo "============================================"
echo "Cross-Variable Mixing Ablation"
echo "============================================"

# 1. Baseline (no mixing — current AdaPatch)
echo "[1/3] No mixing (baseline)"
run_traffic   "none"
run_electricity "none"
run_weather   "none"

# 2. Bottleneck MLP mixing
echo "[2/3] MLP mixing (C→r→C)"
run_traffic   "mlp_r32" --cv_mixing mlp --cv_rank 32
run_electricity "mlp_r32" --cv_mixing mlp --cv_rank 32
run_weather   "mlp_r16" --cv_mixing mlp --cv_rank 16

# 3. Conv mixing (local, kernel=7)
echo "[3/3] Conv mixing (local k=7)"
run_traffic   "conv_k7" --cv_mixing conv --cv_kernel 7
run_electricity "conv_k7" --cv_mixing conv --cv_kernel 7
run_weather   "conv_k7" --cv_mixing conv --cv_kernel 7

echo "============================================"
echo "CV ablation complete. Parse results:"
echo "  grep -r 'mse:' logs/cv_ablation/ | sort"
echo ""
echo "Targets to beat (from Table 13):"
echo "  Traffic   T=96:  xPatch=0.471  iTransformer=0.395"
echo "  Traffic   T=336: xPatch=0.501  iTransformer=0.433"
echo "  Elec     T=96:  xPatch=0.159  CARD=0.141"
echo "  Elec     T=336: xPatch=0.182  CARD=0.173"
echo "  Weather  T=96:  xPatch=0.168  CARD=0.150"
echo "  Weather  T=336: xPatch=0.236  CARD=0.260"
echo "============================================"
