#!/bin/bash
# Cross-Variable Mixing v2 — ON LEARNED REPRESENTATIONS
#
# Key difference from v1 (which failed):
#   v1: mixed raw RevIN-normalized values at input → no temporal context
#   v2: mixes INSIDE streams on patch embeddings (seasonal) and FC features (trend)
#       → each variable's learned temporal patterns inform other variables
#
# Tests: none (baseline) vs mlp (bottleneck) vs conv (local)
# Datasets: Traffic (862v), Electricity (321v), Weather (21v), ETTh1 (7v, control)
# Horizons: 96, 336

if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/cv_ablation_v2" ]; then mkdir ./logs/cv_ablation_v2; fi

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
            --des 'CVv2' \
            --itr 1 \
            --batch_size 192 \
            --learning_rate 0.005 \
            --lradj 'sigmoid' \
            --alpha 0.3 \
            --alpha_lr_mult 50 \
            "$@" > logs/cv_ablation_v2/${tag}_traffic_96_${pred_len}.log
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
            --des 'CVv2' \
            --itr 1 \
            --batch_size 512 \
            --learning_rate 0.005 \
            --lradj 'sigmoid' \
            --alpha 0.3 \
            --alpha_lr_mult 50 \
            "$@" > logs/cv_ablation_v2/${tag}_electricity_96_${pred_len}.log
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
            --des 'CVv2' \
            --itr 1 \
            --batch_size 2048 \
            --learning_rate 0.0005 \
            --lradj 'sigmoid' \
            --alpha 0.3 \
            --alpha_lr_mult 50 \
            "$@" > logs/cv_ablation_v2/${tag}_weather_96_${pred_len}.log
    done
}

# ─── ETTh1 (7 vars) — control, should NOT improve ──────────
run_etth1() {
    local tag=$1
    shift
    for pred_len in 96 336; do
        echo "  ETTh1 pred_len=${pred_len} [${tag}]"
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
            --des 'CVv2' \
            --itr 1 \
            --batch_size 2048 \
            --learning_rate 0.0005 \
            --lradj 'sigmoid' \
            --alpha 0.3 \
            --alpha_lr_mult 50 \
            "$@" > logs/cv_ablation_v2/${tag}_ETTh1_96_${pred_len}.log
    done
}

echo "============================================"
echo "Cross-Variable Mixing v2 (on representations)"
echo "============================================"

# 1. Baseline (no mixing)
echo "[1/3] No mixing (baseline)"
run_traffic     "none"
run_electricity "none"
run_weather     "none"
run_etth1       "none"

# 2. Bottleneck MLP mixing on representations
echo "[2/3] MLP mixing on representations"
run_traffic     "mlp_r32" --cv_mixing mlp --cv_rank 32
run_electricity "mlp_r32" --cv_mixing mlp --cv_rank 32
run_weather     "mlp_r16" --cv_mixing mlp --cv_rank 16
run_etth1       "mlp_r8"  --cv_mixing mlp --cv_rank 8

# 3. Conv mixing on representations
echo "[3/3] Conv mixing on representations"
run_traffic     "conv_k7" --cv_mixing conv --cv_kernel 7
run_electricity "conv_k7" --cv_mixing conv --cv_kernel 7
run_weather     "conv_k7" --cv_mixing conv --cv_kernel 7
run_etth1       "conv_k7" --cv_mixing conv --cv_kernel 7

echo "============================================"
echo "CV v2 ablation complete. Parse results:"
echo "  grep -r 'mse:' logs/cv_ablation_v2/ | sort"
echo ""
echo "Targets to beat (from Table 13):"
echo "  Traffic   T=96:  xPatch=0.471  iTransformer=0.395"
echo "  Traffic   T=336: xPatch=0.501  iTransformer=0.433"
echo "  Elec     T=96:  xPatch=0.159  CARD=0.141"
echo "  Elec     T=336: xPatch=0.182  CARD=0.173"
echo "  Weather  T=96:  xPatch=0.168  CARD=0.150"
echo "  Weather  T=336: xPatch=0.236  CARD=0.260"
echo "  ETTh1   T=96:  xPatch=0.376  (control — should NOT improve)"
echo "  ETTh1   T=336: xPatch=0.449  (control — should NOT improve)"
echo "============================================"
