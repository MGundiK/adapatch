#!/bin/bash
# ============================================================================
# AdaPatch + Hydra Post-Fusion Experiments
# ============================================================================
# Tests three configs per dataset:
#   1. AdaPatch baseline (cv_mixing=none, no post-fusion)
#   2. AdaPatch + MLP in-stream (cv_mixing=mlp, original AdaPatch approach)
#   3. AdaPatch + Hydra post-fusion (cv_mixing=none + post_fusion hydra_gated)
#   4. AdaPatch + MLP in-stream + Hydra post-fusion (both active)
#
# run.py needs these new args:
#   --cv_post_fusion       (store_true flag)
#   --cv_post_fusion_mode  (str, default None)
#   --cv_post_fusion_rank  (int, default None)
# ============================================================================

MODEL="AdaPatch"

# ============================================================
# Electricity (321v) — bs=256, lr=0.005
# ============================================================
ELEC="--is_training 1 --root_path ./dataset/ --data_path electricity.csv \
  --data custom --features M --enc_in 321 --seq_len 96 --label_len 48 \
  --patch_len 16 --stride 8 --padding_patch end \
  --ma_type ema --alpha 0.3 --beta 0.3 \
  --batch_size 256 --learning_rate 0.005 --lradj sigmoid \
  --train_epochs 100 --patience 10 --revin 1 --use_amp"

for PL in 96 192 336 720; do
  # Baseline: no mixing
  python run.py --model $MODEL --model_id "AP_E${PL}_none" \
    --pred_len $PL --cv_mixing none --des "ap_none" $ELEC

  # MLP in-stream (original AdaPatch)
  python run.py --model $MODEL --model_id "AP_E${PL}_mlp" \
    --pred_len $PL --cv_mixing mlp --cv_rank 32 --des "ap_mlp" $ELEC

  # Hydra post-fusion only (no in-stream mixing)
  python run.py --model $MODEL --model_id "AP_E${PL}_hydra_pf" \
    --pred_len $PL --cv_mixing none \
    --cv_post_fusion --cv_post_fusion_mode hydra_gated --cv_post_fusion_rank $PL \
    --des "ap_hydra_pf" $ELEC

  # MLP in-stream + Hydra post-fusion (both)
  python run.py --model $MODEL --model_id "AP_E${PL}_mlp_hydra_pf" \
    --pred_len $PL --cv_mixing mlp --cv_rank 32 \
    --cv_post_fusion --cv_post_fusion_mode hydra_gated --cv_post_fusion_rank $PL \
    --des "ap_mlp_hydra_pf" $ELEC
done

# ============================================================
# Traffic (862v) — bs=96, lr=0.005
# ============================================================
TRAF="--is_training 1 --root_path ./dataset/ --data_path traffic.csv \
  --data custom --features M --enc_in 862 --seq_len 96 --label_len 48 \
  --patch_len 16 --stride 8 --padding_patch end \
  --ma_type ema --alpha 0.3 --beta 0.3 \
  --batch_size 96 --learning_rate 0.005 --lradj sigmoid \
  --train_epochs 100 --patience 10 --revin 1 --use_amp"

for PL in 96 192 336 720; do
  python run.py --model $MODEL --model_id "AP_T${PL}_none" \
    --pred_len $PL --cv_mixing none --des "ap_none" $TRAF

  python run.py --model $MODEL --model_id "AP_T${PL}_mlp" \
    --pred_len $PL --cv_mixing mlp --cv_rank 32 --des "ap_mlp" $TRAF

  python run.py --model $MODEL --model_id "AP_T${PL}_hydra_pf" \
    --pred_len $PL --cv_mixing none \
    --cv_post_fusion --cv_post_fusion_mode hydra_gated --cv_post_fusion_rank $PL \
    --des "ap_hydra_pf" $TRAF

  python run.py --model $MODEL --model_id "AP_T${PL}_mlp_hydra_pf" \
    --pred_len $PL --cv_mixing mlp --cv_rank 32 \
    --cv_post_fusion --cv_post_fusion_mode hydra_gated --cv_post_fusion_rank $PL \
    --des "ap_mlp_hydra_pf" $TRAF
done

# ============================================================
# Solar (137v) — --data Solar, bs=512, lr=0.005
# ============================================================
SOLAR="--is_training 1 --root_path ./dataset/ --data_path solar.txt \
  --data Solar --features M --enc_in 137 --seq_len 96 --label_len 48 \
  --patch_len 16 --stride 8 --padding_patch end \
  --ma_type ema --alpha 0.3 --beta 0.3 \
  --batch_size 512 --learning_rate 0.005 --lradj sigmoid \
  --train_epochs 100 --patience 10 --revin 1 --use_amp"

for PL in 96 192 336 720; do
  python run.py --model $MODEL --model_id "AP_S${PL}_none" \
    --pred_len $PL --cv_mixing none --des "ap_none" $SOLAR

  python run.py --model $MODEL --model_id "AP_S${PL}_mlp" \
    --pred_len $PL --cv_mixing mlp --cv_rank 32 --des "ap_mlp" $SOLAR

  python run.py --model $MODEL --model_id "AP_S${PL}_hydra_pf" \
    --pred_len $PL --cv_mixing none \
    --cv_post_fusion --cv_post_fusion_mode hydra_gated --cv_post_fusion_rank 32 \
    --des "ap_hydra_pf" $SOLAR
done

# ============================================================
# Weather (21v) — bs=2048, lr=0.0005
# ============================================================
WEATHER="--is_training 1 --root_path ./dataset/ --data_path weather.csv \
  --data custom --features M --enc_in 21 --seq_len 96 --label_len 48 \
  --patch_len 16 --stride 8 --padding_patch end \
  --ma_type ema --alpha 0.3 --beta 0.3 \
  --batch_size 2048 --learning_rate 0.0005 --lradj sigmoid \
  --train_epochs 100 --patience 10 --revin 1 --use_amp"

for PL in 96 192 336 720; do
  python run.py --model $MODEL --model_id "AP_W${PL}_none" \
    --pred_len $PL --cv_mixing none --des "ap_none" $WEATHER

  python run.py --model $MODEL --model_id "AP_W${PL}_mlp" \
    --pred_len $PL --cv_mixing mlp --cv_rank 32 --des "ap_mlp" $WEATHER

  python run.py --model $MODEL --model_id "AP_W${PL}_hydra_pf" \
    --pred_len $PL --cv_mixing none \
    --cv_post_fusion --cv_post_fusion_mode hydra_gated --cv_post_fusion_rank 32 \
    --des "ap_hydra_pf" $WEATHER
done

echo ""
echo "============================================================"
echo " Done. Extract results:"
echo "   grep 'ap_' result.txt"
echo ""
echo " Comparison: AdaPatch MLP vs Hydra post-fusion"
echo " Key question: Does Hydra PF beat MLP in-stream?"
echo " Bonus: Does MLP + Hydra PF together beat both?"
echo "============================================================"
