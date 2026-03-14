#!/bin/bash
# ============================================================================
# AdaPatch + Hydra Post-Fusion Experiments
# ============================================================================
# Tests per dataset:
#   1. none       — AdaPatch baseline (no CV mixing)
#   2. mlp        — Original AdaPatch MLP in-stream mixing
#   3. hydra_pf   — Hydra post-fusion only (no in-stream)
#   4. mlp+hydra  — MLP in-stream + Hydra post-fusion (both)
#
# Logs saved to ./logs/adapatch_hydra/<dataset>/<config>/<pred_len>.log
# ============================================================================

MODEL="AdaPatch"
LOGDIR="./logs/adapatch_hydra"

echo ""
echo "========== [$(date '+%H:%M:%S')] AdaPatch + Hydra — HIGH-VAR DATASETS =========="
echo ""

# ============================================================
# Electricity (321v) — bs=256, lr=0.005
# ============================================================
DS="Electricity"
ELEC="--is_training 1 --root_path ./dataset/ --data_path electricity.csv \
  --data custom --features M --enc_in 321 --seq_len 96 --label_len 48 \
  --patch_len 16 --stride 8 --padding_patch end \
  --ma_type ema --alpha 0.3 \
  --batch_size 256 --learning_rate 0.005 --lradj sigmoid \
  --train_epochs 100 --patience 10 --revin 1 --use_amp"

for PL in 96 192 336 720; do
  echo ">>> [$(date '+%H:%M:%S')] ${DS} pl=${PL}"

  sdir="${LOGDIR}/${DS}/none"; mkdir -p ${sdir}
  echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} none"
  python -u run.py --model $MODEL --model_id "AP_E${PL}_none" \
    --pred_len $PL --cv_mixing none --des "ap_none" $ELEC \
    2>&1 | tee ${sdir}/${PL}.log

  sdir="${LOGDIR}/${DS}/mlp"; mkdir -p ${sdir}
  echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} mlp"
  python -u run.py --model $MODEL --model_id "AP_E${PL}_mlp" \
    --pred_len $PL --cv_mixing mlp --cv_rank 32 --des "ap_mlp" $ELEC \
    2>&1 | tee ${sdir}/${PL}.log

  sdir="${LOGDIR}/${DS}/hydra_pf"; mkdir -p ${sdir}
  echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} hydra_pf"
  python -u run.py --model $MODEL --model_id "AP_E${PL}_hydra_pf" \
    --pred_len $PL --cv_mixing none \
    --cv_post_fusion --cv_post_fusion_mode hydra_gated --cv_post_fusion_rank $PL \
    --des "ap_hydra_pf" $ELEC \
    2>&1 | tee ${sdir}/${PL}.log

  sdir="${LOGDIR}/${DS}/mlp_hydra_pf"; mkdir -p ${sdir}
  echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} mlp+hydra_pf"
  python -u run.py --model $MODEL --model_id "AP_E${PL}_mlp_hydra_pf" \
    --pred_len $PL --cv_mixing mlp --cv_rank 32 \
    --cv_post_fusion --cv_post_fusion_mode hydra_gated --cv_post_fusion_rank $PL \
    --des "ap_mlp_hydra_pf" $ELEC \
    2>&1 | tee ${sdir}/${PL}.log
done

echo "  === ${DS} complete ==="
for cfg in none mlp hydra_pf mlp_hydra_pf; do
  printf "  %-16s|" $cfg
  for PL in 96 192 336 720; do
    logfile="${LOGDIR}/${DS}/${cfg}/${PL}.log"
    if [ -f "$logfile" ]; then
      mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
      [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
    else
      printf "     N/A"
    fi
  done
  echo ""
done
echo ""

# ============================================================
# Traffic (862v) — bs=96, lr=0.005
# ============================================================
DS="Traffic"
TRAF="--is_training 1 --root_path ./dataset/ --data_path traffic.csv \
  --data custom --features M --enc_in 862 --seq_len 96 --label_len 48 \
  --patch_len 16 --stride 8 --padding_patch end \
  --alpha 0.3 \
  --batch_size 96 --learning_rate 0.005 --lradj sigmoid \
  --train_epochs 100 --patience 10 --revin 1 --use_amp"

for PL in 96 192 336 720; do
  echo ">>> [$(date '+%H:%M:%S')] ${DS} pl=${PL}"

  sdir="${LOGDIR}/${DS}/none"; mkdir -p ${sdir}
  echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} none"
  python -u run.py --model $MODEL --model_id "AP_T${PL}_none" \
    --pred_len $PL --cv_mixing none --des "ap_none" $TRAF \
    2>&1 | tee ${sdir}/${PL}.log

  sdir="${LOGDIR}/${DS}/mlp"; mkdir -p ${sdir}
  echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} mlp"
  python -u run.py --model $MODEL --model_id "AP_T${PL}_mlp" \
    --pred_len $PL --cv_mixing mlp --cv_rank 32 --des "ap_mlp" $TRAF \
    2>&1 | tee ${sdir}/${PL}.log

  sdir="${LOGDIR}/${DS}/hydra_pf"; mkdir -p ${sdir}
  echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} hydra_pf"
  python -u run.py --model $MODEL --model_id "AP_T${PL}_hydra_pf" \
    --pred_len $PL --cv_mixing none \
    --cv_post_fusion --cv_post_fusion_mode hydra_gated --cv_post_fusion_rank $PL \
    --des "ap_hydra_pf" $TRAF \
    2>&1 | tee ${sdir}/${PL}.log

  sdir="${LOGDIR}/${DS}/mlp_hydra_pf"; mkdir -p ${sdir}
  echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} mlp+hydra_pf"
  python -u run.py --model $MODEL --model_id "AP_T${PL}_mlp_hydra_pf" \
    --pred_len $PL --cv_mixing mlp --cv_rank 32 \
    --cv_post_fusion --cv_post_fusion_mode hydra_gated --cv_post_fusion_rank $PL \
    --des "ap_mlp_hydra_pf" $TRAF \
    2>&1 | tee ${sdir}/${PL}.log
done

echo "  === ${DS} complete ==="
for cfg in none mlp hydra_pf mlp_hydra_pf; do
  printf "  %-16s|" $cfg
  for PL in 96 192 336 720; do
    logfile="${LOGDIR}/${DS}/${cfg}/${PL}.log"
    if [ -f "$logfile" ]; then
      mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
      [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
    else
      printf "     N/A"
    fi
  done
  echo ""
done
echo ""

# ============================================================
# Solar (137v) — --data Solar, bs=512, lr=0.005
# ============================================================
DS="Solar"
SOLAR="--is_training 1 --root_path ./dataset/ --data_path solar.txt \
  --data Solar --features M --enc_in 137 --seq_len 96 --label_len 48 \
  --patch_len 16 --stride 8 --padding_patch end \
  --alpha 0.3 \
  --batch_size 512 --learning_rate 0.005 --lradj sigmoid \
  --train_epochs 100 --patience 10 --revin 1 --use_amp"

for PL in 96 192 336 720; do
  echo ">>> [$(date '+%H:%M:%S')] ${DS} pl=${PL}"

  sdir="${LOGDIR}/${DS}/none"; mkdir -p ${sdir}
  echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} none"
  python -u run.py --model $MODEL --model_id "AP_S${PL}_none" \
    --pred_len $PL --cv_mixing none --des "ap_none" $SOLAR \
    2>&1 | tee ${sdir}/${PL}.log

  sdir="${LOGDIR}/${DS}/mlp"; mkdir -p ${sdir}
  echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} mlp"
  python -u run.py --model $MODEL --model_id "AP_S${PL}_mlp" \
    --pred_len $PL --cv_mixing mlp --cv_rank 32 --des "ap_mlp" $SOLAR \
    2>&1 | tee ${sdir}/${PL}.log

  sdir="${LOGDIR}/${DS}/hydra_pf"; mkdir -p ${sdir}
  echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} hydra_pf (r32)"
  python -u run.py --model $MODEL --model_id "AP_S${PL}_hydra_pf" \
    --pred_len $PL --cv_mixing none \
    --cv_post_fusion --cv_post_fusion_mode hydra_gated --cv_post_fusion_rank 32 \
    --des "ap_hydra_pf" $SOLAR \
    2>&1 | tee ${sdir}/${PL}.log
done

echo "  === ${DS} complete ==="
for cfg in none mlp hydra_pf; do
  printf "  %-16s|" $cfg
  for PL in 96 192 336 720; do
    logfile="${LOGDIR}/${DS}/${cfg}/${PL}.log"
    if [ -f "$logfile" ]; then
      mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
      [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
    else
      printf "     N/A"
    fi
  done
  echo ""
done
echo ""

# ============================================================
# Weather (21v) — bs=2048, lr=0.0005
# ============================================================
DS="Weather"
WEATHER="--is_training 1 --root_path ./dataset/ --data_path weather.csv \
  --data custom --features M --enc_in 21 --seq_len 96 --label_len 48 \
  --patch_len 16 --stride 8 --padding_patch end \
  --alpha 0.3 \
  --batch_size 2048 --learning_rate 0.0005 --lradj sigmoid \
  --train_epochs 100 --patience 10 --revin 1 --use_amp"

for PL in 96 192 336 720; do
  echo ">>> [$(date '+%H:%M:%S')] ${DS} pl=${PL}"

  sdir="${LOGDIR}/${DS}/none"; mkdir -p ${sdir}
  echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} none"
  python -u run.py --model $MODEL --model_id "AP_W${PL}_none" \
    --pred_len $PL --cv_mixing none --des "ap_none" $WEATHER \
    2>&1 | tee ${sdir}/${PL}.log

  sdir="${LOGDIR}/${DS}/mlp"; mkdir -p ${sdir}
  echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} mlp"
  python -u run.py --model $MODEL --model_id "AP_W${PL}_mlp" \
    --pred_len $PL --cv_mixing mlp --cv_rank 32 --des "ap_mlp" $WEATHER \
    2>&1 | tee ${sdir}/${PL}.log

  sdir="${LOGDIR}/${DS}/hydra_pf"; mkdir -p ${sdir}
  echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} hydra_pf (r32)"
  python -u run.py --model $MODEL --model_id "AP_W${PL}_hydra_pf" \
    --pred_len $PL --cv_mixing none \
    --cv_post_fusion --cv_post_fusion_mode hydra_gated --cv_post_fusion_rank 32 \
    --des "ap_hydra_pf" $WEATHER \
    2>&1 | tee ${sdir}/${PL}.log
done

echo "  === ${DS} complete ==="
for cfg in none mlp hydra_pf; do
  printf "  %-16s|" $cfg
  for PL in 96 192 336 720; do
    logfile="${LOGDIR}/${DS}/${cfg}/${PL}.log"
    if [ -f "$logfile" ]; then
      mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
      [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
    else
      printf "     N/A"
    fi
  done
  echo ""
done
echo ""

# ============================================================
# FINAL SUMMARY
# ============================================================
echo ""
echo "========== [$(date '+%H:%M:%S')] ALL COMPLETE =========="
echo ""
echo "Results by dataset:"
echo "==================="

for DS in Electricity Traffic Solar Weather; do
  echo ""
  echo "${DS}:"
  echo "  config           |     96      192      336      720"
  echo "  -----------------+------------------------------------"
  for cfg in none mlp hydra_pf mlp_hydra_pf; do
    dir="${LOGDIR}/${DS}/${cfg}"
    [ ! -d "$dir" ] && continue
    printf "  %-16s |" $cfg
    for PL in 96 192 336 720; do
      logfile="${dir}/${PL}.log"
      if [ -f "$logfile" ]; then
        mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
        [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
      else
        printf "     N/A"
      fi
    done
    echo ""
  done
done

echo ""
echo "Log files: ${LOGDIR}/<dataset>/<config>/<pred_len>.log"
echo "Result file: result.txt (grep 'ap_' result.txt)"
echo ""
echo "Total runs: ~56 (Elec:16 + Traffic:16 + Solar:12 + Weather:12)"
