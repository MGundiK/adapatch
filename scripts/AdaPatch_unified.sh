#!/bin/bash
# AdaPatch - Unified Settings (matches xPatch Table 3 / TimesNet protocol)
# seq_len=96 for all datasets except ILI (seq_len=36)

export CUDA_VISIBLE_DEVICES=0
MODEL=AdaPatch
D_MODEL=128
N_BLOCKS=2

# ─── ETTh1 (7 variables) ─────────────────────────────────────────
for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --model_id ETTh1_96_${pred_len} --model $MODEL --data ETTh1 --features M \
  --seq_len 96 --label_len 48 --pred_len $pred_len --enc_in 7 \
  --d_model $D_MODEL --n_blocks $N_BLOCKS \
  --patch_len 16 --stride 8 --patch_len_trend 32 \
  --learning_rate 0.0001 --lradj sigmoid \
  --train_epochs 100 --patience 10 --batch_size 32 --des unified --itr 1
done

# ─── ETTh2 (7 variables) ─────────────────────────────────────────
for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv \
  --model_id ETTh2_96_${pred_len} --model $MODEL --data ETTh2 --features M \
  --seq_len 96 --label_len 48 --pred_len $pred_len --enc_in 7 \
  --d_model $D_MODEL --n_blocks $N_BLOCKS \
  --patch_len 16 --stride 8 --patch_len_trend 32 \
  --learning_rate 0.0001 --lradj sigmoid \
  --train_epochs 100 --patience 10 --batch_size 32 --des unified --itr 1
done

# ─── ETTm1 (7 variables) ─────────────────────────────────────────
for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv \
  --model_id ETTm1_96_${pred_len} --model $MODEL --data ETTm1 --features M \
  --seq_len 96 --label_len 48 --pred_len $pred_len --enc_in 7 \
  --d_model $D_MODEL --n_blocks $N_BLOCKS \
  --patch_len 16 --stride 8 --patch_len_trend 32 \
  --learning_rate 0.0001 --lradj sigmoid \
  --train_epochs 100 --patience 10 --batch_size 32 --des unified --itr 1
done

# ─── ETTm2 (7 variables) ─────────────────────────────────────────
for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm2.csv \
  --model_id ETTm2_96_${pred_len} --model $MODEL --data ETTm2 --features M \
  --seq_len 96 --label_len 48 --pred_len $pred_len --enc_in 7 \
  --d_model $D_MODEL --n_blocks $N_BLOCKS \
  --patch_len 16 --stride 8 --patch_len_trend 32 \
  --learning_rate 0.0001 --lradj sigmoid \
  --train_epochs 100 --patience 10 --batch_size 32 --des unified --itr 1
done

# ─── Weather (21 variables) ──────────────────────────────────────
for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv \
  --model_id Weather_96_${pred_len} --model $MODEL --data custom --features M \
  --seq_len 96 --label_len 48 --pred_len $pred_len --enc_in 21 \
  --d_model $D_MODEL --n_blocks $N_BLOCKS \
  --patch_len 16 --stride 8 --patch_len_trend 32 \
  --learning_rate 0.0001 --lradj sigmoid \
  --train_epochs 100 --patience 10 --batch_size 32 --des unified --itr 1
done

# ─── Electricity (321 variables) ─────────────────────────────────
for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv \
  --model_id Electricity_96_${pred_len} --model $MODEL --data custom --features M \
  --seq_len 96 --label_len 48 --pred_len $pred_len --enc_in 321 \
  --d_model $D_MODEL --n_blocks $N_BLOCKS \
  --patch_len 16 --stride 8 --patch_len_trend 32 \
  --learning_rate 0.0001 --lradj sigmoid \
  --train_epochs 100 --patience 10 --batch_size 16 --des unified --itr 1
done

# ─── Traffic (862 variables) — with cross-variable gate ──────────
for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv \
  --model_id Traffic_96_${pred_len} --model $MODEL --data custom --features M \
  --seq_len 96 --label_len 48 --pred_len $pred_len --enc_in 862 \
  --d_model $D_MODEL --n_blocks $N_BLOCKS \
  --patch_len 16 --stride 8 --patch_len_trend 32 \
  --use_cross_variable \
  --learning_rate 0.0001 --lradj sigmoid \
  --train_epochs 100 --patience 10 --batch_size 8 --des unified --itr 1
done

# ─── ILI (7 variables, shorter) ──────────────────────────────────
for pred_len in 24 36 48 60; do
python -u run.py \
  --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv \
  --model_id ILI_36_${pred_len} --model $MODEL --data custom --features M \
  --seq_len 36 --label_len 18 --pred_len $pred_len --enc_in 7 \
  --d_model 64 --n_blocks 1 \
  --patch_len 8 --stride 4 --patch_len_trend 12 \
  --learning_rate 0.001 --lradj sigmoid \
  --train_epochs 100 --patience 10 --batch_size 32 --des unified --itr 1
done
