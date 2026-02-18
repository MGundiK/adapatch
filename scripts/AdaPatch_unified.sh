#!/bin/bash
# AdaPatch - Unified Settings (matching xPatch hyperparameters exactly)

alpha=0.3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/adapatch" ]; then
    mkdir ./logs/adapatch
fi

model_name=AdaPatch
seq_len=96

for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${pred_len} \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.0005 \
    --lradj 'sigmoid' \
    --alpha $alpha > logs/adapatch/${model_name}'_ETTh1_'${seq_len}'_'${pred_len}.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_${pred_len} \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.0005 \
    --lradj 'sigmoid' \
    --alpha $alpha > logs/adapatch/${model_name}'_ETTh2_'${seq_len}'_'${pred_len}.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_${pred_len} \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.0005 \
    --lradj 'sigmoid' \
    --alpha $alpha > logs/adapatch/${model_name}'_ETTm1_'${seq_len}'_'${pred_len}.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_${pred_len} \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.0001 \
    --lradj 'sigmoid' \
    --alpha $alpha > logs/adapatch/${model_name}'_ETTm2_'${seq_len}'_'${pred_len}.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id weather_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.0005 \
    --lradj 'sigmoid' \
    --alpha $alpha > logs/adapatch/${model_name}'_weather_'${seq_len}'_'${pred_len}.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id traffic_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 862 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 96 \
    --learning_rate 0.005 \
    --lradj 'sigmoid' \
    --use_cross_variable \
    --alpha $alpha > logs/adapatch/${model_name}'_traffic_'${seq_len}'_'${pred_len}.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id electricity_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 321 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 256 \
    --learning_rate 0.005 \
    --lradj 'sigmoid' \
    --alpha $alpha > logs/adapatch/${model_name}'_electricity_'${seq_len}'_'${pred_len}.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path exchange_rate.csv \
    --model_id exchange_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 8 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.00001 \
    --lradj 'sigmoid' \
    --alpha $alpha > logs/adapatch/${model_name}'_exchange_'${seq_len}'_'${pred_len}.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path solar.txt \
    --model_id solar_${pred_len} \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 137 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 512 \
    --learning_rate 0.005 \
    --lradj 'sigmoid' \
    --alpha $alpha > logs/adapatch/${model_name}'_solar_'${seq_len}'_'${pred_len}.log
done

# ─── ILI (shorter seq_len, different patch/stride/lr) ────────────
seq_len=36

for pred_len in 24 36 48 60
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path national_illness.csv \
    --model_id ili_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --lradj 'type3' \
    --patch_len 6 \
    --stride 3 \
    --patch_len_trend 12 \
    --alpha $alpha > logs/adapatch/${model_name}'_ili_'${seq_len}'_'${pred_len}.log
done
