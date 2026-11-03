#!/bin/bash
# nightly_seeds.sh —— 3 种子复现实验: 训练 → 选模 → 留出评估, 全程自动
cd ~/FinRL_DeepSeek
PY=~/finrl_env/bin/python
export PYTHONPATH=~/spinningup
export HF_ENDPOINT=https://hf-mirror.com
for s in 1 2 3; do
  echo "===== 种子 $s 训练开始 $(date '+%F %T') ====="
  $PY train_web.py --epochs 30 --steps 20000 --alpha 0.85 --beta 3000 \
    --cvar-clip 0.05 --hid 512 --l 2 --seed $s \
    --model-name ashare_idx2_30ep_s$s \
    --data-csv ashare_idx2_2022_2026.csv \
    --data-start 2021-09-01 --data-end 2024-12-31 \
    --pi-lr 0.00001 --save-freq 5 || { echo "种子 $s 训练失败"; continue; }
  echo "===== 种子 $s 选模 ====="
  $PY select_best.py --name ashare_idx2_30ep_s$s --csv ashare_idx2_2022_2026.csv --paths 8 || true
  echo "===== 种子 $s 留出评估(20路径) ====="
  $PY use_model.py --model trained_models/ashare_idx2_30ep_s${s}_best.pth \
    --data-csv ashare_idx2_2022_2026.csv --data-start 2025-07-01 --data-end 2030-01-01 \
    --eval-mode sample --eval-n 20 --hidden 512 512 \
    --outdir results/seed$s || true
  echo "===== 种子 $s 完成 $(date '+%F %T') ====="
done
echo "===== 全部完成 $(date '+%F %T') ====="
