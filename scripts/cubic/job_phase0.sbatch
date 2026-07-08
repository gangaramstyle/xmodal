#!/bin/bash
# CUBIC A40 phase-0 pretraining (full objective: MAE + series-CLS + view-CLS).
#SBATCH --job-name=xmodal_p0
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/cbica/home/gangarav/xmodal/p0_%j.log
set -euo pipefail

cd ~/xmodal
git pull -q || true
export HF_HOME="${HF_HOME:-$HOME/xmodal/.hf}"     # BraTS pulls cache to shared storage
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/python -u scripts/run_phase0.py \
    --patients 300 --holdout 16 --cache-size 64 \
    --batch-size 256 --token-count 128 --steps 20000 \
    --ckpt-dir runs/phase0 \
    --wandb xmodal-phase0 --wandb-run "p0_a40_${SLURM_JOB_ID:-manual}"
