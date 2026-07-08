#!/bin/bash
# CUBIC A40 phased pretraining: self -> cross -> latent on local BraTS-2026 data.
#SBATCH --job-name=xmodal_phased
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/phased_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

cd ~/xmodal
git pull -q || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/python -u scripts/run_phased.py \
    --batch-size 128 --token-count 128 --cache-size 64 \
    --self-steps 25000 --cross-steps 25000 --latent-steps 20000 \
    --ckpt-dir runs/phased \
    --wandb xmodal-phased --wandb-run "phased_a40_${SLURM_JOB_ID:-manual}"
