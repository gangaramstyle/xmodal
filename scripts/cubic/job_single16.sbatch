#!/bin/bash
# CUBIC phased pretraining, SINGLE-SIZE 16mm + MAE-style masking + content blur (matching rework).
# expandable_segments fixes the memory-fragmentation OOM that killed the mixed runs at ~step 11.5k.
#SBATCH --job-name=xmodal_single16
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/single16_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git pull -q || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/python -u scripts/run_phased.py \
    --batch-size 256 --token-count 128 --cache-size 40 \
    --patch-sizes 16 --content-blur 3 \
    --self-steps 25000 --cross-steps 25000 --latent-steps 20000 \
    --ckpt-dir runs/single16 \
    --wandb xmodal-phased --wandb-run "single16_${SLURM_JOB_ID:-manual}"
