#!/bin/bash
# CUBIC phased pretraining -- RANDOM-plane 2.5D slabs (per-draw random through-plane axis) + MAE-style masking.
# Input patches {4,8,16}mm; matching/recon TARGETS restricted to {8,16}mm (4mm stays input-only).
# Pairs with job_native.sbatch (same config, orient=native) to isolate slab orientation as the variable.
# This is the stress test of the additive size/orientation embedding: same anatomy, three planes.
#SBATCH --job-name=xmodal_random
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/random_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git pull -q || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/python -u scripts/run_phased.py \
    --batch-size 224 --token-count 128 --cache-size 32 \
    --patch-sizes 4 8 16 --held-size 8 16 --content-blur 3 \
    --orient random \
    --self-steps 25000 --cross-steps 25000 --latent-steps 20000 \
    --ckpt-dir runs/random \
    --wandb xmodal-phased --wandb-run "random_${SLURM_JOB_ID:-manual}"
