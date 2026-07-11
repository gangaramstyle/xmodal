#!/bin/bash
# FULL multi-phase curriculum from scratch (self 25k -> cross 25k -> latent 20k, same as native) but with
# an EMA (target) color_head for matching (--ema-color). Slots predict the stable EMA color_head's
# embeddings (stop-grad); the online color_head trains to match the slots; the EMA smooths it. This
# prevents target co-adaptation (BYOL/DINO principle) so the encoder must do the real prediction work.
# Single shared color_head across all series types. Baseline = native (identical config, no EMA; F1 0.614
# @40k). Success = encoder held-out patch-F1 above native at matched steps. Matching runs in self+cross
# phases (latent phase predicts the frozen teacher's latents, unaffected).
#SBATCH --job-name=xmodal_emacolor
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/emacolor_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git pull -q || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/python -u scripts/run_phased.py \
    --batch-size 224 --token-count 128 --cache-size 32 \
    --patch-sizes 4 8 16 --held-size 8 16 --content-blur 3 \
    --orient native --seed ${SEED:-0} \
    --self-steps 25000 --cross-steps 25000 --latent-steps 20000 \
    --ema-color \
    --ckpt-dir runs/emacolor_s${SEED:-0} \
    --wandb xmodal-phased --wandb-run "emacolor_s${SEED:-0}_${SLURM_JOB_ID:-manual}"
