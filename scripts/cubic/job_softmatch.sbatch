#!/bin/bash
# Representation experiment (SOFT arm): from-scratch SELF-phase pretraining with the similarity-softened
# matching target (--soft-match-tau 0.1). Paired with job_hardmatch.sbatch (identical config+seed, hard
# matching) so the ONLY difference is the softened target. Question: does not-penalizing ambiguous patch
# confusions yield a better ENCODER (higher held-out patch-F1)? Success = encoder-F1 > the hard arm.
#SBATCH --job-name=xmodal_softmatch
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/softmatch_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git pull -q || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/python -u scripts/run_phased.py \
    --batch-size 224 --token-count 128 --cache-size 32 \
    --patch-sizes 4 8 16 --held-size 8 16 --content-blur 3 \
    --orient native --seed 0 \
    --self-steps 25000 --cross-steps 0 --latent-steps 0 \
    --soft-match-tau 0.1 \
    --ckpt-dir runs/softmatch \
    --wandb xmodal-phased --wandb-run "softmatch_${SLURM_JOB_ID:-manual}"
