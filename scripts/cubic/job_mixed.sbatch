#!/bin/bash
# Mixed-modality conditioned SSL (docs/MIXED_MODAL_DESIGN.md). ONE continuous loop (no phases, no
# latent): series is a per-patch conditioning signal on encoder tokens (Site A) + decoder queries
# (Site B); bags are variable mixed-modality; self<->cross is a stochastic dominant-series alignment
# curriculum. The experiment is EMA vs no-EMA color target, paired at matched seeds for a clean A/B.
# Env: SEED (default 0), EMA=1 -> --ema-color (default off). 70k steps to match the phased compute.
#SBATCH --job-name=xmodal_mixed
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/mixed_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git pull -q || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EMA_FLAG=""; TAG="noema"
if [ "${EMA:-0}" = "1" ]; then EMA_FLAG="--ema-color"; TAG="ema"; fi
VER="${VER:-v2}"                       # run-name version tag (avoids clobbering an earlier batch's dirs/runs)
NAME="mixed_${VER}_${TAG}_s${SEED:-0}"

.venv/bin/python -u scripts/run_mixed.py \
    --batch-size 224 --token-count 128 --held-count 48 --cache-size 32 \
    --patch-sizes 4 8 16 --content-blur 3 --orient native --seed ${SEED:-0} \
    --steps 70000 ${EMA_FLAG} \
    --ckpt-dir runs/${NAME} \
    --wandb xmodal-mixed --wandb-run "${NAME}_${SLURM_JOB_ID:-manual}"
