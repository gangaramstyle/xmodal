#!/bin/bash
# v4 (docs/MIXED_V4_DESIGN.md): DIRECT modality completion. 32 co-located foreground positions in a
# 64mm prism; at each, one modality hidden (balanced) + 3 visible. Loss = position|target-modality
# matching + pixel MAE. Scan-relative target + EMA. No view-CLS/curriculum/panels/registers. Pins SHA.
# Env: SEED (default 0). ckpt/run tagged v4.
#SBATCH --job-name=xmodal_mixv4
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/mixv4_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git fetch -q origin mixed-v4
SHA="${LAUNCH_SHA:-$(git rev-parse origin/mixed-v4)}"
git checkout -q --detach "$SHA"
test "$(git rev-parse HEAD)" = "$SHA" || { echo "FATAL: checkout SHA mismatch"; exit 1; }
echo "PINNED COMMIT $SHA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

NAME="modality_v4_s${SEED:-0}"
.venv/bin/python -u scripts/run_modality.py \
    --batch-size 224 --n-pos 32 --prism 64 --patch 8 --content-blur 3 --orient native \
    --seed ${SEED:-0} --steps 70000 --ema-color \
    --ckpt-dir runs/${NAME} \
    --wandb xmodal-mixed --wandb-run "${NAME}_${SLURM_JOB_ID:-manual}"
