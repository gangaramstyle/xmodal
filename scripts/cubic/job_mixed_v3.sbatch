#!/bin/bash
# v3 (docs/MIXED_V3_DESIGN.md): structured 4-way targets + conditional matching (position|modality +
# modality|position) + scan-context A (scan-relative channels), EMA target. EXPLICIT flags (no reliance
# on env defaults) and --assert-v3 fails fast unless all three features are active (review Blocker 1).
# Env: SEED (default 0). ckpt-dir/run tagged v3 so it never clobbers the v1/v2 batches.
#SBATCH --job-name=xmodal_mixv3
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/mixv3_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git fetch -q origin mixed-v3                                   # PIN the exact launch commit (no silent stale code)
SHA="${LAUNCH_SHA:-$(git rev-parse origin/mixed-v3)}"
git checkout -q --detach "$SHA"
test "$(git rev-parse HEAD)" = "$SHA" || { echo "FATAL: checkout SHA mismatch"; exit 1; }
echo "PINNED COMMIT $SHA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

NAME="mixed_v3_s${SEED:-0}"
.venv/bin/python -u scripts/run_mixed.py \
    --batch-size 224 --token-count 128 --cache-size 32 \
    --patch-sizes 4 8 16 --content-blur 3 --orient native --seed ${SEED:-0} \
    --steps 70000 \
    --structured --n-pos 12 --target-size 8 \
    --scan-context --ema-color --assert-v3 \
    --ckpt-dir runs/${NAME} \
    --wandb xmodal-mixed --wandb-run "${NAME}_${SLURM_JOB_ID:-manual}"
