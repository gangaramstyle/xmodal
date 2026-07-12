#!/bin/bash
# v5 (docs/MIXED_V5_DESIGN.md): cross-modal ORDERING with 3D cube patches. Context = 3 modalities + a few
# target-modality (D) anchors; recreate the ordering of held D patches. Off v2 (trusted encoder).
# Ablation via env: MATCH_W (ordering weight) + MAE_W (pixel weight) -> ordering-only / MAE-only / both.
# Env: SEED (0), MATCH_W (1.0), MAE_W (0.25), TAG (both). Pins SHA.
#SBATCH --job-name=xmodal_v5
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/v5_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
# serialize the checkout across concurrently-starting jobs (shared working tree -> .git/index.lock race)
(
  flock -w 300 9 || { echo "FATAL: git lock timeout"; exit 1; }
  git fetch -q origin v5
  SHA="${LAUNCH_SHA:-$(git rev-parse origin/v5)}"
  git checkout -q --detach "$SHA"
  test "$(git rev-parse HEAD)" = "$SHA" || { echo "FATAL: SHA mismatch"; exit 1; }
  echo "PINNED $SHA"
) 9>/tmp/xmodal_git_checkout.lock
SHA="${LAUNCH_SHA:-$(git rev-parse HEAD)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

NAME="v5_${TAG:-both}_s${SEED:-0}"
.venv/bin/python -u scripts/run_v5.py \
    --batch-size 224 --n-src 90 --n-anchor 6 --n-tgt 32 --voxels 8 --prisms 32 64 \
    --match-weight ${MATCH_W:-1.0} --mae-weight ${MAE_W:-0.25} --tumor-frac ${TUMOR_FRAC:-0.0} \
    --content-blur 1 --seed ${SEED:-0} --steps 70000 \
    --ckpt-dir runs/${NAME} \
    --wandb xmodal-mixed --wandb-run "${NAME}_${SLURM_JOB_ID:-manual}"
