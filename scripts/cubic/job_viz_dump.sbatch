#!/bin/bash
# Dump per-lesion inputs+outputs (best model dense-seg) for the interactive molab viz.
#SBATCH --job-name=xmodal_viz
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/viz_%j.log
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/xmodal
git fetch -q origin v5
SHA="${SHA:-$(git rev-parse origin/v5)}"
git checkout -qf --detach "$SHA" && echo "PINNED $SHA"
nvidia-smi --query-gpu=name --format=csv,noheader

.venv/bin/python scripts/make_heldout_local.py /tmp/ho/mets_ho data/brats26/mets_train
[ -d /tmp/ho/mets_ho ] && [ -n "$(ls -A /tmp/ho/mets_ho 2>/dev/null)" ] || { echo "FATAL: held-out reconstruction failed"; exit 1; }

.venv/bin/python -u scripts/viz_dump.py \
    --checkpoint "runs/${ARM:-v5_multisize_vitbase_s0}/step_070000.pt" \
    --enc-width "${EWIDTH:-768}" --enc-heads "${EHEADS:-12}" \
    --n-patients "${NPAT:-16}" --n-prisms "${NPRISM:-6}" --res "${RES:-1.0}" --qsize "${QSIZE:-2.0}" \
    --out "/cbica/home/gangarav/xmodal/viz_dump.npz" --device cuda
echo "=== viz_dump done -> viz_dump.npz ($(du -h /cbica/home/gangarav/xmodal/viz_dump.npz 2>/dev/null | cut -f1)) ==="
