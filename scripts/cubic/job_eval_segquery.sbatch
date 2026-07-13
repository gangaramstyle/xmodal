#!/bin/bash
# Seg-query decoder readout: segmentation as a 5th "modality" -- encode a multi-modal prism source (encoder
# frozen), decode a query position with a learnable SEG token + class head. --unfreeze N last decoder blocks
# (0 = frozen decoder). Latest checkpoint per arm. Env: ARMS, UNFREEZE, EWIDTH/EHEADS (768/12 for ViT-base).
#SBATCH --job-name=xmodal_segq
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/segq_%j.log
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git fetch -q origin v5
SHA="${SHA:-$(git rev-parse origin/v5)}"
git checkout -qf --detach "$SHA" && echo "PINNED $SHA"
nvidia-smi --query-gpu=name --format=csv,noheader

.venv/bin/python - <<'PY'
import wandb, os, tarfile, glob
p = wandb.Api().artifact("vineeth-gangaram-penn/xmodal-phased/heldout-mets-51:v0").download("/tmp/ho_art")
os.makedirs("/tmp/ho", exist_ok=True)
tarfile.open(glob.glob("/tmp/ho_art/*.tar")[0]).extractall("/tmp/ho")
print("held-out extracted ->", os.listdir("/tmp/ho"))
PY

ARMS="${ARMS:-v5_both_tumor_s0 v5_both_s0 v5_mae_tumor_s0}"
CK=()
for arm in $ARMS; do
  ls runs/$arm/step_*.pt >/dev/null 2>&1 && CK+=("${arm}=runs/${arm}/step_*.pt")
done
echo "seg-query readout (unfreeze=${UNFREEZE:-0}) over: $ARMS"
.venv/bin/python -u scripts/eval_segquery.py --data-root /tmp/ho --tracks mets_ho --device cuda \
    --unfreeze "${UNFREEZE:-0}" --enc-width "${EWIDTH:-384}" --enc-heads "${EHEADS:-6}" \
    --checkpoints "${CK[@]}"
echo "=== eval_segquery done (unfreeze=${UNFREEZE:-0}) ==="
