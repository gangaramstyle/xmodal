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

# Reconstruct the held-out METS set from disk via the deterministic hash split (no wandb; the old
# xmodal-phased/heldout-mets-51 artifact was deleted). Provably disjoint from pretraining (see make_heldout_local.py).
.venv/bin/python scripts/make_heldout_local.py /tmp/ho/mets_ho data/brats26/mets_train
[ -d /tmp/ho/mets_ho ] && [ -n "$(ls -A /tmp/ho/mets_ho 2>/dev/null)" ] || { echo "FATAL: held-out reconstruction failed"; exit 1; }

ARMS="${ARMS:-v5_both_tumor_s0 v5_both_s0 v5_mae_tumor_s0}"
STRIDE="${STRIDE:-0}"                       # 0 = final checkpoint only; N>0 = every Nth checkpoint (trajectory)
CK=()
for arm in $ARMS; do
  if [ "$STRIDE" -gt 0 ]; then
    i=0
    for f in runs/$arm/step_*.pt; do
      [ -e "$f" ] || continue
      [ $((i % STRIDE)) -eq 0 ] && CK+=("${arm}@$(basename "$f" .pt)=$f")
      i=$((i+1))
    done
  else
    ls runs/$arm/step_*.pt >/dev/null 2>&1 && CK+=("${arm}=runs/${arm}/step_*.pt")
  fi
done
echo "seg-query readout (unfreeze=${UNFREEZE:-0}) over: $ARMS"
.venv/bin/python -u scripts/eval_segquery.py --data-root /tmp/ho --tracks mets_ho --device cuda \
    --unfreeze "${UNFREEZE:-0}" --enc-width "${EWIDTH:-384}" --enc-heads "${EHEADS:-6}" \
    --checkpoints "${CK[@]}"
echo "=== eval_segquery done (unfreeze=${UNFREEZE:-0}) ==="
