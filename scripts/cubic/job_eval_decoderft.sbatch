#!/bin/bash
# Decoder-head fine-tune readout: reuse the pretrained matching decoder as a task classifier (frozen
# encoder, fine-tune decoder + head), GroupKFold-5 -> macro-F1. Latest checkpoint of each arm.
# Env: ARMS (run dirs), SHA (eval-code commit).
#SBATCH --job-name=xmodal_decft
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/decft_%j.log
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
CK=()
for arm in $ARMS; do
  ls runs/$arm/step_*.pt >/dev/null 2>&1 && CK+=("${arm}=runs/${arm}/step_*.pt")
done
echo "decoder-FT readout over arms: $ARMS"
.venv/bin/python -u scripts/eval_decoder_finetune.py --data-root /tmp/ho --tracks mets_ho --device cuda \
    --checkpoints "${CK[@]}"
echo "=== eval_decoderft done ==="
