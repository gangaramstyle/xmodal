#!/bin/bash
# Dense-segmentation eval (ORACLE detection): given a tumor-containing prism, densely query the fine-tuned
# seg decoder at every --res grid voxel -> per-voxel class -> DSC + NSD for ET/TC/WT vs GT. Isolates
# segmentation QUALITY from detection -> comparable to BraTS-METS lesion-wise DSC (~0.75 ET at SOTA).
# Encoder frozen; last --unfreeze decoder blocks + seg-token + head fine-tuned; GroupKFold-5.
# Env: ARMS, UNFREEZE (12), RES (2), EWIDTH/EHEADS (768/12 for ViT-base).
#SBATCH --job-name=xmodal_dseg
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/dseg_%j.log
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git fetch -q origin v5
SHA="${SHA:-$(git rev-parse origin/v5)}"
git checkout -qf --detach "$SHA" && echo "PINNED $SHA"
nvidia-smi --query-gpu=name --format=csv,noheader

for attempt in 1 2 3 4 5; do
.venv/bin/python - <<'PY' && break
import wandb, os, tarfile, glob
p = wandb.Api().artifact("vineeth-gangaram-penn/xmodal-phased/heldout-mets-51:v0").download("/tmp/ho_art")
os.makedirs("/tmp/ho", exist_ok=True)
tarfile.open(glob.glob("/tmp/ho_art/*.tar")[0]).extractall("/tmp/ho")
print("held-out extracted ->", os.listdir("/tmp/ho"))
PY
echo "wandb download attempt $attempt failed (transient), retrying..."; sleep 15
done
[ -d /tmp/ho/mets_ho ] || { echo "FATAL: held-out download failed after retries"; exit 1; }

ARMS="${ARMS:-v5_both_tumor_s0}"
CK=()
for arm in $ARMS; do
  ls runs/$arm/step_*.pt >/dev/null 2>&1 && CK+=("${arm}=runs/${arm}/step_*.pt")
done
echo "dense-seg (unfreeze=${UNFREEZE:-12} res=${RES:-2}mm) over: $ARMS"
POOL=()
[ -n "${TRAIN_TRACKS:-}" ] && POOL+=(--train-tracks ${TRAIN_TRACKS} --n-train-patients "${NTRAIN:-600}" --n-train-prisms "${NTRAINP:-4}")
.venv/bin/python -u scripts/eval_denseseg.py --data-root /tmp/ho --tracks mets_ho --device cuda \
    --unfreeze "${UNFREEZE:-12}" --res "${RES:-2}" --qsize "${QSIZE:-4}" \
    --head "${HEAD:-conv}" --block "${BLOCK:-11}" \
    --train-root "${TRAIN_ROOT:-~/xmodal/data/brats26}" ${POOL[@]+"${POOL[@]}"} \
    --epochs "${EPOCHS:-30}" --epochs2 "${EPOCHS2:-60}" \
    --enc-width "${EWIDTH:-384}" --enc-heads "${EHEADS:-6}" \
    --checkpoints "${CK[@]}"
echo "=== eval_denseseg done (unfreeze=${UNFREEZE:-12} res=${RES:-2}mm) ==="
