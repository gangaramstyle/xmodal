#!/bin/bash
# Bag-size x spacing 2x2 on the pretext task: source bag {96,256} x {random,grid} -> match_acc/MAE.
# Cleanly tests whether oversized / grid-spaced readout bags degrade the encoder (readout stays
# in-distribution since it IS the pretext task). Latest checkpoint of each arm.
#SBATCH --job-name=xmodal_bagsz
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/bagsz_%j.log
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
echo "bag-size 2x2 over arms: $ARMS"
.venv/bin/python -u scripts/eval_bagsize_ablation.py --data-root /tmp/ho --tracks mets_ho --device cuda \
    --n-src 90 250 --prisms 64 --checkpoints "${CK[@]}"
echo "=== eval_bagsize done ==="
