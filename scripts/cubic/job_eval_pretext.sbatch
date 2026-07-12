#!/bin/bash
# Pretext-on-tumor: does tumor-focused training improve the v5 pretext (match_acc + MAE) ON tumor grids?
# Runs eval_pretext_tumor.py on the LATEST checkpoint of each arm (tumor bags vs non-tumor bags).
# Env: ARMS (run dirs), SHA (eval-code commit).
#SBATCH --job-name=xmodal_pretext
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/pretext_%j.log
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
echo "pretext-on-tumor over arms: $ARMS"
.venv/bin/python -u scripts/eval_pretext_tumor.py --data-root /tmp/ho --tracks mets_ho --device cuda \
    --checkpoints "${CK[@]}"
echo "=== eval_pretext done ==="
