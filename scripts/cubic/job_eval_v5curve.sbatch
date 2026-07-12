#!/bin/bash
# Full-curve held-out patch-F1 eval of the v5 (3D-cube) arms. Builds the 8^3 cube cache ONCE from the
# 51-METS held-out set, then evals EVERY step_*.pt of each arm (averages the per-eval probe noise, and
# shows whether F1 is still climbing). Torch-native probe (no sklearn) -> runs on the CUBIC venv.
# Compare arms (both_tumor vs both = tumor-focus; both_tumor vs mae_tumor = ordering) and vs the ~0.75 ceiling.
# Env: ARMS (space-sep run dirs under runs/), SHA (eval-code commit to pin).
#SBATCH --job-name=xmodal_evalv5
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/evalv5_%j.log
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git fetch -q origin v5
SHA="${SHA:-$(git rev-parse origin/v5)}"
git checkout -qf --detach "$SHA" && echo "PINNED $SHA"   # -f: discard any stray local changes in the shared tree
nvidia-smi --query-gpu=name --format=csv,noheader

# pull the 51-METS held-out set from wandb, extract
.venv/bin/python - <<'PY'
import wandb, os, tarfile, glob
p = wandb.Api().artifact("vineeth-gangaram-penn/xmodal-phased/heldout-mets-51:v0").download("/tmp/ho_art")
os.makedirs("/tmp/ho", exist_ok=True)
tarfile.open(glob.glob("/tmp/ho_art/*.tar")[0]).extractall("/tmp/ho")
print("held-out extracted ->", os.listdir("/tmp/ho"))
PY

# enumerate EVERY checkpoint of each arm as its own name=path (full curve)
ARMS="${ARMS:-v5_both_tumor_s0 v5_both_s0 v5_mae_tumor_s0}"
CK=()
for arm in $ARMS; do
  for f in runs/$arm/step_*.pt; do
    [ -e "$f" ] && CK+=("${arm}@$(basename "$f" .pt)=$f")
  done
done
echo "evaluating ${#CK[@]} checkpoints across arms: $ARMS"
[ ${#CK[@]} -eq 0 ] && { echo "FATAL: no checkpoints found under runs/{$ARMS}"; exit 1; }

# --cube 8 builds the cube cache once (first ckpt), then reuses it; --mixed auto-on for cube.
# SAMPLING=random (train-like sparse) or grid (dense full coverage) -> the train->eval domain-gap test.
SAMPLING="${SAMPLING:-random}"; PROBE="${PROBE:-linear}"
.venv/bin/python -u scripts/eval_battery.py \
    --data-root /tmp/ho --tracks mets_ho --build --cube 8 --sampling "$SAMPLING" --probe "$PROBE" --device cuda \
    --cache "/tmp/patchcache_v5cube_${SAMPLING}.npz" \
    --checkpoints "${CK[@]}"
echo "=== eval_v5curve done (sampling=$SAMPLING probe=$PROBE) ==="
