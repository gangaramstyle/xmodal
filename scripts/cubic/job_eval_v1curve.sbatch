#!/bin/bash
# Full F1-vs-step curve for v1_noema: eval EVERY checkpoint (step_*.pt) on the held-out set, one shared
# cache. Many points average out per-eval probe noise so a slow trend (or a true plateau) is visible.
#SBATCH --job-name=xmodal_v1curve
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/v1curve_%j.log
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git fetch -q origin mixed-v4 && git checkout -q mixed-v4 && git pull -q origin mixed-v4
echo "COMMIT $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader

.venv/bin/python - <<'PY'
import wandb, os, tarfile, glob
p = wandb.Api().artifact("vineeth-gangaram-penn/xmodal-phased/heldout-mets-51:v0").download("/tmp/ho_art")
os.makedirs("/tmp/ho", exist_ok=True)
tarfile.open(glob.glob("/tmp/ho_art/*.tar")[0]).extractall("/tmp/ho")
print("extracted ->", os.listdir("/tmp/ho"))
PY

# enumerate every checkpoint as name=exact-path (eval_battery globs an exact path to itself)
RUN="${RUN:-runs/mixed_noema_s0}"
CKPTS=""
for f in $(ls -v $RUN/step_*.pt); do s=$(basename "$f" .pt | sed 's/step_0*/s/'); CKPTS="$CKPTS ${s}=$f"; done
echo "RUN=$RUN evaling $(echo $CKPTS | wc -w) checkpoints"

# shared held-out cache (patch content is checkpoint-independent) -> build once, reuse across runs
CACHE=/tmp/patchcache_curve.npz
BUILD=""; [ -f "$CACHE" ] || BUILD="--build"
.venv/bin/python -u scripts/eval_battery.py \
    --cache "$CACHE" --data-root /tmp/ho --tracks mets_ho $BUILD --device cuda --mixed \
    --checkpoints $CKPTS
echo "=== curve done ($RUN) ==="
