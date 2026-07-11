#!/bin/bash
# Held-out patch-F1 eval of v1/v2 (no scan-context) AND v3 (scan-context) together. eval_battery reads
# scan_context PER checkpoint from its saved cfg and builds/loads each model accordingly + passes per-scan
# stats, so this is a fair single-cache comparison. Rebuilds the cache with sstats (--build).
#SBATCH --job-name=xmodal_evalv3
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/evalv3_%j.log
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

# rebuild cache (now stores per-scan sstats) and eval all three families with --mixed (scan_context auto per ckpt)
.venv/bin/python -u scripts/eval_battery.py \
    --cache /tmp/patchcache_v3.npz --data-root /tmp/ho --tracks mets_ho --build --device cuda --mixed \
    --checkpoints \
      "v1_noema=runs/mixed_noema_s0/step_0*.pt" \
      "v2_noema=runs/mixed_v2_noema_s0/step_0*.pt" \
      "v3=runs/mixed_v3_s0/step_0*.pt"
echo "=== eval_v3 done ==="
