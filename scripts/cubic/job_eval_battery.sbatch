#!/bin/bash
# On-cluster held-out patch-F1 eval (reliable; molab boxes keep dying). Installs sklearn, pulls the
# 51-METS held-out set from wandb, builds the cache, and evals the battery: native@40k, EMA-color
# @5k/25k/30k, pixel-soft @25k x3, hard @25k x3. Checkpoints read from runs/ on disk.
#SBATCH --job-name=xmodal_evalbat
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/evalbat_%j.log
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git pull -q || true
nvidia-smi --query-gpu=name --format=csv,noheader

# sklearn (not in the base venv)
.venv/bin/python -m pip install -q scikit-learn 2>&1 | tail -1 || true

# pull the 51-METS held-out set (same set the molab baselines used) from wandb, extract
.venv/bin/python - <<'PY'
import wandb, os
p = wandb.Api().artifact("vineeth-gangaram-penn/xmodal-phased/heldout-mets-51:v0").download("/tmp/ho_art")
import tarfile, glob
tar = glob.glob("/tmp/ho_art/*.tar")[0]
os.makedirs("/tmp/ho", exist_ok=True)
tarfile.open(tar).extractall("/tmp/ho")
print("extracted heldout ->", os.listdir("/tmp/ho"))
PY

.venv/bin/python -u scripts/eval_battery.py \
    --data-root /tmp/ho --tracks mets_ho --build --device cuda \
    --checkpoints \
      native40k=runs/native/step_040000.pt \
      ema_5k=runs/emacolor_s0/step_005000.pt \
      ema_25k=runs/emacolor_s0/step_025000.pt \
      ema_30k=runs/emacolor_s0/step_030000.pt \
      softpix_s0_25k=runs/softpix_s0/step_025000.pt \
      softpix_s1_25k=runs/softpix_s1/step_025000.pt \
      softpix_s2_25k=runs/softpix_s2/step_025000.pt \
      hard_s0_25k=runs/hardmatch_s0/step_025000.pt \
      hard_s1_25k=runs/hardmatch_s1/step_025000.pt \
      hard_s2_25k=runs/hardmatch_s2/step_025000.pt
echo "=== eval_battery done ==="
