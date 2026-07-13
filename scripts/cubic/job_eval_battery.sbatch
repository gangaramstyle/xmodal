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

# (readout is now GPU torch logistic regression -- no sklearn dependency)

# pull the 51-METS held-out set (same set the molab baselines used) from wandb, extract
# Reconstruct the held-out METS set from disk via the deterministic hash split (no wandb; the old
# xmodal-phased/heldout-mets-51 artifact was deleted). Provably disjoint from pretraining (see make_heldout_local.py).
.venv/bin/python scripts/make_heldout_local.py /tmp/ho/mets_ho data/brats26/mets_train
[ -d /tmp/ho/mets_ho ] && [ -n "$(ls -A /tmp/ho/mets_ho 2>/dev/null)" ] || { echo "FATAL: held-out reconstruction failed"; exit 1; }

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
