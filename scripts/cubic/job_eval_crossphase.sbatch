#!/bin/bash
# Cross-phase trajectory eval: does EMA-color's end-of-self lead over the no-EMA baseline HOLD, GROW,
# or FADE through the cross phase? Eval EMA-color_s0 vs native (full curriculum, no EMA) at matched
# steps 25k (end of self) / 30k / 35k (mid cross). Same torch-probe readout as job_eval_battery.
#SBATCH --job-name=xmodal_evalcross
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/evalcross_%j.log
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git pull -q || true
nvidia-smi --query-gpu=name --format=csv,noheader

# Reconstruct the held-out METS set from disk via the deterministic hash split (no wandb; the old
# xmodal-phased/heldout-mets-51 artifact was deleted). Provably disjoint from pretraining (see make_heldout_local.py).
.venv/bin/python scripts/make_heldout_local.py /tmp/ho/mets_ho data/brats26/mets_train
[ -d /tmp/ho/mets_ho ] && [ -n "$(ls -A /tmp/ho/mets_ho 2>/dev/null)" ] || { echo "FATAL: held-out reconstruction failed"; exit 1; }

.venv/bin/python -u scripts/eval_battery.py \
    --data-root /tmp/ho --tracks mets_ho --build --device cuda \
    --checkpoints \
      nat_25k=runs/native/step_025000.pt \
      nat_30k=runs/native/step_030000.pt \
      nat_35k=runs/native/step_035000.pt \
      ema_25k=runs/emacolor_s0/step_025000.pt \
      ema_30k=runs/emacolor_s0/step_030000.pt \
      ema_35k=runs/emacolor_s0/step_035000.pt
echo "=== eval_crossphase done ==="
