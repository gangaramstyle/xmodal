#!/bin/bash
# Mid-training held-out patch-F1 eval of the in-flight mixed runs (v1 & v2 seed-0, EMA on/off).
# Uses eval_battery --mixed so the readout conditions the encoder with per-patch series_ids the SAME
# way training did (Site A) — the key thing to validate. Also runs ONE checkpoint WITHOUT --mixed as a
# sanity check that the series conditioning actually changes the readout (F1 should differ).
# Globs pick the latest step_*.pt per run, so this works at any training progress.
#SBATCH --job-name=xmodal_evalmix
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/evalmix_%j.log
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git checkout -q main && git pull -q || true
nvidia-smi --query-gpu=name --format=csv,noheader

# pull the 51-METS held-out set from wandb, extract
.venv/bin/python - <<'PY'
import wandb, os, tarfile, glob
p = wandb.Api().artifact("vineeth-gangaram-penn/xmodal-phased/heldout-mets-51:v0").download("/tmp/ho_art")
os.makedirs("/tmp/ho", exist_ok=True)
tarfile.open(glob.glob("/tmp/ho_art/*.tar")[0]).extractall("/tmp/ho")
print("extracted ->", os.listdir("/tmp/ho"))
PY

# --mixed: correct readout for the series-conditioned encoders (builds the cache once)
.venv/bin/python -u scripts/eval_battery.py \
    --data-root /tmp/ho --tracks mets_ho --build --device cuda --mixed \
    --checkpoints \
      "v1_ema=runs/mixed_ema_s0/step_0*.pt" \
      "v1_noema=runs/mixed_noema_s0/step_0*.pt" \
      "v2_ema=runs/mixed_v2_ema_s0/step_0*.pt" \
      "v2_noema=runs/mixed_v2_noema_s0/step_0*.pt"

# sanity: the MOST-TRAINED run (v1_ema) WITHOUT --mixed -> should differ if series conditioning matters
# for the frozen-encoder readout (v2@1k earlier was too early to tell)
.venv/bin/python -u scripts/eval_battery.py \
    --data-root /tmp/ho --tracks mets_ho --device cuda \
    --checkpoints "v1_ema_NOMIX=runs/mixed_ema_s0/step_0*.pt"
echo "=== eval_mixed done ==="
