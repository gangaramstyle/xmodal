#!/bin/bash
# GPU: train + eval a readout from the prism cache (fast, no rebuild). Per-task result CSV (merge after).
# NO git checkout (pre-checkout before submitting a sweep array).
#SBATCH --job-name=abl_train
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/abltrain_%A_%a.log
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/xmodal
CK=$(ls runs/v5_multisize_vitbase_s0/step_070000.pt)
.venv/bin/python scripts/train_from_cache.py \
    --cache "${CACHE:-/cbica/home/gangarav/xmodal/prism_cache}" \
    --val-cache "${VALCACHE:-/cbica/home/gangarav/xmodal/prism_cache_val}" --ckpt "$CK" \
    --n-patients "${NPAT:-100}" --n-tumor "${NTUMOR:-6}" --n-neg "${NNEG:-6}" \
    --sampling "${SAMPLING:-random}" --n-src "${NSRC:-2048}" \
    --epochs "${EPOCHS:-20}" --epochs2 "${EPOCHS2:-40}" --unfreeze "${UNFREEZE:-12}" \
    --results "${RESULTS:-/cbica/home/gangarav/xmodal/abl}_${SLURM_ARRAY_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0}.csv"
