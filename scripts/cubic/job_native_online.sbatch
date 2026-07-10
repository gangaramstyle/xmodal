#!/bin/bash
# FORK off native @40k: ONLINE-ENCODER latent (JEPA-style). Same as native_latent BUT WITHOUT
# --freeze-encoder, so the student ENCODER trains on self-MAE + latent cross-prediction while the
# frozen native@40k teacher supplies the (stationary, non-collapsing) target latents. Question:
# does baking the cross-modal-prediction objective into the encoder finally lift the encoder-readout
# F1 off ~0.64? Success metric = molab encoder-embedding F1 moving (NOT latent_cos). Self-MAE stays
# on as the anti-collapse anchor; frozen (not EMA) teacher = no collapse but a fixed target.
#SBATCH --job-name=xmodal_native_online
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/native_online_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git pull -q || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

INIT=runs/native/step_040000.pt          # pinned 40k fork point == the frozen teacher
echo "init-from (native @40k, pinned): $INIT"
[ -f "$INIT" ] || { echo "missing $INIT"; exit 1; }

# self/cross 0 -> straight into latent; teacher snapshots from the 40k init weights (frozen).
# NO --freeze-encoder: encoder trains (self-MAE + latent). Native sampling; gentle 1e-4 fine-tune LR.
.venv/bin/python -u scripts/run_phased.py \
    --batch-size 224 --token-count 128 --cache-size 32 \
    --patch-sizes 4 8 16 --held-size 8 16 --content-blur 3 \
    --orient native --lr 1e-4 \
    --self-steps 0 --cross-steps 0 --latent-steps 40000 \
    --init-from "$INIT" \
    --ckpt-dir runs/native_online \
    --wandb xmodal-phased --wandb-run "native_online_${SLURM_JOB_ID:-manual}"
