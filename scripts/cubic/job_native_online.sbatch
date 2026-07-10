#!/bin/bash
# FORK off native @40k: ONLINE-ENCODER latent (JEPA-style). Like job_native_latent but WITHOUT
# --freeze-encoder, so the student ENCODER trains on self-MAE + latent cross-prediction against the
# FROZEN native@40k teacher's target latents (self-MAE grounds the encoder; frozen rich target also
# blocks collapse). Same lr 6e-4 / 40k init / teacher as native_latent for comparability. Question:
# does baking cross-modal prediction into the encoder lift the encoder-readout F1 off ~0.64?
# Success metric = molab encoder-embedding F1 moving (NOT latent_cos).
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

# Like job_native_latent but NO --freeze-encoder: latent phase runs self-MAE + latent, encoder TRAINS.
# Same lr 6e-4 / 40k init / frozen teacher as native_latent (only freeze + self-MAE differ).
.venv/bin/python -u scripts/run_phased.py \
    --batch-size 224 --token-count 128 --cache-size 32 \
    --orient native --lr 6e-4 \
    --self-steps 0 --cross-steps 0 --latent-steps 40000 \
    --init-from "$INIT" \
    --ckpt-dir runs/native_online \
    --wandb xmodal-phased --wandb-run "native_online_${SLURM_JOB_ID:-manual}"
