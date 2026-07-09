#!/bin/bash
# FORK off native @ 40k: PHASE-4 FAITHFUL latent cross-prediction (reproduces tcia_crossmodal July-5 phase4).
# Freeze the native-40k encoder; train ONLY the decoder + latent_head to cross-predict the FROZEN teacher's
# target-modality per-patch latents (1 - cosine). Teacher == the frozen encoder, so its targets are stationary
# and its per-patch embeddings (what our downstream MLP-F1 reads) do NOT move -- the payoff is the decoder's
# latent cross-prediction / mismatch anomaly signal, evaluated separately (viz_latent-style).
#SBATCH --job-name=xmodal_native_latent
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/native_latent_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git pull -q || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

INIT=runs/native/step_040000.pt          # pinned to the 40k fork point (NOT latest)
echo "init-from (native @40k, pinned): $INIT"
[ -f "$INIT" ] || { echo "missing $INIT"; exit 1; }

# self/cross steps = 0 -> straight into latent at step 0; teacher snapshots from the 40k init weights.
# Sampling left at native defaults (patch {4,8,16} / held {8,16} / prism {32,64,128}) to match the frozen
# encoder's training distribution. lr 6e-4 + warmup (phase-4 values).
.venv/bin/python -u scripts/run_phased.py \
    --batch-size 224 --token-count 128 --cache-size 32 \
    --orient native --lr 6e-4 \
    --self-steps 0 --cross-steps 0 --latent-steps 40000 \
    --freeze-encoder \
    --init-from "$INIT" \
    --ckpt-dir runs/native_latent \
    --wandb xmodal-phased --wandb-run "native_latent_${SLURM_JOB_ID:-manual}"
