#!/bin/bash
# RESUME the native run from its latest disk checkpoint, WITH the corrupt-file loader fix.
# Same config/phases as job_native.sbatch so the LR schedule + phase positions line up exactly.
#SBATCH --job-name=xmodal_native_rs
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/native_resume_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git pull -q || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RESUME=$(ls -t runs/native/step_*.pt 2>/dev/null | head -1)
echo "resuming native from: ${RESUME:-<none found>}"
[ -n "$RESUME" ] || { echo "no checkpoint in runs/native"; exit 1; }

.venv/bin/python -u scripts/run_phased.py \
    --batch-size 224 --token-count 128 --cache-size 32 \
    --patch-sizes 4 8 16 --held-size 8 16 --content-blur 3 \
    --orient native \
    --self-steps 25000 --cross-steps 25000 --latent-steps 20000 \
    --resume "$RESUME" \
    --ckpt-dir runs/native \
    --wandb xmodal-phased --wandb-run "native_resume_${SLURM_JOB_ID:-manual}"
