#!/bin/bash
# FORK off native_resume: continued CROSS-phase training focused on 4mm input + 4mm matching/recon targets.
# Inits from the latest runs/native checkpoint (weights only, fresh schedule); the frozen teacher (series-CLS)
# snapshots from those same latest weights at cross-start. Large prisms {32,64,128} give spatial context via
# attention across the fine 4mm tokens. Parallel to native_resume ({4,8,16}in/{8,16}out) which stays as control.
#SBATCH --job-name=xmodal_native4mm
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/native4mm_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git pull -q || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

INIT=$(ls -t runs/native/step_*.pt 2>/dev/null | head -1)
echo "init-from (latest native): ${INIT:-<none found>}"
[ -n "$INIT" ] || { echo "no checkpoint in runs/native"; exit 1; }

.venv/bin/python -u scripts/run_phased.py \
    --batch-size 224 --token-count 128 --cache-size 32 \
    --patch-sizes 4 --held-size 4 --prism-choices 32 64 128 --content-blur 3 \
    --orient native --lr 1e-4 \
    --self-steps 0 --cross-steps 20000 --latent-steps 0 \
    --init-from "$INIT" \
    --ckpt-dir runs/native4mm \
    --wandb xmodal-phased --wandb-run "native4mm_${SLURM_JOB_ID:-manual}"
