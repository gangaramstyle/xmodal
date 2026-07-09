#!/bin/bash
# Latent-mismatch anomaly eval: does 1-cos latent cross-prediction error separate tumor from non-tumor
# patches better than pixel-cross MSE? Runs on native@40k (pixel-cross baseline; its latent_head is
# untrained) and the latest native_latent ckpt (the trained latent signal). Both share the frozen encoder.
#SBATCH --job-name=xmodal_latmm
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/latmm_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git pull -q || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

LATENT=$(ls -t runs/native_latent/step_*.pt 2>/dev/null | head -1)
echo "############ BASELINE: native @40k (read pixel_err row) ############"
.venv/bin/python -u scripts/eval_latent_mismatch.py --checkpoint runs/native/step_040000.pt \
    --orient native --max-patients 60 --prisms-per-patient 8 --patch-sizes 8

echo; echo "############ LATENT: ${LATENT:-<none yet>} (read latent_mismatch row) ############"
[ -n "$LATENT" ] && .venv/bin/python -u scripts/eval_latent_mismatch.py --checkpoint "$LATENT" \
    --orient native --max-patients 60 --prisms-per-patient 8 --patch-sizes 8 || echo "no native_latent checkpoint yet"
