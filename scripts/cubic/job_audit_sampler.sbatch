#!/bin/bash
# Sampler-only audit (NO training): does real BraTS support the v3 structured sampling? Stratified by
# track x prism x source-size. Sets validity thresholds + register budget before any model change.
#SBATCH --job-name=xmodal_audit
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/audit_%j.log
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git checkout -q mixed-v3 && git pull -q origin mixed-v3 || true
echo "COMMIT $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader

.venv/bin/python -u scripts/audit_sampler.py \
    --data-root ~/xmodal/data/brats26 --tracks mets_train ped_train goat_gt goat_nogt \
    --patients-per-track 30 --bags 1500 --device cuda
echo "=== audit done ==="
