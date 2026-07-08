#!/bin/bash
# CUBIC A40 ABLATION: same phased run but ONE size per bag (--size-per-bag), i.e. no within-bag
# scale mixing. Compares against the mixed-size run to isolate the value of mixing scales per forward.
#SBATCH --job-name=xmodal_ablation
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/phased_ablation_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

cd ~/xmodal
git pull -q || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/python -u scripts/run_phased.py \
    --batch-size 256 --token-count 128 --cache-size 40 \
    --self-steps 25000 --cross-steps 25000 --latent-steps 20000 \
    --size-per-bag \
    --ckpt-dir runs/phased_ablation \
    --wandb xmodal-phased --wandb-run "ablation_perbag_a40_${SLURM_JOB_ID:-manual}"
