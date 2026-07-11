#!/bin/bash
# Representation experiment (SOFT-PIXEL arm): from-scratch SELF-phase pretraining with the COLLAPSE-SAFE
# similarity-softened matching target -- similarity from FIXED raw blurred pixels (RBF), NOT the trainable
# color_head. This is the fix for the model-based soft matching, which collapsed (from a random-init
# color_head the similarity is ~uniform -> the target is ~uniform -> the matcher never learns to
# discriminate; self_match_acc cratered to ~0.04 and encoder-F1 fell BELOW hard). The fixed external
# target cannot go uniform-from-random-init, so it cannot collapse that way. Compare vs the existing
# hardmatch_s{0,1,2} runs (identical config, no softening). Same config/seed -> only the softened target
# differs. Success = encoder held-out patch-F1 >= hard.
#SBATCH --job-name=xmodal_softpix
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/softpix_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/xmodal
git pull -q || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/python -u scripts/run_phased.py \
    --batch-size 224 --token-count 128 --cache-size 32 \
    --patch-sizes 4 8 16 --held-size 8 16 --content-blur 3 \
    --orient native --seed ${SEED:-0} \
    --self-steps 25000 --cross-steps 0 --latent-steps 0 \
    --soft-match-tau 0.15 --soft-match-sim pixel \
    --ckpt-dir runs/softpix_s${SEED:-0} \
    --wandb xmodal-phased --wandb-run "softpix_s${SEED:-0}_${SLURM_JOB_ID:-manual}"
