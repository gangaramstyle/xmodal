#!/bin/bash
# CUBIC A40 phase-0 pretraining. Draft — partition/env finalized live once connected.
#SBATCH --job-name=xmodal_p0
#SBATCH --partition=ltdai          # or 'ai' — either A40 queue is fine
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=xmodal_p0_%j.log
set -euo pipefail

# Repo (public) — clone once, pull each run
XMODAL_DIR="${XMODAL_DIR:-$HOME/xmodal}"
[ -d "$XMODAL_DIR/.git" ] || git clone https://github.com/gangaramstyle/xmodal.git "$XMODAL_DIR"
cd "$XMODAL_DIR"
git pull -q || true

# Env: uv venv with a CUDA torch matching the A40 driver (verified live before first run).
# HF cache on shared storage so compute-node pulls persist across jobs.
export HF_HOME="${HF_HOME:-$XMODAL_DIR/.hf}"
export HF_TOKEN="${HF_TOKEN:-}"           # optional: avoids HF throttling

uv run python scripts/run_phase0.py \
    --patients 300 --holdout 16 --cache-size 64 \
    --batch-size 256 --token-count 128 --steps 20000 \
    --ckpt-dir runs/phase0
