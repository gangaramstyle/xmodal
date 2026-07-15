#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/cbica/home/gangarav/xmodal/awork_%A_%a.log
# Stage-2 conv embeddings now go to node-local $TMPDIR (mmap), so we no longer need to hold them in RAM.
# 32G is page-cache headroom (never OOMs — overflow just re-reads from local disk).
# partition / --gres=gpu:<type>:1 / --time / --mem are passed at submit time (see launch below).
# TIER (small|big) and WALL (seconds, match --time) come from the environment.
set -e
cd /cbica/home/gangarav/xmodal
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/ablation/worker.py \
  --root /cbica/home/gangarav/xmodal/tickets \
  --ckpt "${CKPT}" \
  --cache /cbica/home/gangarav/xmodal/prism_cache \
  --val-cache /cbica/home/gangarav/xmodal/prism_cache_val \
  --gpu-tier "${TIER:-big}" \
  --wall "${WALL:-5400}"
