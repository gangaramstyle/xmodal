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

# self-chain: `short` caps at 90min, so before exiting, relaunch a fresh worker if tickets remain — this spans
# a ~12h budget WITHOUT login (compute nodes can sbatch). Bounded by CHAIN_MAX generations; stops when queue empty.
GEN=${GEN:-0}; CHAIN_MAX=${CHAIN_MAX:-8}
NPEND=$(ls /cbica/home/gangarav/xmodal/tickets/pending/*.json 2>/dev/null | wc -l)
if [ "${CHAIN:-0}" = "1" ] && [ "$GEN" -lt "$CHAIN_MAX" ] && [ "$NPEND" -gt 0 ]; then
  GEN=$((GEN+1)) CHAIN=1 CHAIN_MAX="$CHAIN_MAX" CKPT="$CKPT" TIER="${TIER:-big}" WALL="${WALL:-5400}" \
    sbatch --array=0-0 --partition=short --gres=gpu:p100:1 --time=01:30:00 --job-name=awork \
      /cbica/home/gangarav/xmodal/scripts/ablation/worker.sbatch
fi
