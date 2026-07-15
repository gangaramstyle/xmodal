#!/bin/bash
# CPU array: precompute the prism cache. Chunk patients by SLURM_ARRAY_TASK_ID (stride). ~16s/patient.
# NO git checkout here (24+ concurrent tasks would race) — pre-checkout ~/xmodal before submitting.
#SBATCH --job-name=prism_cache
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/cbica/home/gangarav/xmodal/pcache_%A_%a.log
set -uo pipefail
export OMP_NUM_THREADS=4
cd ~/xmodal
.venv/bin/python scripts/build_prism_cache.py --split "${SPLIT:-train}" --n-patients "${NPAT:-648}" \
    --n-tumor "${NTUMOR:-12}" --n-neg "${NNEG:-12}" --n-src "${NSRC:-4096}" \
    --out "${OUT:-/cbica/home/gangarav/xmodal/prism_cache}"
