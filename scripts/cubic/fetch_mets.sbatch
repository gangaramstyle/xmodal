#!/bin/bash
# Download BraTS-2025 MET training data + corrected labels from Synapse. CPU node (no GPU).
#SBATCH --job-name=xmodal_fetch_mets
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/cbica/home/gangarav/xmodal/fetch_mets_%j.log
set -euo pipefail
ulimit -t unlimited 2>/dev/null || true       # avoid the CPU-time cap that killed the GPU job

cd ~/xmodal
git pull -q || true
~/xmodal/.venv/bin/python -u scripts/cubic/fetch_mets.py ~/xmodal/data/mets
