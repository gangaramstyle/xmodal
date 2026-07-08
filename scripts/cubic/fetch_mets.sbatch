#!/bin/bash
# Download BraTS-2026 challenge seg data (METS + PED + GoAT) from Synapse. CPU node, no GPU.
#SBATCH --job-name=xmodal_fetch
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/fetch_%j.log
set -euo pipefail

cd ~/xmodal
git pull -q || true
# Fold the earlier METS download (data/mets/{train,labels}) into the unified layout so it's skipped.
mkdir -p data/brats26
[ -d data/mets/train ]  && [ ! -e data/brats26/mets_train ]  && mv data/mets/train  data/brats26/mets_train  || true
[ -d data/mets/labels ] && [ ! -e data/brats26/mets_labels ] && mv data/mets/labels data/brats26/mets_labels || true
~/xmodal/.venv/bin/python -u scripts/cubic/fetch_mets.py ~/xmodal/data/brats26
