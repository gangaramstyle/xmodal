#!/bin/bash
# CPU-only: push raw held-out patients + the best checkpoint to a wandb artifact so molab can run inference itself.
#SBATCH --job-name=xmodal_push
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/push_%j.log
set -uo pipefail
cd ~/xmodal
git fetch -q origin v5; git checkout -qf --detach "$(git rev-parse origin/v5)"

ARM="${ARM:-v5_multisize_vitbase_s0}"; CKPT="runs/${ARM}/step_070000.pt"
[ -f "$CKPT" ] || { echo "FATAL: $CKPT missing"; exit 1; }

# stage NPAT held-out patients (deterministic hash split of mets_train) into a flat dir
STAGE=/tmp/molab_stage; rm -rf $STAGE; mkdir -p $STAGE/patients $STAGE/ckpt
.venv/bin/python - <<PY
import os, glob, shutil
import sys; sys.path.insert(0, "src")
from xmodal import holdout as H, data as D
mets = D.find_brats_patients("data/brats26/mets_train")
_, val = H.split_patients(sorted(mets), seed=0, val_frac=0.1)
n = int(os.environ.get("NPAT", "12")); k = 0
for p in val:
    if k >= n: break
    src = mets[p]; pid = os.path.basename(src)
    dst = f"/tmp/molab_stage/patients/{pid}"; os.makedirs(dst, exist_ok=True)
    for f in glob.glob(f"{src}/*.nii.gz"): shutil.copy(f, dst)
    k += 1
print(f"staged {k} held-out patients")
PY
# slim the checkpoint: keep model+step+cfg, drop the optimizer state (~900MB)
.venv/bin/python - <<PY
import torch
ck = torch.load("$CKPT", map_location="cpu")
torch.save({k: ck[k] for k in ("model", "step", "cfg") if k in ck}, "/tmp/molab_stage/ckpt/${ARM}_step70000_slim.pt")
print("slim ckpt saved")
PY
echo "artifact size: $(du -sh $STAGE | cut -f1)"

.venv/bin/python - <<PY
import wandb
run = wandb.init(project="xmodal-mixed", name="molab-assets-push", job_type="dataset")
art = wandb.Artifact("molab-assets", type="dataset",
                     metadata={"desc": "raw held-out METS patients + multisize-vitbase step70000 ckpt for molab live inference",
                               "arm": "${ARM}", "enc_width": 768, "enc_heads": 12})
art.add_dir("/tmp/molab_stage")
run.log_artifact(art); run.finish()
print("DONE -> vineeth-gangaram-penn/xmodal-mixed/molab-assets:latest")
PY
