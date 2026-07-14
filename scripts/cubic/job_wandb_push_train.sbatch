#!/bin/bash
# CPU-only: push the METS TRAIN split (held-out val excluded) to a wandb artifact `mets-train`, so molab
# can pull the large readout-training pool (companion to molab-assets = 116 val + ckpt). Symlink-staged
# (no local copy of ~29GB); wandb reads file content through the links.
#SBATCH --job-name=xmodal_push_train
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/cbica/home/gangarav/xmodal/push_train_%j.log
set -uo pipefail
cd ~/xmodal
git fetch -q origin v5; git checkout -qf --detach "$(git rev-parse origin/v5)"
nvidia-smi >/dev/null 2>&1 || echo "(cpu node, no gpu — fine)"

STAGE=/tmp/molab_train_stage; rm -rf $STAGE; mkdir -p $STAGE/patients
.venv/bin/python - <<'PY'
import os, glob, sys
sys.path.insert(0, "src")
from xmodal import holdout as H, data as D
mets = D.find_brats_patients("data/brats26/mets_train")            # {pid: dir}
train, val = H.split_patients(sorted(mets), seed=0, val_frac=0.1)  # deterministic; val = held-out (in molab-assets)
print(f"mets_train total {len(mets)} | train split {len(train)} | val (excluded) {len(val)}", flush=True)
stage = "/tmp/molab_train_stage/patients"; n = 0
for p in train:
    src = mets[p]; pid = os.path.basename(src); dst = f"{stage}/{pid}"; os.makedirs(dst, exist_ok=True)
    for f in glob.glob(f"{src}/*.nii.gz"):
        link = f"{dst}/{os.path.basename(f)}"
        if not os.path.exists(link):
            os.symlink(os.path.abspath(f), link)
    n += 1
print(f"symlinked {n} train patients -> {stage}", flush=True)
import wandb
run = wandb.init(project="xmodal-mixed", name="mets-train-push", job_type="dataset")
art = wandb.Artifact("mets-train", type="dataset",
                     metadata={"desc": "METS pretrain-split patients (deterministic hash split, held-out val excluded) "
                                       "for molab large-pool readout training", "n_patients": n, "seed": 0, "val_frac": 0.1})
art.add_dir("/tmp/molab_train_stage")
run.log_artifact(art); run.finish()
print(f"DONE -> vineeth-gangaram-penn/xmodal-mixed/mets-train:latest ({n} patients)", flush=True)
PY
echo "=== push_train done ==="
