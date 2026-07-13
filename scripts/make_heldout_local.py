"""Reconstruct the held-out METS test set on local disk from the deterministic hash split (no wandb).

The held-out set = mets_train patients with is_heldout(seed=0, frac=0.1). Because split_patients() is a
per-patient md5 hash (list-independent), this set is provably a SUBSET of the pretraining val split -> the
encoder never trained on any of them (contamination_check == 0, verified). Replaces the old
`heldout-mets-51` wandb artifact (project xmodal-phased was deleted); the on-disk split now yields ~116
patients (mets_train grew ~510->1296; superset of the original 51).

Symlinks each held-out patient's niftis into <out>/<pid>/ so evals read them exactly like the artifact.
Usage: python scripts/make_heldout_local.py [/tmp/ho/mets_ho] [data/brats26/mets_train]
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xmodal import holdout as H, data as D  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ho/mets_ho"
src_root = sys.argv[2] if len(sys.argv) > 2 else "data/brats26/mets_train"

mets = D.find_brats_patients(src_root)
_, val = H.split_patients(sorted(mets), seed=0, val_frac=0.1)
os.makedirs(out, exist_ok=True)
n = 0
for p in val:
    srcdir = mets[p]
    dst = os.path.join(out, os.path.basename(srcdir))
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(srcdir, "*.nii.gz")):        # symlink FILES (real dirs) -> glob-safe
        link = os.path.join(dst, os.path.basename(f))
        if not os.path.exists(link):
            os.symlink(os.path.abspath(f), link)
    n += 1
print(f"reconstructed held-out: {n} METS patients -> {out}", flush=True)
