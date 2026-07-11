#!/usr/bin/env python
"""Batch held-out patch-F1 eval: build the cache once, then eval many checkpoints (8mm 4-mod,
strict=False so pre-EMA checkpoints load). For cluster use. --checkpoints are name=glob-path pairs."""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from xmodal import model as M, sampling as S  # noqa: E402
import eval_patch_f1 as EPF  # noqa: E402

MODS = ["t1", "t1c", "t2", "flair"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/patchcache_battery.npz")
    ap.add_argument("--data-root", default="/tmp/ho")
    ap.add_argument("--tracks", nargs="+", default=["mets_ho"])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="name=globpath ...")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    if a.build or not os.path.exists(a.cache):
        print("building cache...", flush=True)
        EPF.build_cache(a.data_root, a.tracks, a.cache, device=dev)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold
    Z = np.load(a.cache); coords = Z["coords"]; labels = Z["labels"].astype(int); groups = Z["groups"].astype(int)

    def ev(ckpt):
        ck = torch.load(ckpt, map_location=dev); step = int(ck.get("step", -1))
        E = M.Phase0Encoder(M.EncoderConfig(width=384, depth=12, heads=6, n_series=8)).to(dev)
        E.load_state_dict(ck["model"], strict=False); E.eval()
        s = 8; cols = {m: [] for m in MODS}
        for g in sorted(set(groups.tolist())):
            idx = np.where(groups == g)[0]; co = torch.as_tensor(coords[idx], device=dev)[None].float()
            sz3 = S.size_to_extent(torch.full((1, len(idx)), float(s), device=dev), 2)
            for m in MODS:
                pt = torch.as_tensor(Z["%d_%s" % (s, m)][idx], device=dev).float()[None, ..., None]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _, lat = E.teacher_readout(pt, co, sz3)
                cols[m].append(lat[0].float().cpu().numpy())
        Fm = {m: np.concatenate(cols[m]) for m in MODS}
        keep = labels != 1; Y = labels[keep]; G = groups[keep]; X = np.concatenate([Fm[m] for m in MODS], -1)[keep]
        yt, yp = [], []
        for tr, te in GroupKFold(5).split(X, Y, G):
            clf = LogisticRegression(max_iter=150, class_weight="balanced").fit(X[tr], Y[tr])
            yt.append(Y[te]); yp.append(clf.predict(X[te]))
        yt = np.concatenate(yt); yp = np.concatenate(yp)
        per = f1_score(yt, yp, labels=[0, 2, 3], average=None, zero_division=0)
        return step, float(per[2]), float(per.mean())

    for spec in a.checkpoints:
        nm, path = spec.split("=", 1)
        g = glob.glob(path)
        if not g:
            print("%s NO_CKPT %s" % (nm, path), flush=True); continue
        try:
            step, enh, mac = ev(sorted(g)[-1])
            print("%s step%d: enh %.3f macro %.3f" % (nm, step, enh, mac), flush=True)
        except Exception:
            import traceback
            print("%s ERR %s" % (nm, traceback.format_exc()[-160:]), flush=True)


if __name__ == "__main__":
    main()
