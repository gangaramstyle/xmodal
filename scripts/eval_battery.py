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
    Z = np.load(a.cache); coords = Z["coords"]; labels = Z["labels"].astype(int); groups = Z["groups"].astype(int)
    CLASSES = [0, 2, 3]                                              # non-tumor, edema, ET (necrosis dropped)

    def torch_probe(X, Y, G, steps=300):
        """GPU class-balanced multinomial logistic regression, GroupKFold-5. Replaces sklearn (no dep,
        faster). Returns per-class F1 [non-tumor, edema, ET]. enh=ET, macro=mean."""
        cmap = {c: i for i, c in enumerate(CLASSES)}
        Xt = torch.tensor(X, device=dev, dtype=torch.float32)
        Yt = torch.tensor([cmap[int(y)] for y in Y], device=dev)
        gids = sorted(set(G.tolist())); fold_of = {g: i % 5 for i, g in enumerate(gids)}
        foldY = np.array([fold_of[int(g)] for g in G])
        pred = np.zeros(len(Y), int)
        for f in range(5):
            tr = torch.tensor(foldY != f, device=dev); te = torch.tensor(foldY == f, device=dev)
            Xtr, Ytr = Xt[tr], Yt[tr]
            mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
            cnt = torch.bincount(Ytr, minlength=len(CLASSES)).float()
            w = (len(Ytr) / (len(CLASSES) * cnt.clamp(min=1))).to(dev)
            clf = torch.nn.Linear(Xt.shape[1], len(CLASSES)).to(dev)
            opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
            for _ in range(steps):
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(clf((Xtr - mu) / sd), Ytr, weight=w)
                loss.backward(); opt.step()
            with torch.no_grad():
                pred[foldY == f] = clf((Xt[te] - mu) / sd).argmax(1).cpu().numpy()
        Yn = np.array([cmap[int(y)] for y in Y]); f1 = []
        for c in range(len(CLASSES)):
            tp = ((pred == c) & (Yn == c)).sum(); fp = ((pred == c) & (Yn != c)).sum(); fn = ((pred != c) & (Yn == c)).sum()
            p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9); f1.append(2 * p * r / (p + r + 1e-9))
        return f1

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
        f1 = torch_probe(X, Y, G)
        return step, float(f1[2]), float(sum(f1) / len(f1))

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
