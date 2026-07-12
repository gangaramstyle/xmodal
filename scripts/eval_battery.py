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
    ap.add_argument("--mixed", action="store_true",
                    help="mixed-modality checkpoints: pass per-patch series_ids to teacher_readout so the "
                         "encoder is conditioned the SAME way training was (Site A). Off for legacy phased ckpts.")
    ap.add_argument("--cube", type=int, default=0,
                    help="v5 3D-cube encoders: cube voxel grid (e.g. 8). 0 = legacy 2.5D slabs.")
    ap.add_argument("--sampling", choices=["random", "grid"], default="random",
                    help="random = train-like sparse foreground centers; grid = dense full-coverage lattice.")
    ap.add_argument("--probe", choices=["linear", "mlp"], default="linear",
                    help="readout head: linear logreg (~0.75-comparable) or 1-hidden-layer MLP.")
    ap.add_argument("--probe-hidden", type=int, default=256, help="MLP hidden width (--probe mlp).")
    a = ap.parse_args()
    dev = a.device
    if a.build or not os.path.exists(a.cache):
        print("building cache (sampling=%s cube=%d)..." % (a.sampling, a.cube), flush=True)
        EPF.build_cache(a.data_root, a.tracks, a.cache, device=dev, cube=a.cube, sampling=a.sampling)
    Z = np.load(a.cache); coords = Z["coords"]; labels = Z["labels"].astype(int); groups = Z["groups"].astype(int)
    cube = int(a.cube or (int(Z["cube"]) if "cube" in Z else 0))     # cache remembers its geometry
    mixed = a.mixed or bool(cube)                                    # v5 (cube) always uses series conditioning
    CLASSES = [0, 2, 3]                                              # non-tumor, edema, ET (necrosis dropped)

    def torch_probe(X, Y, G, steps=300, probe="linear", hidden=256):
        """GPU class-balanced GroupKFold-5 readout (no sklearn). probe='linear' = multinomial logreg
        (baseline, ~0.75-comparable); probe='mlp' = 1-hidden-layer MLP head. Returns per-class F1
        [non-tumor, edema, ET]. enh=ET, macro=mean."""
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
            if probe == "mlp":
                clf = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], hidden), torch.nn.ReLU(),
                                          torch.nn.Linear(hidden, len(CLASSES))).to(dev)
            else:
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
        ecfg = M.EncoderConfig(width=384, depth=12, heads=6, n_series=8,
                               patch_grid=(cube, cube, cube) if cube else None)
        E = M.Phase0Encoder(ecfg).to(dev)
        E.load_state_dict(ck["model"], strict=False); E.eval()
        s = 8; cols = {m: [] for m in MODS}                          # 8mm patch (v5 trained 4&8mm cubes)
        for g in sorted(set(groups.tolist())):
            idx = np.where(groups == g)[0]; co = torch.as_tensor(coords[idx], device=dev)[None].float()
            if cube:
                sz3 = torch.full((1, len(idx), 3), float(s), device=dev)                 # isotropic cube extent
            else:
                sz3 = S.size_to_extent(torch.full((1, len(idx)), float(s), device=dev), 2)
            for mi, m in enumerate(MODS):
                arr = torch.as_tensor(Z["%d_%s" % (s, m)][idx], device=dev).float()
                pt = arr[None] if cube else arr[None, ..., None]                          # cube [1,n,V,V,V]; slab [1,n,V,V,1]
                sid = torch.full((1, len(idx)), mi, device=dev, dtype=torch.long) if mixed else None
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _, lat = E.teacher_readout(pt, co, sz3, sid)
                cols[m].append(lat[0].float().cpu().numpy())
        Fm = {m: np.concatenate(cols[m]) for m in MODS}
        keep = labels != 1; Y = labels[keep]; G = groups[keep]; X = np.concatenate([Fm[m] for m in MODS], -1)[keep]
        f1 = torch_probe(X, Y, G, probe=a.probe, hidden=a.probe_hidden)
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
