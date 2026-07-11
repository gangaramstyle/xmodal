#!/usr/bin/env python
"""Held-out patch-level ENCODER-F1 (the trusted molab readout, ported faithfully from the notebook
_build_cache / _fast_eval cells). Builds a patch cache ONCE (K patches/patient x {4,8,16}mm x 4
modalities + 4mm-footprint labels + patient groups), then per checkpoint runs teacher_readout per
modality, concats the 4 modalities, and does LogisticRegression GroupKFold-5 -> per-class F1.

Labels (4mm footprint, priority ET>NCR>ED): tumor if >25% seg; then ET if ET/tumor>=0.15, else NCR
if NCR/tumor>=0.15, else edema. Necrosis (class 1) dropped at eval. enh-F1 = ET (class 3); macro-F1 =
mean over {non-tumor, edema, ET}. Needs sklearn (present in molab, not the CUBIC venv) -> run in molab.
"""
import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xmodal import data as D, model as M, sampling as S  # noqa: E402

MODS = ["t1", "t1c", "t2", "flair"]
SIZES = [4, 8, 16]


def naxis(sc):
    shp = np.asarray(sc.volume.shape, float)
    return int(np.argmax(np.abs(shp - np.median(shp))))          # outlier-shaped axis = native through-plane


def build_cache(data_root, tracks, out, *, K=700, LS=4, seed=0, device="cuda"):
    torch.manual_seed(seed)
    dirs = []
    for tr in tracks:
        dirs += sorted(glob.glob(os.path.join(os.path.expanduser(data_root), tr, "BraTS-*")))
    store = {f"{s}_{m}": [] for s in SIZES for m in MODS}
    COORD, LAB, GRP = [], [], []
    used = 0
    for gi, d in enumerate(dirs):
        pid = os.path.basename(d)
        segp = glob.glob(f"{d}/{pid}-seg.nii.gz")
        if not segp:
            continue
        try:
            b = D.load_local_bundle(pid, d, device=device)[0]
        except Exception:
            continue
        sc0 = b["t1c"]; thick = naxis(sc0); unit = S.slab_unit_offsets(thick, 16, device)
        c = sc0.foreground_mm[torch.randint(sc0.foreground_mm.shape[0], (K,), device=device)]; cm = c.mean(0)
        shp = torch.as_tensor(sc0.volume.shape, device=device)
        segt = torch.as_tensor(np.asarray(nib.load(segp[0]).get_fdata()), dtype=torch.int16, device=device)
        vi = S.mixed_bag_vox(sc0, c[None], torch.full((1, K), float(LS), device=device), unit).round().long()
        vi = vi.clamp(min=torch.zeros(3, device=device, dtype=torch.long), max=shp - 1)
        sl = segt[vi[..., 0], vi[..., 1], vi[..., 2]].reshape(K, -1)
        tf = (sl > 0).float().mean(1).cpu().numpy()
        c1 = (sl == 1).float().mean(1).cpu().numpy(); c2 = (sl == 2).float().mean(1).cpu().numpy(); c3 = (sl == 3).float().mean(1).cpu().numpy()
        tv = c1 + c2 + c3 + 1e-9; lab = np.zeros(K, int); tum = tf > 0.25; tau = 0.15
        et = tum & (c3 / tv >= tau); nc = tum & ~et & (c1 / tv >= tau); ed = tum & ~et & ~nc
        lab[ed] = 2; lab[nc] = 1; lab[et] = 3
        for s in SIZES:
            for m in MODS:
                pt = S.sample_patches_group(b[m].volume, S.mixed_bag_vox(b[m], c[None], torch.full((1, K), float(s), device=device), unit))[0, :, :, :, 0]
                store[f"{s}_{m}"].append(pt.half().cpu().numpy())
        COORD.append((c - cm).cpu().numpy()); LAB.append(lab); GRP.append(np.full(K, gi))
        used += 1; del b, segt; torch.cuda.empty_cache()
    arrs = {k: np.concatenate(v).astype(np.float16) for k, v in store.items()}
    np.savez(out, coords=np.concatenate(COORD).astype(np.float32), labels=np.concatenate(LAB).astype(np.int8),
             groups=np.concatenate(GRP).astype(np.int16), **arrs)
    print(f"cache built: {used} patients, {len(np.concatenate(LAB))} patches -> {out}", flush=True)


def eval_ckpt(cache, ckpt, *, sizes=(4, 8), device="cuda", mixed=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold
    Z = np.load(cache); coords = Z["coords"]; labels = Z["labels"].astype(int); groups = Z["groups"].astype(int)
    ck = torch.load(ckpt, map_location=device); step = int(ck.get("step", -1))
    E = M.Phase0Encoder(M.EncoderConfig(width=384, depth=12, heads=6, n_series=8)).to(device)
    E.load_state_dict(ck["model"]); E.eval()
    gids = sorted(set(groups.tolist()))

    def feats_for(s):
        cols = {m: [] for m in MODS}
        for g in gids:
            idx = np.where(groups == g)[0]; co = torch.as_tensor(coords[idx], device=device)[None].float()
            sz3 = S.size_to_extent(torch.full((1, len(idx)), float(s), device=device), 2)   # native axis=2 (as notebook)
            for mi, m in enumerate(MODS):
                pt = torch.as_tensor(Z[f"{s}_{m}"][idx], device=device).float()[None, ..., None]
                sid = torch.full((1, len(idx)), mi, device=device, dtype=torch.long) if mixed else None
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, lat = E.teacher_readout(pt, co, sz3, sid)
                cols[m].append(lat[0].float().cpu().numpy())
        return {m: np.concatenate(cols[m]) for m in MODS}

    keep = labels != 1; Y = labels[keep]; G = groups[keep]

    def ev(X):
        yt, yp = [], []
        for tr, te in GroupKFold(5).split(X, Y, G):
            clf = LogisticRegression(max_iter=1200, class_weight="balanced").fit(X[tr], Y[tr])
            yt.append(Y[te]); yp.append(clf.predict(X[te]))
        yt = np.concatenate(yt); yp = np.concatenate(yp)
        per = f1_score(yt, yp, labels=[0, 2, 3], average=None, zero_division=0)
        return float(per.mean()), float(per[2])                  # macro-F1, enh(ET)-F1

    out = [f"ckpt step {step} | patches {len(labels)} | ET+ {int((labels==3).sum())} edema+ {int((labels==2).sum())}"]
    for s in sizes:
        F = feats_for(s); X4 = np.concatenate([F[m] for m in MODS], -1)[keep]; Xt = F["t1c"][keep]
        m4, e4 = ev(X4); mt, et = ev(Xt)
        out.append(f"  s={s}mm: t1c enh {et:.3f} macro {mt:.3f} | 4mod enh {e4:.3f} macro {m4:.3f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--cache", default="/tmp/patchcache.npz")
    ap.add_argument("--data-root", default="/tmp/heldout")
    ap.add_argument("--tracks", nargs="+", default=["mets_ho"])
    ap.add_argument("--build", action="store_true", help="(re)build the patch cache before eval")
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mixed", action="store_true",
                    help="mixed-modality checkpoint: pass per-patch series_ids to teacher_readout (Site A "
                         "conditioning) so the readout matches training. Off for legacy phased checkpoints.")
    a = ap.parse_args()
    if a.build or not os.path.exists(a.cache):
        build_cache(a.data_root, a.tracks, a.cache, device=a.device)
    if a.checkpoint:
        print(eval_ckpt(a.cache, a.checkpoint, sizes=tuple(a.sizes), device=a.device, mixed=a.mixed))


if __name__ == "__main__":
    main()
