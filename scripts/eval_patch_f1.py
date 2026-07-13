#!/usr/bin/env python
"""Held-out patch-level ENCODER-F1 (the trusted molab readout, ported faithfully from the notebook
_build_cache / _fast_eval cells). Builds a patch cache ONCE (K patches/patient x {4,8,16}mm x 4
modalities + 4mm-footprint labels + patient groups), then per checkpoint runs teacher_readout per
modality, concats the 4 modalities, and does LogisticRegression GroupKFold-5 -> per-class F1.

Labels (4mm footprint, priority ET>NCR>ED): tumor if >25% seg; then ET if ET/tumor>=0.15, else NCR
if NCR/tumor>=0.15, else edema. Necrosis (class 1) dropped at eval. enh-F1 = ET (class 3); macro-F1 =
mean over {non-tumor, edema, ET}. --cube V evaluates v5 3D-cube encoders (V^3 patches) instead of the
legacy 2.5D slabs. Probe is torch-native (no sklearn) so it runs on the CUBIC venv.
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


# --- sklearn-free probe (CUBIC has no sklearn; torch reproduces it exactly) --------------------------
def _f1_per_class(y_true, y_pred, labels):
    """Per-class F1 for the given labels (matches sklearn f1_score(average=None, zero_division=0))."""
    out = []
    for c in labels:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        den = 2 * tp + fp + fn
        out.append(0.0 if den == 0 else 2.0 * tp / den)
    return np.asarray(out, float)


def _group_kfold(groups, n_splits):
    """GroupKFold: no group spans train+test. Reproduces sklearn's greedy assignment (largest groups
    to the currently-lightest fold) so folds match the sklearn baseline exactly."""
    _, g = np.unique(groups, return_inverse=True)                # encode groups -> 0..G-1
    n_per_group = np.bincount(g)
    order = np.argsort(n_per_group)[::-1]                         # groups largest-first
    n_per_fold = np.zeros(n_splits, np.int64); grp_fold = np.empty(len(n_per_group), np.int64)
    for gi in order:
        f = int(np.argmin(n_per_fold)); grp_fold[gi] = f; n_per_fold[f] += n_per_group[gi]
    fold_of = grp_fold[g]
    for f in range(n_splits):
        te = np.where(fold_of == f)[0]; tr = np.where(fold_of != f)[0]
        yield tr, te


class _TorchLogReg:
    """Multinomial logistic regression = sklearn LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
    multi_class='multinomial', class_weight='balanced', fit_intercept=True). Convex objective ->
    torch.LBFGS converges to the SAME global optimum as sklearn's lbfgs (verified to match F1)."""
    def __init__(self, *, device="cuda", C=1.0, max_iter=500):
        self.device = device; self.C = float(C); self.max_iter = int(max_iter)

    def fit(self, X, y):
        classes = np.unique(y); self.classes_ = classes
        idx = {c: i for i, c in enumerate(classes)}
        yi = np.array([idx[v] for v in y]); nC = len(classes); nF = X.shape[1]
        counts = np.bincount(yi, minlength=nC).astype(float)
        cw = len(yi) / (nC * counts)                             # class_weight='balanced'
        dev = self.device
        Xt = torch.as_tensor(X, dtype=torch.float64, device=dev)
        yt = torch.as_tensor(yi, dtype=torch.long, device=dev)
        swt = torch.as_tensor(cw[yi], dtype=torch.float64, device=dev)
        W = torch.zeros(nC, nF, dtype=torch.float64, device=dev, requires_grad=True)
        b = torch.zeros(nC, dtype=torch.float64, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=self.max_iter, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-7, tolerance_change=1e-9)
        rows = torch.arange(len(yt), device=dev)

        def closure():
            opt.zero_grad()
            logp = torch.log_softmax(Xt @ W.T + b, dim=1)
            nll = -(swt * logp[rows, yt]).sum()
            loss = 0.5 * (W * W).sum() + self.C * nll            # L2 on W only (intercept unregularized)
            loss.backward(); return loss
        opt.step(closure)
        self.W = W.detach(); self.b = b.detach(); return self

    def predict(self, X):
        Xt = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        return self.classes_[(Xt @ self.W.T + self.b).argmax(1).cpu().numpy()]


def build_cache(data_root, tracks, out, *, K=700, LS=4, seed=0, device="cuda", cube=0,
                sampling="random", grid_step=8.0, grid_cap=6000,
                n_prisms=24, per_prism=96, prism_tumor=0.7, prism_mm=64.0):
    """Whole-brain modes (readout = one big per-PATIENT bag): sampling='random' (K sparse foreground
    centers, patient-centered coords) or 'grid' (dense lattice, full coverage). Prism modes (readout =
    per-PRISM bags ~per_prism patches, matching training's ~96-patch context; tumor-enriched for label
    coverage; prism-anchor-relative coords): sampling='prism_random' (sparse-in-prism, train-matched) or
    'prism_grid' (FPS-even-in-prism). cube=V -> 3D cubes (V^3) at v5 sizes; labels = 4mm seg footprint."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    prism_mode = sampling.startswith("prism")
    sizes = [4, 8] if cube else SIZES
    dirs = []
    for tr in tracks:
        dirs += sorted(glob.glob(os.path.join(os.path.expanduser(data_root), tr, "BraTS-*")))
    store = {f"{s}_{m}": [] for s in sizes for m in MODS}
    COORD, LAB, GRP, PID = [], [], [], []
    used = 0; prism_ctr = 0
    for gi, d in enumerate(dirs):
        pid = os.path.basename(d)
        segp = glob.glob(f"{d}/{pid}-seg.nii.gz")
        if not segp:
            continue
        try:
            b = D.load_local_bundle(pid, d, device=device, with_seg=prism_mode)[0]
        except Exception:
            continue
        sc0 = b["t1c"]
        if prism_mode:                                                           # per-prism bags (training-matched)
            spacing = "grid" if sampling.endswith("grid") else "random"
            fgn = sc0.foreground_np if sc0.foreground_np is not None else sc0.foreground_mm.cpu().numpy()
            tmm = sc0.tumor_np if sc0.tumor_np is not None else None
            half = prism_mm / 2.0; cs, cds, pids = [], [], []
            for _ in range(n_prisms):
                if tmm is not None and rng.random() < prism_tumor:
                    a = tmm[rng.integers(len(tmm))] + (rng.random(3) * 2 - 1) * half
                else:
                    a = fgn[rng.integers(len(fgn))]
                loc = fgn[np.abs(fgn - a).max(-1) <= half]
                if len(loc) < 8:
                    continue
                idx = S._fps_pick(loc, per_prism, rng) if spacing == "grid" else rng.integers(len(loc), size=per_prism)
                cc = loc[idx].astype(np.float32)
                cs.append(cc); cds.append(cc - a.astype(np.float32)); pids.append(np.full(per_prism, prism_ctr)); prism_ctr += 1
            if not cs:
                continue
            c = torch.as_tensor(np.concatenate(cs), device=device); coords = np.concatenate(cds).astype(np.float32)
            prism_ids = np.concatenate(pids)
        elif sampling == "grid":                                                 # dense grid over anatomy (full coverage)
            c = torch.unique(torch.round(sc0.foreground_mm / grid_step), dim=0) * grid_step
            if c.shape[0] > grid_cap:
                c = c[torch.randperm(c.shape[0], device=device)[:grid_cap]]
            coords = (c - c.mean(0)).cpu().numpy(); prism_ids = np.full(c.shape[0], gi)
        else:                                                                    # random foreground (whole-brain sparse)
            c = sc0.foreground_mm[torch.randint(sc0.foreground_mm.shape[0], (K,), device=device)]
            coords = (c - c.mean(0)).cpu().numpy(); prism_ids = np.full(c.shape[0], gi)
        K = c.shape[0]
        shp = torch.as_tensor(sc0.volume.shape, device=device)
        segt = torch.as_tensor(np.asarray(nib.load(segp[0]).get_fdata()), dtype=torch.int16, device=device)
        if cube:
            unit = S._cube_unit(cube, device)                                    # [V,V,V,3] unit cube offsets
            phys = unit[None] * float(LS) + c[:, None, None, None, :]            # [K,V,V,V,3] 4mm label footprint
            vi = ((phys - sc0.affine_trans) @ sc0.affine_inv.T).round().long()
            vi = vi.clamp(min=torch.zeros(3, device=device, dtype=torch.long), max=shp - 1)
            sl = segt[vi[..., 0], vi[..., 1], vi[..., 2]].reshape(K, -1)
        else:
            thick = naxis(sc0); unit = S.slab_unit_offsets(thick, 16, device)
            vi = S.mixed_bag_vox(sc0, c[None], torch.full((1, K), float(LS), device=device), unit).round().long()
            vi = vi.clamp(min=torch.zeros(3, device=device, dtype=torch.long), max=shp - 1)
            sl = segt[vi[..., 0], vi[..., 1], vi[..., 2]].reshape(K, -1)
        tf = (sl > 0).float().mean(1).cpu().numpy()
        c1 = (sl == 1).float().mean(1).cpu().numpy(); c2 = (sl == 2).float().mean(1).cpu().numpy(); c3 = (sl == 3).float().mean(1).cpu().numpy()
        tv = c1 + c2 + c3 + 1e-9; lab = np.zeros(K, int); tum = tf > 0.25; tau = 0.15
        et = tum & (c3 / tv >= tau); nc = tum & ~et & (c1 / tv >= tau); ed = tum & ~et & ~nc
        lab[ed] = 2; lab[nc] = 1; lab[et] = 3
        for s in sizes:
            for m in MODS:
                if cube:
                    pt = S._gather_cubes(b[m], c, torch.full((K,), float(s), device=device), unit, device)   # [K,V,V,V]
                else:
                    pt = S.sample_patches_group(b[m].volume, S.mixed_bag_vox(b[m], c[None], torch.full((1, K), float(s), device=device), unit))[0, :, :, :, 0]
                store[f"{s}_{m}"].append(pt.half().cpu().numpy())
        COORD.append(coords); LAB.append(lab); GRP.append(np.full(K, gi)); PID.append(prism_ids)
        used += 1; del b, segt; torch.cuda.empty_cache()
    arrs = {k: np.concatenate(v).astype(np.float16) for k, v in store.items()}
    np.savez(out, coords=np.concatenate(COORD).astype(np.float32), labels=np.concatenate(LAB).astype(np.int8),
             groups=np.concatenate(GRP).astype(np.int16), prisms=np.concatenate(PID).astype(np.int32),
             cube=np.int16(cube), sampling=np.str_(sampling), **arrs)
    lab_all = np.concatenate(LAB); npr = len(np.unique(np.concatenate(PID)))
    print(f"cache built: {used} patients, {len(lab_all)} patches ({len(lab_all)//max(used,1)}/patient), "
          f"{npr} bags ({len(lab_all)//max(npr,1)}/bag), cube={cube} sampling={sampling} -> {out}", flush=True)


def eval_ckpt(cache, ckpt, *, sizes=(4, 8), device="cuda", mixed=False, cube=0):
    Z = np.load(cache); coords = Z["coords"]; labels = Z["labels"].astype(int); groups = Z["groups"].astype(int)
    cube = int(cube or Z["cube"]) if "cube" in Z else cube                 # cache remembers its geometry
    mixed = mixed or bool(cube)                                            # v5 (cube) always uses series conditioning
    ck = torch.load(ckpt, map_location=device); step = int(ck.get("step", -1))
    ecfg = M.EncoderConfig(width=384, depth=12, heads=6, n_series=8, patch_grid=(cube, cube, cube) if cube else None)
    E = M.Phase0Encoder(ecfg).to(device)
    E.load_state_dict(ck["model"]); E.eval()
    gids = sorted(set(groups.tolist()))

    def feats_for(s):
        cols = {m: [] for m in MODS}
        for g in gids:
            idx = np.where(groups == g)[0]; co = torch.as_tensor(coords[idx], device=device)[None].float()
            if cube:
                sz3 = torch.full((1, len(idx), 3), float(s), device=device)                 # isotropic cube extent
            else:
                sz3 = S.size_to_extent(torch.full((1, len(idx)), float(s), device=device), 2)  # native axis=2 (notebook)
            for mi, m in enumerate(MODS):
                arr = torch.as_tensor(Z[f"{s}_{m}"][idx], device=device).float()
                pt = arr[None] if cube else arr[None, ..., None]                             # cube [1,n,V,V,V]; slab [1,n,V,V,1]
                sid = torch.full((1, len(idx)), mi, device=device, dtype=torch.long) if mixed else None
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, lat = E.teacher_readout(pt, co, sz3, sid)
                cols[m].append(lat[0].float().cpu().numpy())
        return {m: np.concatenate(cols[m]) for m in MODS}

    keep = labels != 1; Y = labels[keep]; G = groups[keep]

    def ev(X):
        yt, yp = [], []
        for tr, te in _group_kfold(G, 5):
            clf = _TorchLogReg(device=device).fit(X[tr], Y[tr])
            yt.append(Y[te]); yp.append(clf.predict(X[te]))
        yt = np.concatenate(yt); yp = np.concatenate(yp)
        per = _f1_per_class(yt, yp, [0, 2, 3])
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
    ap.add_argument("--cube", type=int, default=0,
                    help="v5 3D-cube encoders: cube voxel grid (e.g. 8). 0 = legacy 2.5D slabs.")
    ap.add_argument("--sampling", choices=["random", "grid", "prism_random", "prism_grid"], default="random",
                    help="random = train-like sparse foreground centers; grid = dense full-coverage lattice.")
    a = ap.parse_args()
    if a.build or not os.path.exists(a.cache):
        build_cache(a.data_root, a.tracks, a.cache, device=a.device, cube=a.cube, sampling=a.sampling)
    if a.checkpoint:
        print(eval_ckpt(a.cache, a.checkpoint, sizes=tuple(a.sizes), device=a.device, mixed=a.mixed, cube=a.cube))


if __name__ == "__main__":
    main()
