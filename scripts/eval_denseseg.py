#!/usr/bin/env python
"""Dense-segmentation eval, tumor-prism-scoped (ORACLE detection). Given a tumor-containing prism, densely
query the fine-tuned seg decoder at every grid voxel -> per-voxel class -> DSC + NSD for ET/TC/WT vs GT.
Because we hand it the tumor prism, this isolates segmentation QUALITY from detection -> comparable to the
BraTS-METS lesion-wise DSC (~0.75 ET at SOTA). Encoder frozen; decoder fine-tuned on dense voxels; GroupKFold-5.
Regions: ET={3}, TC={1,3}(NCR+ET), WT={1,2,3}. Query at --res mm (default 2mm) inside a --prism-mm box.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from xmodal import data as D, model as M, sampling as S  # noqa: E402

MODS = ["t1", "t1c", "t2", "flair"]
REGIONS = {"ET": {3}, "TC": {1, 3}, "WT": {1, 2, 3}}


def seg_at(scan, pts_mm, dev):
    """nearest seg class at world-mm points [K,3]."""
    vox = ((pts_mm - scan.affine_trans) @ scan.affine_inv.T).round().long()
    shp = torch.as_tensor(scan.seg_vol.shape, device=dev)
    vox = vox.clamp(min=torch.zeros(3, device=dev, dtype=torch.long), max=shp - 1)
    return scan.seg_vol[vox[..., 0], vox[..., 1], vox[..., 2]].long()          # [K]


def dsc(pred, gt):
    inter = (pred & gt).sum().item(); s = pred.sum().item() + gt.sum().item()
    return None if s == 0 else 2.0 * inter / s                                  # None = region absent in both (no penalty)


def nsd(pred, gt, coords, tol):
    """Normalized surface distance @ tol mm (brute force over boundary points; prism-sized so cheap)."""
    if pred.sum() == 0 or gt.sum() == 0:
        return None
    # boundary = points of the mask with a neighbor (within ~res) of the opposite label
    def surf(mask):
        c = coords[mask]
        if len(c) == 0:
            return c
        return c                                                                # prism is small -> use all mask pts as surface proxy
    pc, gc = coords[pred], coords[gt]
    if len(pc) == 0 or len(gc) == 0:
        return None
    d_pg = torch.cdist(pc, gc).min(1).values; d_gp = torch.cdist(gc, pc).min(1).values
    return float(((d_pg <= tol).float().sum() + (d_gp <= tol).float().sum()) / (len(pc) + len(gc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data-root", default="/tmp/ho"); ap.add_argument("--tracks", nargs="+", default=["mets_ho"])
    ap.add_argument("--n-patients", type=int, default=40); ap.add_argument("--n-prisms", type=int, default=8)
    ap.add_argument("--n-src", type=int, default=96); ap.add_argument("--voxels", type=int, default=8)
    ap.add_argument("--size", type=float, default=4.0, help="source patch mm"); ap.add_argument("--prism-mm", type=float, default=32.0)
    ap.add_argument("--res", type=float, default=2.0, help="dense query grid STRIDE mm (metric resolution)")
    ap.add_argument("--qsize", type=float, default=4.0, help="query SIZE embedding mm (keep in-distribution w/ pretrain patch size, independent of stride)")
    ap.add_argument("--tol", type=float, default=2.0, help="NSD tolerance mm")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tgt-train", type=int, default=384, help="voxels/prism for fine-tune"); ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--unfreeze", type=int, default=12); ap.add_argument("--enc-width", type=int, default=384); ap.add_argument("--enc-heads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device; V = a.voxels; unit = S._cube_unit(V, dev); rng = np.random.default_rng(a.seed)

    dirs = []
    for tr in a.tracks:
        dirs += sorted(glob.glob(os.path.join(os.path.expanduser(a.data_root), tr, "BraTS-*")))
    prisms = []; pid_ctr = 0
    for d in dirs:
        if pid_ctr >= a.n_patients:
            break
        pid = os.path.basename(d); segp = glob.glob(f"{d}/{pid}-seg.nii.gz")
        if not segp:
            continue
        try:
            b = D.load_local_bundle(pid, d, device=dev, with_seg=True)[0]
        except Exception:
            continue
        sc0 = b["t1c"]; tmm = sc0.tumor_np
        if tmm is None or len(tmm) == 0:
            continue
        fgn = sc0.foreground_np if sc0.foreground_np is not None else sc0.foreground_mm.cpu().numpy()
        half = a.prism_mm / 2.0
        lin = np.arange(-half, half + 1e-3, a.res, dtype=np.float32)             # dense grid offsets (mm)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij"); grid = np.stack([gx, gy, gz], -1).reshape(-1, 3)
        for _ in range(a.n_prisms):
            anch = tmm[rng.integers(len(tmm))].astype(np.float32)                # anchor ON a tumor voxel
            gpts = (grid + anch).astype(np.float32)                             # dense query voxels (world-mm)
            cs = fgn[rng.integers(len(fgn), size=a.n_src)].astype(np.float32)   # source context near the prism
            keep = np.abs(cs - anch).max(-1) <= half
            if keep.sum() < a.n_src // 2:
                continue
            cm = fgn[np.abs(fgn - anch).max(-1) <= half]
            if len(cm) < 8:
                continue
            cs = cm[rng.integers(len(cm), size=a.n_src)].astype(np.float32)
            sm = rng.integers(4, size=a.n_src)
            sp = np.zeros((a.n_src, V, V, V), np.float16)
            for mi, m in enumerate(MODS):
                sel = np.nonzero(sm == mi)[0]
                if sel.size:
                    sp[sel] = S._gather_cubes(b[m], torch.as_tensor(cs[sel], device=dev),
                                              torch.full((sel.size,), a.size, device=dev), unit, dev).half().cpu().numpy()
            gt = seg_at(sc0, torch.as_tensor(gpts, device=dev), dev).cpu().numpy()   # class per query voxel
            prisms.append(dict(sp=sp, sc=cs - anch, sm=sm, gpts=gpts - anch, gt=gt, pid=pid_ctr))
        pid_ctr += 1; del b; torch.cuda.empty_cache()
    print(f"built {len(prisms)} tumor prisms over {pid_ctr} patients (res={a.res}mm grid ~{len(grid)} vox/prism)", flush=True)
    assert prisms

    def slots(E, seg_tok, sp, sc, sm, qpts):                                    # frozen-enc source -> decode dense queries
        nb = sp.shape[0]; nreg = 2 + E.registers.shape[0]
        zs = torch.full((nb, sp.shape[1], 3), a.size, device=dev); zt = torch.full((nb, qpts.shape[1], 3), a.qsize, device=dev)
        with torch.no_grad():
            x = E.encode(*E._context([E.embed(sp, zs, sm)], [sc], dev, nb)); ctx, cc = E._context([x[:, nreg:]], [sc], dev, nb)
        q = (E.query_seed[None, None, :] + E._size_emb(zt) + seg_tok[None, None, :]).contiguous()
        return E._decode(q, ctx, cc, qpts)

    def run(E):
        pids = sorted(set(p["pid"] for p in prisms)); fold = {p: i % 5 for i, p in enumerate(pids)}
        agg = {r: [] for r in REGIONS}; aggn = {r: [] for r in REGIONS}
        for f in range(5):
            tr = [p for p in prisms if fold[p["pid"]] != f]; te = [p for p in prisms if fold[p["pid"]] == f]
            seg_tok = torch.zeros(E.cfg.width, device=dev, requires_grad=True)
            head = torch.nn.Linear(E.cfg.width, 4).to(dev)
            params = list(head.parameters()) + [seg_tok, E.query_seed]
            for blk in list(E.decoder)[-a.unfreeze:]:
                for p in blk.parameters():
                    p.requires_grad_(True); params.append(p)
            opt = torch.optim.Adam(params, lr=a.lr)
            for _ in range(a.epochs):                                          # fine-tune on a subsample of prism voxels
                for pr in [tr[i] for i in rng.permutation(len(tr))]:
                    sub = rng.integers(len(pr["gt"]), size=a.tgt_train)
                    sp = torch.as_tensor(pr["sp"][None], device=dev).float(); sc = torch.as_tensor(pr["sc"][None], device=dev).float()
                    sm = torch.as_tensor(pr["sm"][None], device=dev).long(); qp = torch.as_tensor(pr["gpts"][sub][None], device=dev).float()
                    y = torch.as_tensor(pr["gt"][sub], device=dev).long()
                    cnt = torch.bincount(y, minlength=4).float(); w = len(y) / (4 * cnt.clamp(min=1))
                    logit = head(slots(E, seg_tok, sp, sc, sm, qp)).reshape(-1, 4)
                    opt.zero_grad(); torch.nn.functional.cross_entropy(logit, y, weight=w).backward(); opt.step()
            with torch.no_grad():                                             # dense-query every voxel of test prisms
                for pr in te:
                    sp = torch.as_tensor(pr["sp"][None], device=dev).float(); sc = torch.as_tensor(pr["sc"][None], device=dev).float()
                    sm = torch.as_tensor(pr["sm"][None], device=dev).long(); coords = torch.as_tensor(pr["gpts"], device=dev).float()
                    gtc = torch.as_tensor(pr["gt"], device=dev).long(); preds = []
                    for c0 in range(0, coords.shape[0], 4096):
                        qp = coords[c0:c0 + 4096][None]
                        preds.append(head(slots(E, seg_tok, sp, sc, sm, qp)).reshape(-1, 4).argmax(1))
                    pc = torch.cat(preds)
                    for r, cls in REGIONS.items():
                        gm = torch.zeros_like(gtc, dtype=torch.bool); pm = torch.zeros_like(pc, dtype=torch.bool)
                        for k in cls:
                            gm |= (gtc == k); pm |= (pc == k)
                        dv = dsc(pm, gm)
                        if dv is not None:
                            agg[r].append(dv); nv = nsd(pm, gm, coords, a.tol)
                            if nv is not None:
                                aggn[r].append(nv)
        return {r: (float(np.mean(agg[r])) if agg[r] else 0, float(np.mean(aggn[r])) if aggn[r] else 0, len(agg[r])) for r in REGIONS}

    for spec in a.checkpoints:
        nm, path = spec.split("=", 1); g = sorted(glob.glob(path))
        if not g:
            print(f"{nm} NO_CKPT {path}", flush=True); continue
        ck = torch.load(g[-1], map_location=dev); step = int(ck.get("step", -1))
        E = M.Phase0Encoder(M.EncoderConfig(width=a.enc_width, depth=12, heads=a.enc_heads, n_series=8,
                                            patch_grid=(V, V, V))).to(dev)
        E.load_state_dict(ck["model"], strict=False)
        for p in E.parameters():
            p.requires_grad_(False)
        r = run(E)
        msg = " | ".join(f"{k} DSC {v[0]:.3f} NSD {v[1]:.3f}(n{v[2]})" for k, v in r.items())
        print(f"{nm} step{step} denseseg qsize{a.qsize}mm/stride{a.res}mm: {msg}", flush=True)


if __name__ == "__main__":
    main()
