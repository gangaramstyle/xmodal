#!/usr/bin/env python
"""Grid of soft-assignment examples for eyeballing: N_PRISMS different mets (different patients) x N_QUERIES
query patches each. Each cell is the anatomical locator (dense 2mm tiling) with one yellow query and
continuous GREEN ~ soft-target interchangeability. Queries per prism are chosen to span the ambiguity
range (sharpest -> most ambiguous). Row 0 of each prism = plain anatomy (all patch positions).
"""
import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xmodal import data as D, model as M, sampling as S  # noqa: E402
from xmodal.matching import blur_contents  # noqa: E402


def u8(a):
    return (np.clip(np.asarray(a, np.float32), 0, 1) * 255).astype(np.uint8)


def find_mets(dirs, n_prisms):
    """One focal enhancing met per patient, first n_prisms distinct patients that have one."""
    out = []
    for d in dirs:
        pid = os.path.basename(d); sp = f"{d}/{pid}-seg.nii.gz"
        if not os.path.exists(sp):
            continue
        seg = np.asarray(nib.load(sp).get_fdata()).astype(int); lid, nl = ndi.label(seg == 3)
        best = None
        for c in range(1, nl + 1):
            szc = int((lid == c).sum())
            if szc < 300:
                continue
            if best is None or abs(szc - 2500) < best[0]:
                best = (abs(szc - 2500), lid == c)
        if best is not None:
            out.append((d, pid, best[1]))
        if len(out) >= n_prisms:
            break
    return out


def prism_soft(E, b, tgt, mask, G, pm, blur, tau, dev, soft_sim="model"):
    sc = b[tgt]
    ax = int(np.argmax(np.abs(np.array(sc.volume.shape, float) - np.median(sc.volume.shape))))
    inpl = [x for x in range(3) if x != ax]
    ctrd = np.argwhere(mask).mean(0).astype(np.float32)
    anchor_mm = (np.linalg.inv(sc.affine_inv.cpu().numpy()) @ ctrd) + sc.affine_trans.cpu().numpy()
    av = (sc.affine_inv.cpu().numpy() @ (anchor_mm - sc.affine_trans.cpu().numpy()))
    spacing = pm * 15 / 16; N = G * G
    lin = (np.arange(G) - (G - 1) / 2) * spacing; gi, gj = np.meshgrid(lin, lin, indexing="ij")
    baseoff = np.zeros((N, 3), np.float32); baseoff[:, inpl[0]] = gi.ravel(); baseoff[:, inpl[1]] = gj.ravel()
    unit = S.slab_unit_offsets(ax, 16, dev); sz = torch.full((1, N), pm, device=dev)
    ctr = torch.as_tensor(anchor_mm + baseoff, device=dev, dtype=torch.float32)
    tgtp = S.sample_patches_group(b[tgt].volume, S.mixed_bag_vox(b[tgt], ctr[None], sz, unit))
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if soft_sim == "pixel":                                            # FIXED raw blurred pixels, RBF (intensity-aware)
            feat = blur_contents(tgtp, blur).reshape(N, -1).float()
            d2 = torch.cdist(feat[None], feat[None])[0] ** 2               # squared Euclidean [N,N]; respects brightness
            sim = (-d2 / (d2[d2 > 0].median() + 1e-9)).float().cpu().numpy()  # neg normalized dist; self=0=max
        else:                                                              # trainable color_head (model-based), cosine
            feat = torch.nn.functional.normalize(E.color_head(blur_contents(tgtp, blur)), dim=-1)
            sim = (feat[0] @ feat[0].T).float().cpu().numpy()
    soft = np.exp(sim / tau); soft = soft / soft.sum(1, keepdims=True)
    amb = 1.0 / (soft ** 2).sum(1)
    return dict(ax=ax, inpl=inpl, av=av, baseoff=baseoff, spacing=spacing, N=N,
                vol=sc.volume.cpu().numpy(), soft=soft, amb=amb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="/tmp/heldout")
    ap.add_argument("--tracks", nargs="+", default=["mets_ho"])
    ap.add_argument("--tgt", default="t1c")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--patch-mm", type=float, default=2.0)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--content-blur", type=int, default=3)
    ap.add_argument("--soft-sim", choices=["model", "pixel"], default="model",
                    help="pixel = FIXED blurred-pixel similarity (non-circular); model = color_head")
    ap.add_argument("--n-prisms", type=int, default=5)
    ap.add_argument("--n-queries", type=int, default=10)
    ap.add_argument("--out", default="/tmp/soft_grid.png")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    ck = torch.load(a.checkpoint, map_location=dev)
    E = M.Phase0Encoder(M.EncoderConfig(width=384, depth=12, heads=6, n_series=8)).to(dev)
    E.load_state_dict(ck["model"]); E.eval(); step = int(ck.get("step", -1))

    dirs = []
    for tr in a.tracks:
        dirs += sorted(glob.glob(os.path.join(os.path.expanduser(a.data_root), tr, "BraTS-*")))
    mets = find_mets(dirs, a.n_prisms)

    CELL = 150; G = a.grid; pm = a.patch_mm
    ncol = 1 + a.n_queries
    rows = []
    row_labels = []
    for d, pid, mask in mets:
        b = D.load_local_bundle(pid, d, device=dev)[0]
        P = prism_soft(E, b, a.tgt, mask, G, pm, a.content_blur, a.tau, dev, soft_sim=a.soft_sim)
        av, inpl, ax, baseoff, soft, amb, vol = P["av"], P["inpl"], P["ax"], P["baseoff"], P["soft"], P["amb"], P["vol"]
        zc = int(round(av[ax])); sl = np.take(vol, int(np.clip(zc, 0, vol.shape[ax] - 1)), axis=ax); fimg = np.stack([u8(sl)] * 3, -1)
        half = G * P["spacing"] / 2 + 3 * pm
        y0 = max(0, int(av[inpl[0]] - half)); y1 = min(fimg.shape[0], int(av[inpl[0]] + half))
        x0 = max(0, int(av[inpl[1]] - half)); x1 = min(fimg.shape[1], int(av[inpl[1]] + half))
        sy = CELL / max(1, y1 - y0); sx = CELL / max(1, x1 - x0); r = max(2, pm / 2 * (sy + sx) / 2)
        crop = np.array(Image.fromarray(fimg[y0:y1, x0:x1]).resize((CELL, CELL), Image.NEAREST)).astype(np.float32)

        def pos(n):
            return (av[inpl[1]] + baseoff[n][inpl[1]] - x0) * sx, (av[inpl[0]] + baseoff[n][inpl[0]] - y0) * sy

        def cell(q=None):
            base = crop.copy()
            if q is not None:
                w = soft[q] / soft[q].max()
                for n in range(P["N"]):
                    if n == q:
                        continue
                    px, py = pos(n); base[max(0, int(py - r)):int(py + r), max(0, int(px - r)):int(px + r), 1] = \
                        np.clip(base[max(0, int(py - r)):int(py + r), max(0, int(px - r)):int(px + r), 1] + 235 * w[n], 0, 255)
            im = Image.fromarray(base.astype(np.uint8)); dr = ImageDraw.Draw(im)
            if q is not None:
                px, py = pos(q); dr.rectangle([px - r, py - r, px + r, py + r], outline=(255, 230, 40), width=2)
                dr.text((2, 2), f"amb{amb[q]:.0f}", fill=(255, 235, 120))
            return im

        order = np.argsort(amb)                                            # sharp -> ambiguous
        qs = [int(order[int(round(k))]) for k in np.linspace(0, P["N"] - 1, a.n_queries)]
        rows.append([cell(None)] + [cell(q) for q in qs]); row_labels.append(pid.replace("BraTS-MET-", ""))

    GAP = 3
    W = ncol * CELL + (ncol + 1) * GAP; H = 16 + len(rows) * (CELL + GAP) + GAP
    canvas = Image.new("RGB", (W, H), (18, 18, 18)); d = ImageDraw.Draw(canvas)
    d.text((6, 3), f"SOFT-ASSIGN grid  tau={a.tau} patch={pm:g}mm dense{G}x{G}  step {step}  col0=anatomy, then {a.n_queries} queries (yellow), green=interchangeable", fill=(220, 220, 220))
    for ri, row in enumerate(rows):
        ry = 16 + ri * (CELL + GAP) + GAP
        for ci, im in enumerate(row):
            canvas.paste(im, (GAP + ci * (CELL + GAP), ry))
        d.text((GAP + 2, ry + 2), row_labels[ri], fill=(150, 200, 255))
    canvas.save(a.out)
    print(f"WROTE {a.out} | {len(rows)} prisms x {a.n_queries} queries | tau {a.tau} patch {pm}mm | step {step}")


if __name__ == "__main__":
    main()
