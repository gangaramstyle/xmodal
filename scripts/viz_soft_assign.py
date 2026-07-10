#!/usr/bin/env python
"""Visualize the similarity-softened matching target: which patches it treats as INTERCHANGEABLE
(share positive mass -> confusing them isn't penalized) vs DIFFERENT.

For a prism of target-modality patches, compute colors = normalize(color_head(blur(patches))) and the
soft-assignment matrix soft = softmax(colors @ colors.T / tau) -- exactly the target slot_match_loss uses
with --soft-match-tau. Then render, for a few example query patches (a distinctive one, an ambiguous one,
and the central/tumor one), the grid tinted GREEN by how much soft mass each other patch receives
(bright = interchangeable), with the query boxed yellow and "considered same" patches (soft>thr) boxed
green. Also a soft-matrix heatmap. Per-patch ambiguity = effective #interchangeable = 1/sum_j soft[q,j]^2.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="/tmp/heldout")
    ap.add_argument("--tracks", nargs="+", default=["mets_ho"])
    ap.add_argument("--tgt", default="t1c")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--patch-mm", type=float, default=8.0, help="physical patch size (mm); try 4 or 2 for finer")
    ap.add_argument("--content-blur", type=int, default=3)
    ap.add_argument("--out", default="/tmp/soft_assign.png")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    ck = torch.load(a.checkpoint, map_location=dev)
    E = M.Phase0Encoder(M.EncoderConfig(width=384, depth=12, heads=6, n_series=8)).to(dev)
    E.load_state_dict(ck["model"]); E.eval(); step = int(ck.get("step", -1))

    # focal enhancing met -> center a grid on it (so we get a tumor patch + surrounding normal tissue)
    dirs = []
    for tr in a.tracks:
        dirs += sorted(glob.glob(os.path.join(os.path.expanduser(a.data_root), tr, "BraTS-*")))
    best = None
    for d in dirs:
        pid = os.path.basename(d); sp = f"{d}/{pid}-seg.nii.gz"
        if not os.path.exists(sp):
            continue
        seg = np.asarray(nib.load(sp).get_fdata()).astype(int); lid, nl = ndi.label(seg == 3)
        for c in range(1, nl + 1):
            szc = int((lid == c).sum())
            if szc < 200:
                continue
            if best is None or abs(szc - 2000) < best[0]:
                best = (abs(szc - 2000), d, lid == c)
    _, bd, mask = best; pid = os.path.basename(bd)
    b = D.load_local_bundle(pid, bd, device=dev)[0]; sc = b[a.tgt]
    ax = int(np.argmax(np.abs(np.array(sc.volume.shape, float) - np.median(sc.volume.shape))))
    inpl = [x for x in range(3) if x != ax]
    ctrd = np.argwhere(mask).mean(0).astype(np.float32)
    anchor_mm = (np.linalg.inv(sc.affine_inv.cpu().numpy()) @ ctrd) + sc.affine_trans.cpu().numpy()

    G, pm, p = 6, a.patch_mm, 16; spacing = pm * 15 / 16; N = G * G
    lin = (np.arange(G) - (G - 1) / 2) * spacing; gi, gj = np.meshgrid(lin, lin, indexing="ij")
    baseoff = np.zeros((N, 3), np.float32); baseoff[:, inpl[0]] = gi.ravel(); baseoff[:, inpl[1]] = gj.ravel()
    unit = S.slab_unit_offsets(ax, 16, dev); sz = torch.full((1, N), pm, device=dev)
    ctr = torch.as_tensor(anchor_mm + baseoff, device=dev, dtype=torch.float32)
    tgt = S.sample_patches_group(b[a.tgt].volume, S.mixed_bag_vox(b[a.tgt], ctr[None], sz, unit))  # [1,N,16,16,1]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        colors = torch.nn.functional.normalize(E.color_head(blur_contents(tgt, a.content_blur)), dim=-1)  # [1,N,W]
    sim = (colors[0] @ colors[0].T).float().cpu().numpy()                      # content-content cosine [N,N]
    soft = np.exp(sim / a.tau); soft = soft / soft.sum(1, keepdims=True)        # softmax rows = soft targets
    amb = 1.0 / (soft ** 2).sum(1)                                              # effective # interchangeable per patch
    tnp = tgt[0, :, :, :, 0].float().cpu().numpy()

    # queries to illustrate: most-distinctive (min amb), most-ambiguous (max amb), central (tumor-ish)
    queries = {"distinctive (amb %.1f)" % amb[int(amb.argmin())]: int(amb.argmin()),
               "ambiguous (amb %.1f)" % amb[int(amb.argmax())]: int(amb.argmax()),
               "central/met (amb %.1f)" % amb[N // 2 + G // 2]: N // 2 + G // 2}

    CELL = 40; thr = None
    def grid_img(tint_q=None):
        img = Image.new("RGB", (G * CELL, G * CELL), (10, 10, 10)); dr = ImageDraw.Draw(img)
        w = soft[tint_q] / soft[tint_q].max() if tint_q is not None else None
        for n in range(N):
            i, j = n // G, n % G
            patch = np.array(Image.fromarray(u8(tnp[n])).resize((CELL, CELL), Image.NEAREST))
            rgb = np.stack([patch] * 3, -1).astype(np.float32)
            if tint_q is not None:
                rgb[..., 1] = np.clip(rgb[..., 1] + 200 * w[n], 0, 255)             # green glow ~ soft mass
            img.paste(Image.fromarray(rgb.astype(np.uint8)), (j * CELL, i * CELL))
            if tint_q is not None:
                if n == tint_q:
                    dr.rectangle([j * CELL, i * CELL, (j + 1) * CELL - 1, (i + 1) * CELL - 1], outline=(255, 230, 40), width=3)
                elif w[n] > 0.35:                                                    # "considered the same"
                    dr.rectangle([j * CELL, i * CELL, (j + 1) * CELL - 1, (i + 1) * CELL - 1], outline=(40, 220, 60), width=2)
        return img

    panels = [("all patches (target t1c)", grid_img(None))]
    for lbl, q in queries.items():
        panels.append((lbl, grid_img(q)))
    # soft-matrix heatmap (green), upscaled
    hm = (soft / soft.max(1, keepdims=True) * 255).astype(np.uint8)
    hmimg = Image.fromarray(np.stack([np.zeros_like(hm), hm, np.zeros_like(hm)], -1)).resize((G * CELL, G * CELL), Image.NEAREST)
    panels.append(("soft matrix (row=query)", hmimg))

    LM, TM, GAP = 8, 20, 12; PW = G * CELL
    W = LM + len(panels) * (PW + GAP); H = TM + PW + 24
    canvas = Image.new("RGB", (W, H), (18, 18, 18)); d = ImageDraw.Draw(canvas)
    d.text((LM, 4), f"SOFT-ASSIGNMENT tau={a.tau}  patch={pm:g}mm  ckpt step {step}  {pid}  amb {amb.min():.1f}-{amb.max():.1f}/{N}  (yellow=query, green=interchangeable)", fill=(220, 220, 220))
    for idx, (lbl, im) in enumerate(panels):
        x = LM + idx * (PW + GAP); canvas.paste(im, (x, TM)); d.text((x, TM + PW + 4), lbl, fill=(210, 210, 210))
    canvas.save(a.out)
    print(f"WROTE {a.out} | {pid} | tau {a.tau} | ambiguity range {amb.min():.1f}-{amb.max():.1f} (of {N}) | ckpt step {step}")


if __name__ == "__main__":
    main()
