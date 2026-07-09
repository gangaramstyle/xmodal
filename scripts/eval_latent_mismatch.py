#!/usr/bin/env python
"""Latent-mismatch anomaly eval (phase-4 payoff metric).

For held-out patients, run the decoder's cross-modal prediction and score each recon patch by
  latent_mismatch = 1 - cos(pred_latent, frozen-teacher target latent)   [the latent ckpt's signal]
  pixel_err       = MSE(pred_pixels, true target pixels)                   [the pixel-cross baseline]
Then ask: does that per-patch anomaly score separate TUMOR from non-tumor patches (AUROC)? A patch
is tumor-positive if >25% of its voxels are tumor (seg>0); ET-positive if >25% are enhancing (seg==3).

Run on native@40k -> read the pixel_err AUROC (valid pixel-cross baseline; its latent_head is untrained).
Run on native_latent -> read the latent_mismatch AUROC (the trained latent signal). Both share the
same frozen encoder, so it's an apples-to-apples "does latent-space cross-prediction beat pixel-space".
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from xmodal import data as D, holdout as H, model as M, sampling as S
from xmodal.sampling import (draw_patch_sizes, draw_prism_half, mixed_bag_vox, resolve_thick_axis,
                             sample_patches_group, size_to_extent, slab_unit_offsets)

V = 16  # patch_voxels


def seg_fracs(t_scan, centers, sizes, unit_t, seg_t):
    """Per-patch tumor / ET voxel fraction from the target-modality patch footprint."""
    vox = mixed_bag_vox(t_scan, centers, sizes, unit_t)          # [G,n,V,V,1,3]
    idx = vox.round().long()
    Dz, Hy, Wx = seg_t.shape
    ix = idx[..., 0].clamp(0, Dz - 1); iy = idx[..., 1].clamp(0, Hy - 1); iz = idx[..., 2].clamp(0, Wx - 1)
    sp = seg_t[ix, iy, iz].reshape(idx.shape[0], idx.shape[1], -1).float()   # [G,n,vox]
    return (sp > 0).float().mean(-1), (sp == 3).float().mean(-1)             # tumor_frac, et_frac  [G,n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="~/xmodal/data/brats26")
    ap.add_argument("--tracks", nargs="+", default=["mets_train", "ped_train", "goat_gt", "goat_nogt"])
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--holdout-seed", type=int, default=0)
    ap.add_argument("--max-patients", type=int, default=60)
    ap.add_argument("--prisms-per-patient", type=int, default=8)
    ap.add_argument("--token-count", type=int, default=128)
    ap.add_argument("--patch-sizes", type=float, nargs="+", default=[8.0])
    ap.add_argument("--prism-choices", type=float, nargs="+", default=[32.0, 64.0, 128.0])
    ap.add_argument("--orient", default="native")
    ap.add_argument("--anchor-frac", type=float, default=0.05)
    ap.add_argument("--gate", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device
    root = os.path.expanduser(args.data_root)
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)

    pid2dir = {}
    for tr in args.tracks:
        d = os.path.join(root, tr)
        if os.path.isdir(d):
            pid2dir.update(D.find_brats_patients(d))
    all_pids = sorted(pid2dir)
    _, val_pids = H.split_patients(all_pids, seed=args.holdout_seed, val_frac=args.holdout_frac)
    print(f"held-out patients: {len(val_pids)} (using up to {args.max_patients})", flush=True)

    enc = M.Phase0Encoder(M.EncoderConfig(width=384, depth=12, heads=6, n_series=8)).to(dev)
    ck = torch.load(os.path.expanduser(args.checkpoint), map_location=dev)
    enc.load_state_dict(ck["model"]); enc.eval()
    print(f"loaded {args.checkpoint} (step {ck.get('step')})", flush=True)

    LAT, PXE, TUM, ET = [], [], [], []
    n = args.token_count; used = 0
    for pid in val_pids[:args.max_patients]:
        try:
            bundle, seg = D.load_local_bundle(pid, pid2dir[pid], device=dev, with_seg=True)
        except Exception as e:
            print(f"  skip {pid}: {e}", flush=True); continue
        if seg is None or len(bundle) < 2:
            continue
        seg_t = torch.as_tensor(seg, device=dev)
        mods = list(bundle)
        src_l, tgt_l, co_l, sz_l, tum_l, et_l = [], [], [], [], [], []
        for _ in range(args.prisms_per_patient):
            i, j = rng.choice(len(mods), size=2, replace=False)
            s, t = bundle[mods[i]], bundle[mods[j]]
            thick = resolve_thick_axis(s, args.orient, rng)
            unit_s = slab_unit_offsets(thick, V, dev); unit_t = slab_unit_offsets(thick, V, dev)
            half = draw_prism_half(rng, 1, tuple(args.prism_choices), dev)
            Mf = s.foreground_mm.shape[0]
            anchor = s.foreground_mm[torch.randint(Mf, (1,), device=dev)]
            centers = anchor[:, None] + (torch.rand(1, n, 3, device=dev) * 2 - 1) * half
            coords = centers - anchor[:, None]
            sizes = draw_patch_sizes(rng, 1, n, tuple(args.patch_sizes), dev, False)
            ps = sample_patches_group(s.volume, mixed_bag_vox(s, centers, sizes, unit_s))
            pt = sample_patches_group(t.volume, mixed_bag_vox(t, centers, sizes, unit_t))
            ext = size_to_extent(sizes, thick)
            tum, et = seg_fracs(t, centers, sizes, unit_t, seg_t)
            src_l.append(ps[0]); tgt_l.append(pt[0]); co_l.append(coords[0]); sz_l.append(ext[0])
            tum_l.append(tum[0]); et_l.append(et[0])
        source = torch.stack(src_l).float(); target = torch.stack(tgt_l).float()
        cds = torch.stack(co_l).float(); szs = torch.stack(sz_l).float()
        tum_b = torch.stack(tum_l); et_b = torch.stack(et_l)                     # [P,n]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            s_scls, _ = enc.teacher_readout(source, cds, szs)
            t_scls, t_lat = enc.teacher_readout(target, cds, szs)
            out = enc.cross_eval(source, target, cds, szs, s_scls.float(), t_scls.float(),
                                 t_lat.float(), anchor_frac=args.anchor_frac)
        rec = out["recon"]                                                        # [P,n_recon]
        LAT.append(out["latent_mismatch"].float().cpu().numpy().ravel())
        PXE.append(out["pixel_err"].float().cpu().numpy().ravel())
        TUM.append(torch.gather(tum_b, 1, rec).cpu().numpy().ravel())
        ET.append(torch.gather(et_b, 1, rec).cpu().numpy().ravel())
        used += 1
        if used % 10 == 0:
            print(f"  {used} patients done", flush=True)

    lat = np.concatenate(LAT); pxe = np.concatenate(PXE)
    tum = np.concatenate(TUM) > args.gate; et = np.concatenate(ET) > args.gate
    print(f"\npatients used: {used} | patches: {len(lat)} | tumor+: {tum.sum()} ({tum.mean():.3f}) | ET+: {et.sum()} ({et.mean():.3f})")

    def au(score, y):
        return roc_auc_score(y, score) if (y.sum() > 0 and y.sum() < len(y)) else float("nan")

    print("\n=== anomaly AUROC (higher score -> tumor) ===")
    print(f"                     tumor(any)   ET")
    print(f"latent_mismatch   |   {au(lat, tum):.4f}    {au(lat, et):.4f}")
    print(f"pixel_err (base)  |   {au(pxe, tum):.4f}    {au(pxe, et):.4f}")
    # ET vs strictly-normal (drop edema/necrosis-only patches) -> specificity for enhancement
    keep = et | ~tum
    if keep.sum() > 0:
        print("\n=== ET vs strictly-non-tumor (edema/necrosis-only patches removed) ===")
        print(f"latent_mismatch   |   {au(lat[keep], et[keep]):.4f}")
        print(f"pixel_err (base)  |   {au(pxe[keep], et[keep]):.4f}")


if __name__ == "__main__":
    main()
