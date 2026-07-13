#!/usr/bin/env python
"""Bag-size x spacing 2x2 on the PRETEXT task. The encoder was trained on 96-patch source bags (n_src 90
+ 6 anchors), randomly spaced within a prism. Here we run forward_v5 (match_acc + MAE) with the source bag
set to {96, 256} patches x {random, grid(FPS-even)} spacing, on tumor-enriched held-out prisms. Because the
readout IS the pretext task, it stays in-distribution -> any drop cleanly attributes to bag-size/spacing
(no probe confound). Tells us whether the oversized/grid readout is what degrades things (hyp: larger ->
worse). Run per checkpoint (latest per arm).
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from xmodal import data as D, model as M, sampling as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="name=globpath ... (latest per name)")
    ap.add_argument("--data-root", default="/tmp/ho"); ap.add_argument("--tracks", nargs="+", default=["mets_ho"])
    ap.add_argument("--n-patients", type=int, default=20)
    ap.add_argument("--batches", type=int, default=12); ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-src", type=int, nargs="+", default=[90, 250], help="source patches (enc sees +6 anchors)")
    ap.add_argument("--n-tgt", type=int, default=32); ap.add_argument("--voxels", type=int, default=8)
    ap.add_argument("--prisms", type=float, nargs="+", default=[64.], help="prism mm (single -> fixed scale)")
    ap.add_argument("--tumor-frac", type=float, default=0.8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    dirs = []
    for tr in a.tracks:
        dirs += sorted(glob.glob(os.path.join(os.path.expanduser(a.data_root), tr, "BraTS-*")))
    bundles = []
    for d in dirs:
        if len(bundles) >= a.n_patients:
            break
        pid = os.path.basename(d)
        if not glob.glob(f"{d}/{pid}-seg.nii.gz"):
            continue
        try:
            b = D.load_local_bundle(pid, d, device=dev, with_seg=True)[0]
        except Exception:
            continue
        if all(sc.tumor_np is not None and len(sc.tumor_np) > 0 for sc in b.values()):
            bundles.append(b)
    print(f"loaded {len(bundles)} held-out bundles (tumor-enriched prisms, tumor_frac={a.tumor_frac})", flush=True)
    assert bundles

    def cell(E, cb, n_src, spacing):
        accs, maes = [], []; rng = np.random.default_rng(a.seed)
        for _ in range(a.batches):
            bt = S.sample_v5_batch(bundles, batch_size=a.batch_size, n_src=n_src, n_anchor=6, n_tgt=a.n_tgt,
                                   prism_choices=tuple(a.prisms), voxels=a.voxels, rng=rng, device=dev,
                                   tumor_frac=a.tumor_frac, src_spacing=spacing)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                o = E.forward_v5(bt, content_blur=cb, mae_weight=0.25, match_weight=1.0)
            accs.append(float(o["match_acc"])); maes.append(float(o["mae"]))
        return float(np.mean(accs)), float(np.mean(maes))

    for spec in a.checkpoints:
        nm, path = spec.split("=", 1); g = sorted(glob.glob(path))
        if not g:
            print(f"{nm} NO_CKPT {path}", flush=True); continue
        ck = torch.load(g[-1], map_location=dev); step = int(ck.get("step", -1))
        cb = int(ck.get("cfg", {}).get("content_blur", 1))
        E = M.Phase0Encoder(M.EncoderConfig(width=384, depth=12, heads=6, n_series=8,
                                            patch_grid=(a.voxels,) * 3)).to(dev)
        E.load_state_dict(ck["model"], strict=False); E.eval()
        print(f"\n{nm} step{step} blur{cb}  (trained on 96-patch random source)", flush=True)
        print(f"  {'enc_patches':>12} | {'random macc':>12} {'grid macc':>12} | {'random mae':>11} {'grid mae':>10}", flush=True)
        for ns in a.n_src:
            ra, rm = cell(E, cb, ns, "random"); ga, gm = cell(E, cb, ns, "grid")
            print(f"  {ns + 6:>12} | {ra:>12.3f} {ga:>12.3f} | {rm:>11.3f} {gm:>10.3f}", flush=True)


if __name__ == "__main__":
    main()
