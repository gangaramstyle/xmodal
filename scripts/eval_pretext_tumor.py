#!/usr/bin/env python
"""Pretext-on-tumor: does tumor-focused training make the v5 PRETEXT task (cross-modal ordering match_acc
+ pixel MAE) better ON TUMOR-CONTAINING grids? For each checkpoint, sample TUMOR bags (tumor_frac=1 ->
prism straddles segmented tumor) vs NON-TUMOR bags (tumor_frac=0 -> random foreground), run forward_v5,
and report the match_acc / MAE split. Compare tumor-trained (both_tumor) vs not (both): if the extra
tumor training helps, its tumor-bag match_acc should rise (and MAE fall) relative to the non-tumor model.

Uses the model's OWN content_blur (read from the checkpoint cfg) so match_acc is on the trained task.
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
    ap.add_argument("--data-root", default="/tmp/ho")
    ap.add_argument("--tracks", nargs="+", default=["mets_ho"])
    ap.add_argument("--n-patients", type=int, default=20, help="held-out patients (w/ seg) to load for bag sampling")
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--n-src", type=int, default=90)
    ap.add_argument("--n-anchor", type=int, default=6)
    ap.add_argument("--n-tgt", type=int, default=32)
    ap.add_argument("--voxels", type=int, default=8)
    ap.add_argument("--prisms", type=float, nargs="+", default=[32., 64.])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    # load held-out bundles WITH seg (so tumor_frac=1 can anchor on segmented tumor)
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
    print(f"loaded {len(bundles)} held-out bundles (w/ tumor seg)", flush=True)
    assert bundles, "no held-out bundles with tumor seg found"

    def probe(E, cb):
        out = {}
        for regime, tf in (("tumor", 1.0), ("nontumor", 0.0)):
            accs, maes, chn = [], [], []
            rng = np.random.default_rng(a.seed)
            for _ in range(a.batches):
                bt = S.sample_v5_batch(bundles, batch_size=a.batch_size, n_src=a.n_src, n_anchor=a.n_anchor,
                                       n_tgt=a.n_tgt, prism_choices=tuple(a.prisms), voxels=a.voxels,
                                       rng=rng, device=dev, tumor_frac=tf)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    o = E.forward_v5(bt, content_blur=cb, mae_weight=0.25, match_weight=1.0)
                accs.append(float(o["match_acc"])); maes.append(float(o["mae"])); chn.append(float(o["chance"]))
            out[regime] = (float(np.mean(accs)), float(np.mean(maes)), float(np.mean(chn)))
        return out

    for spec in a.checkpoints:
        nm, path = spec.split("=", 1)
        g = sorted(glob.glob(path))
        if not g:
            print(f"{nm} NO_CKPT {path}", flush=True); continue
        ck = torch.load(g[-1], map_location=dev); step = int(ck.get("step", -1))
        cb = int(ck.get("cfg", {}).get("content_blur", 1))                 # eval on the model's OWN trained task
        E = M.Phase0Encoder(M.EncoderConfig(width=384, depth=12, heads=6, n_series=8,
                                            patch_grid=(a.voxels,) * 3)).to(dev)
        E.load_state_dict(ck["model"], strict=False); E.eval()
        r = probe(E, cb)
        (ta, tm, tc), (na, nm2, nc) = r["tumor"], r["nontumor"]
        print(f"{nm} step{step} blur{cb} | TUMOR macc {ta:.3f} mae {tm:.3f} | NONTUMOR macc {na:.3f} mae {nm2:.3f} "
              f"| dmacc {ta - na:+.3f} (chance {tc:.3f})", flush=True)


if __name__ == "__main__":
    main()
