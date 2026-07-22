"""Precompute the FROZEN provenance series-CLS latent per BraTS scan-modality -> {(patient, modality): np[D]} pkl.
This is the drop-in replacement for the one-hot series lookup used by run_v5.py --series-latent-cls (the ablation:
does a learned continuous series latent match the ground-truth one-hot conditioning?).

    python scripts/precompute_brats_cls.py --prov-ckpt runs/provm_full_paired_s0/step_090000.pt \
        --out ~/xmodal/brats_cls_vitb.pkl --tracks mets_train ped_train goat_gt goat_nogt
"""
from __future__ import annotations
import argparse, glob, os, sys, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from xmodal import data as D, sampling as S, model as M  # noqa: E402


@torch.no_grad()
def series_cls(prov_E, sc, K, voxels, dev, prisms=(32., 48., 64., 96., 128.),
               patch_sizes=(4., 6., 8., 12., 16.), n=96, seed=0):
    """Mean series-CLS over K random-prism slab bags (same recipe as the viz embed) — the frozen per-scan latent."""
    rng = np.random.default_rng(seed)
    wthin = sc.axis_map[sc.thick_axis]; unit = S.slab_unit_offsets(wthin, voxels, dev); fg = sc.foreground_mm
    ss = []
    for _ in range(K):
        pm = float(rng.choice(prisms)); a = fg[torch.randint(fg.shape[0], (1,), device=dev)]
        ctr = a[:, None] + (torch.rand(1, n, 3, device=dev) * 2 - 1) * (pm / 2)
        sz = S.draw_patch_sizes(rng, 1, n, patch_sizes, dev, False)
        pat = S.sample_patches_group(sc.volume, S.mixed_bag_vox(sc, ctr, sz, unit)).float()
        co = (ctr - a[:, None]).float(); ext = S.size_to_extent(sz, wthin).float()
        scl, _ = prov_E._cls_readout(pat, co, ext); ss.append(scl[0])
    return torch.stack(ss).mean(0).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prov-ckpt", required=True, help="frozen provenance checkpoint (run_provenance format)")
    ap.add_argument("--data-root", default="~/xmodal/data/brats26")
    ap.add_argument("--tracks", nargs="+", default=["mets_train", "ped_train", "goat_gt", "goat_nogt"])
    ap.add_argument("--out", required=True); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    ck = torch.load(os.path.expanduser(a.prov_ckpt), map_location="cpu"); cfg = ck.get("cfg", {})
    W, Dp, Hh, Vx = cfg.get("width", 768), cfg.get("depth", 12), cfg.get("heads", 12), cfg.get("voxels", 16)
    prov = M.Phase0Encoder(M.EncoderConfig(width=W, depth=Dp, heads=Hh, n_series=8, patch_voxels=Vx)).to(dev).eval()
    prov.load_state_dict(ck["model"]); [p.requires_grad_(False) for p in prov.parameters()]
    print(f"provenance encoder: width {W} depth {Dp} heads {Hh} voxels {Vx} | step {ck.get('step')}", flush=True)

    root = os.path.expanduser(a.data_root); pid2dir = {}
    for tr in a.tracks:
        d = os.path.join(root, tr)
        if os.path.isdir(d):
            pid2dir.update(D.find_brats_patients(d))
    def _complete(dd):
        return all(glob.glob(os.path.join(dd, "**", f"*-{suf}.nii.gz"), recursive=True) for suf in D.LOCAL_SUFFIX.values())
    pid2dir = {p: dd for p, dd in pid2dir.items() if _complete(dd)}
    pids = sorted(pid2dir)
    print(f"complete 4-modality patients: {len(pids)}", flush=True)

    out = {}; t0 = time.time()
    for i, pid in enumerate(pids):
        try:
            bundle = D.load_local_bundle(pid, pid2dir[pid], device=dev)[0]     # {modality: CachedScan}
        except Exception as e:
            print(f"skip {pid}: {type(e).__name__}", flush=True); continue
        for mod, sc in bundle.items():
            out[(pid, mod)] = series_cls(prov, sc, a.K, Vx, dev, seed=abs(hash((pid, mod))) % (2**31))
        del bundle; torch.cuda.empty_cache()
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(pids)} patients | {len(out)} scan-CLS | {time.time()-t0:.0f}s", flush=True)
    pickle.dump(out, open(os.path.expanduser(a.out), "wb"))
    print(f"DONE {len(out)} scan-CLS ({len(pids)} patients) -> {a.out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
