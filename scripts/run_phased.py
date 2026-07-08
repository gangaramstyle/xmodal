"""Phased pretraining entrypoint (self -> cross -> latent) on local BraTS-2026 data.

Discovers patients across tracks (METS/PED/GoAT nii dirs), applies the deterministic holdout,
builds a rotating GPU cache, and runs the phased curriculum with wandb.

    python scripts/run_phased.py --wandb xmodal-phased --self-steps 8000 --cross-steps 8000 --latent-steps 4000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from xmodal import data as D, holdout as H, model as M, sampling as S, train as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="~/xmodal/data/brats26")
    ap.add_argument("--tracks", nargs="+", default=["mets_train", "ped_train", "goat_gt", "goat_nogt"])
    ap.add_argument("--cache-size", type=int, default=64)
    ap.add_argument("--prefetch-workers", type=int, default=4)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--holdout-seed", type=int, default=0)
    ap.add_argument("--val-patients", type=int, default=16)
    ap.add_argument("--self-steps", type=int, default=25000)
    ap.add_argument("--cross-steps", type=int, default=25000)
    ap.add_argument("--latent-steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--token-count", type=int, default=128)
    ap.add_argument("--ckpt-dir", default="runs/phased")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--wandb", default=None)
    ap.add_argument("--wandb-run", default=None)
    args = ap.parse_args()
    dev = args.device
    root = os.path.expanduser(args.data_root)

    pid2dir = {}
    for tr in args.tracks:
        d = os.path.join(root, tr)
        if os.path.isdir(d):
            found = D.find_brats_patients(d)
            pid2dir.update(found)
            print(f"{tr}: {len(found)} patients", flush=True)
    all_pids = sorted(pid2dir)
    train_pids, val_pids = H.split_patients(all_pids, seed=args.holdout_seed, val_frac=args.holdout_frac)
    bad = H.contamination_check(train_pids, val_pids)
    assert not bad, f"contamination: {bad[:5]}"
    print(f"total {len(all_pids)} | train {len(train_pids)} | val {len(val_pids)}", flush=True)

    loader = lambda pid: D.load_local_cpu(pid, pid2dir[pid])[0]        # noqa: E731  (cpu bundle)
    placer = lambda raw: D.place_bundle(raw, dev)                       # noqa: E731
    cache = D.JitteredRotatingCache(train_pids, loader, size=args.cache_size, placer=placer, warmup_log_every=8)
    cache.start_prefetch(workers=args.prefetch_workers, depth=8)
    val_bundles = [D.load_local_bundle(p, pid2dir[p], device=dev)[0] for p in val_pids[:args.val_patients]]

    # 2.5D-only, multi-scale: slabs at 2/4/8/16 mm (fine -> coarse). 2.5D lets us pretrain on far
    # more data (thick-slice / non-isotropic: OpenMIND / FOMO / RSNA) than 3D cubes would.
    specs = {"slab4": S.slice_spec(4, 16), "slab8": S.slice_spec(8, 16), "slab16": S.slice_spec(16, 16)}
    enc = M.Phase0Encoder(M.EncoderConfig(width=384, depth=12, heads=6, n_series=8), list(specs.values())).to(dev)
    cfg = T.TrainConfig(batch_size=args.batch_size, token_count=args.token_count,
                        compile=not args.no_compile, ckpt_dir=args.ckpt_dir,
                        wandb=args.wandb, wandb_run=args.wandb_run)
    phases = [("self", args.self_steps), ("cross", args.cross_steps), ("latent", args.latent_steps)]
    print(f"model {sum(p.numel() for p in enc.parameters())/1e6:.1f}M | bs {args.batch_size} | phases {phases}", flush=True)
    T.train_phased(enc, cache, val_bundles, specs, cfg, phases=phases, device=dev)
    cache.stop_prefetch()


if __name__ == "__main__":
    main()
