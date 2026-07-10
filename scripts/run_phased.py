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
    ap.add_argument("--size-per-bag", action="store_true",
                    help="ablation: one patch size per bag (no within-bag scale mixing)")
    ap.add_argument("--patch-sizes", type=float, nargs="+", default=[4., 8., 16.],
                    help="per-patch physical sizes (mm) to sample from; pass one for single-size")
    ap.add_argument("--content-blur", type=int, default=3, help="blur held-out contents before color head")
    ap.add_argument("--orient", choices=["scan", "native", "random"], default="scan",
                    help="2.5D slab orientation: scan (geometry), native (outlier/acquisition axis), random")
    ap.add_argument("--held-size", type=float, nargs="+", default=[8., 16.],
                    help="held-out matching/recon TARGET size(s) in mm (4mm stays input-only); pass 0 for any-size")
    ap.add_argument("--wandb", default=None)
    ap.add_argument("--wandb-run", default=None)
    ap.add_argument("--resume", default=None,
                    help="path to a checkpoint (step_XXXXXX.pt) to resume from: restores weights + step "
                         "(LR schedule position), and optimizer state if present in the checkpoint")
    ap.add_argument("--init-from", default=None,
                    help="load ONLY the weights from a checkpoint and start a FRESH schedule at step 0 "
                         "(for forking a new phase off an existing encoder; teacher snapshots from these weights)")
    ap.add_argument("--lr", type=float, default=3e-4, help="base (peak) learning rate")
    ap.add_argument("--prism-choices", type=float, nargs="+", default=[32., 64., 128.],
                    help="per-item prism extents (mm) to sample from")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="phase-4 faithful LATENT fork: freeze the encoder and train ONLY the decoder+latent_head "
                         "(no self-MAE). Teacher == the frozen encoder; targets stationary.")
    ap.add_argument("--seed", type=int, default=0, help="training seed (data sampling + init); vary for a battery")
    ap.add_argument("--soft-match-tau", type=float, default=None,
                    help="similarity-softened matching target temperature (e.g. 0.1-0.2); omit for hard identity. "
                         "Near-identical patches share the positive so ambiguous confusions aren't penalized.")
    ap.add_argument("--soft-match-sim", choices=["model", "pixel"], default="model",
                    help="soft-target similarity source: 'pixel' = FIXED raw blurred pixels (non-circular, "
                         "collapse-safe); 'model' = trainable color_head (circular/collapse-prone).")
    ap.add_argument("--latent-only", action="store_true",
                    help="latent phase runs ONLY the latent loss (skip self-MAE); encoder still TRAINS unless "
                         "--freeze-encoder. Use for the online-encoder ablation (native_latent minus the freeze).")
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

    # 2.5D-only, MIXED multi-scale: per-patch physical size {4,8,16} mm + per-item prism {32,64,128} mm
    # (TrainConfig defaults). One shared stem/heads; scale rides in via the per-patch size embedding.
    torch.manual_seed(args.seed)                                       # seed WEIGHT init too (before encoder build)
    enc = M.Phase0Encoder(M.EncoderConfig(width=384, depth=12, heads=6, n_series=8)).to(dev)
    held = None if (len(args.held_size) == 1 and args.held_size[0] == 0) else tuple(args.held_size)
    cfg = T.TrainConfig(batch_size=args.batch_size, token_count=args.token_count, lr=args.lr, seed=args.seed,
                        compile=not args.no_compile, size_per_bag=args.size_per_bag,
                        patch_sizes=tuple(args.patch_sizes), prism_choices=tuple(args.prism_choices),
                        content_blur=args.content_blur, orient=args.orient, held_size=held,
                        freeze_encoder=args.freeze_encoder, latent_only=args.latent_only,
                        soft_match_tau=args.soft_match_tau, soft_match_sim=args.soft_match_sim,
                        ckpt_dir=args.ckpt_dir, wandb=args.wandb, wandb_run=args.wandb_run)
    phases = [("self", args.self_steps), ("cross", args.cross_steps), ("latent", args.latent_steps)]
    print(f"model {sum(p.numel() for p in enc.parameters())/1e6:.1f}M | bs {args.batch_size} | phases {phases}", flush=True)
    resume_step, resume_opt = 0, None
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=dev)
        enc.load_state_dict(ckpt["model"])                       # weights only -> fresh schedule at step 0
        print(f"INIT-FROM {args.init_from}: loaded weights (was step {ckpt.get('step')}), fresh schedule; "
              f"teacher will snapshot from these weights at the first non-self phase", flush=True)
    elif args.resume:
        ckpt = torch.load(args.resume, map_location=dev)
        enc.load_state_dict(ckpt["model"]); resume_step = int(ckpt["step"]); resume_opt = ckpt.get("opt")
        print(f"RESUME from {args.resume}: step {resume_step}, phase {ckpt.get('phase')}, "
              f"optimizer {'restored' if resume_opt is not None else 'FRESH (not in ckpt)'}", flush=True)
    T.train_phased(enc, cache, val_bundles, {}, cfg, phases=phases, device=dev,
                   resume_step=resume_step, resume_opt_state=resume_opt)
    cache.stop_prefetch()


if __name__ == "__main__":
    main()
