"""Mixed-modality conditioned pretraining entrypoint (docs/MIXED_MODAL_DESIGN.md).

One continuous loop (no phases, no latent). Series is a per-patch conditioning signal on both encoder
tokens (Site A) and decoder queries (Site B); bags are variable mixed-modality; self<->cross is a
stochastic dominant-series alignment curriculum. `--ema-color` swaps the matching target to an EMA
color_head (the ablation arm).

    python scripts/run_mixed.py --wandb xmodal-mixed --steps 70000 --ema-color --seed 0 \
        --ckpt-dir runs/mixed_ema_s0 --wandb-run mixed_ema_s0
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from xmodal import data as D, holdout as H, model as M, train as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="~/xmodal/data/brats26")
    ap.add_argument("--tracks", nargs="+", default=["mets_train", "ped_train", "goat_gt", "goat_nogt"])
    ap.add_argument("--cache-size", type=int, default=64)
    ap.add_argument("--prefetch-workers", type=int, default=4)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--holdout-seed", type=int, default=0)
    ap.add_argument("--val-patients", type=int, default=16)
    ap.add_argument("--steps", type=int, default=70000, help="total training steps (single continuous loop)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--token-count", type=int, default=128, help="source-bag patches per item (encoder context)")
    ap.add_argument("--held-count", type=int, default=48, help="disjoint held/target positions per item")
    ap.add_argument("--ckpt-dir", default="runs/mixed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--size-per-bag", action="store_true", help="ablation: one patch size per bag")
    ap.add_argument("--patch-sizes", type=float, nargs="+", default=[4., 8., 16.])
    ap.add_argument("--content-blur", type=int, default=3, help="blur held-out contents before color head")
    ap.add_argument("--orient", choices=["scan", "native", "random"], default="native")
    ap.add_argument("--prism-choices", type=float, nargs="+", default=[32., 64., 128.])
    ap.add_argument("--lr", type=float, default=3e-4, help="base (peak) learning rate")
    ap.add_argument("--seed", type=int, default=0, help="training seed (data sampling + weight init)")
    ap.add_argument("--ema-color", action="store_true",
                    help="BYOL/DINO-style: slots match an EMA (target) color_head (single shared head). "
                         "The ablation arm — pair on/off at matched seeds for a clean A/B.")
    # alignment curriculum
    ap.add_argument("--align-ramp-frac", type=float, default=0.8,
                    help="target-dom==source-dom prob ramps 1->floor over this fraction of steps")
    ap.add_argument("--align-floor", type=float, default=0.1, help="min alignment prob (late same-modal floor)")
    ap.add_argument("--dom-lo", type=float, default=0.7, help="dominant-series share lower bound (sampled per item)")
    ap.add_argument("--dom-hi", type=float, default=0.95, help="dominant-series share upper bound")
    ap.add_argument("--wandb", default=None)
    ap.add_argument("--wandb-run", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint to resume from (weights + step + optimizer if present)")
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

    loader = lambda pid: D.load_local_cpu(pid, pid2dir[pid])[0]        # noqa: E731
    placer = lambda raw: D.place_bundle(raw, dev)                       # noqa: E731
    cache = D.JitteredRotatingCache(train_pids, loader, size=args.cache_size, placer=placer, warmup_log_every=8)
    cache.start_prefetch(workers=args.prefetch_workers, depth=8)
    val_bundles = [D.load_local_bundle(p, pid2dir[p], device=dev)[0] for p in val_pids[:args.val_patients]]

    torch.manual_seed(args.seed)                                       # seed WEIGHT init (before encoder build)
    enc = M.Phase0Encoder(M.EncoderConfig(width=384, depth=12, heads=6, n_series=8)).to(dev)
    cfg = T.TrainConfig(steps=args.steps, batch_size=args.batch_size, token_count=args.token_count,
                        held_count=args.held_count, lr=args.lr, seed=args.seed, compile=not args.no_compile,
                        size_per_bag=args.size_per_bag, patch_sizes=tuple(args.patch_sizes),
                        prism_choices=tuple(args.prism_choices), content_blur=args.content_blur, orient=args.orient,
                        ema_color=args.ema_color, align_ramp_frac=args.align_ramp_frac, align_floor=args.align_floor,
                        dom_lo=args.dom_lo, dom_hi=args.dom_hi,
                        ckpt_dir=args.ckpt_dir, wandb=args.wandb, wandb_run=args.wandb_run)
    print(f"model {sum(p.numel() for p in enc.parameters())/1e6:.1f}M | bs {args.batch_size} | "
          f"steps {args.steps} | ema_color {args.ema_color} | held {args.held_count}", flush=True)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=dev)
        enc.load_state_dict(ckpt["model"], strict=False)
        print(f"RESUME from {args.resume}: step {ckpt.get('step')} (weights loaded; fresh optimizer/schedule)", flush=True)
    T.train_mixed(enc, cache, val_bundles, {}, cfg, device=dev)
    cache.stop_prefetch()


if __name__ == "__main__":
    main()
