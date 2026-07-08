"""Phase-0 pretraining entrypoint — runs on molab / Betty / CUBIC A40.

Builds a rotating GPU cache over BraTS patients, a mm-RoPE ViT encoder, and trains the full
phase-0 objective (MAE + series-CLS + view-CLS) with bf16 + torch.compile + fused AdamW.

    python scripts/run_phase0.py --patients 300 --cache-size 64 --batch-size 256 --steps 20000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from xmodal import data as D, model as M, sampling as S, train as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=300, help="train patients (cache cycles through these)")
    ap.add_argument("--holdout", type=int, default=16, help="held-out val patients")
    ap.add_argument("--cache-size", type=int, default=64, help="resident bundles on GPU")
    ap.add_argument("--prefetch-workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--token-count", type=int, default=128)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--ckpt-dir", default="runs/phase0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    dev = args.device

    pats = D.discover_brats_patients(limit=args.patients + args.holdout)
    train_pats, val_pats = pats[:-args.holdout], pats[-args.holdout:]
    loader = D.load_brats_cpu
    placer = lambda raw: D.place_bundle(raw, dev)  # noqa: E731
    print(f"cache: {min(args.cache_size, len(train_pats))} resident of {len(train_pats)} train patients", flush=True)
    cache = D.JitteredRotatingCache(train_pats, loader, size=args.cache_size, placer=placer, warmup_log_every=8)
    cache.start_prefetch(workers=args.prefetch_workers, depth=8)
    print(f"loading {len(val_pats)} val patients ...", flush=True)
    val_bundles = [D.load_brats_bundle(p, device=dev) for p in val_pats]

    specs = {"cube4": S.cube_spec(4, 16), "slab4": S.slice_spec(4, 16)}
    enc_cfg = M.EncoderConfig(width=args.width, depth=args.depth, heads=args.heads, n_series=8)
    enc = M.Phase0Encoder(enc_cfg, list(specs.values())).to(dev)
    cfg = T.TrainConfig(steps=args.steps, batch_size=args.batch_size, token_count=args.token_count,
                        compile=not args.no_compile, ckpt_dir=args.ckpt_dir)
    print(f"model {sum(p.numel() for p in enc.parameters())/1e6:.1f}M | bs {args.batch_size} | compile {not args.no_compile}", flush=True)
    T.train_phase0(enc, cache, val_bundles, specs, cfg, device=dev)
    cache.stop_prefetch()


if __name__ == "__main__":
    main()
