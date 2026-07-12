"""v5 cross-modal ordering pretraining (docs/MIXED_V5_DESIGN.md).

3D CUBE patches. Per bag: pick a target modality D; the encoder sees the other 3 modalities (random
positions) + a few D anchors; the decoder must recreate the ORDERING of held modality-D patches
(position matching, honest since all targets are D) + optional pixel MAE. Ablate the two losses via
--match-weight / --mae-weight (ordering-only / MAE-only / both). Branches off v2 (trusted encoder).

    python scripts/run_v5.py --wandb xmodal-mixed --steps 70000 --match-weight 1 --mae-weight 0.25 \
        --seed 0 --ckpt-dir runs/v5_both_s0 --wandb-run v5_both_s0
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch  # noqa: E402
from xmodal import data as D, holdout as H, model as M, train as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="~/xmodal/data/brats26")
    ap.add_argument("--tracks", nargs="+", default=["mets_train", "ped_train", "goat_gt", "goat_nogt"])
    ap.add_argument("--cache-size", type=int, default=32)
    ap.add_argument("--prefetch-workers", type=int, default=4)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--holdout-seed", type=int, default=0)
    ap.add_argument("--val-patients", type=int, default=16)
    ap.add_argument("--steps", type=int, default=70000)
    ap.add_argument("--batch-size", type=int, default=224)
    ap.add_argument("--n-src", type=int, default=90)
    ap.add_argument("--n-anchor", type=int, default=6)
    ap.add_argument("--n-tgt", type=int, default=32)
    ap.add_argument("--voxels", type=int, default=8, help="cube sample grid (voxels^3)")
    ap.add_argument("--prisms", type=float, nargs="+", default=[32., 64.])
    ap.add_argument("--match-weight", type=float, default=1.0, help="ordering loss weight (0 = MAE-only)")
    ap.add_argument("--mae-weight", type=float, default=0.25, help="pixel MAE weight (0 = ordering-only)")
    ap.add_argument("--tumor-frac", type=float, default=0.0, help="fraction of bags anchored on segmented tissue")
    ap.add_argument("--sampler-workers", type=int, default=0, help="parallel CPU-geometry prefetch workers (0=sync)")
    ap.add_argument("--ckpt-dir", default="runs/v5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--content-blur", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", default=None)
    ap.add_argument("--wandb-run", default=None)
    args = ap.parse_args()
    import subprocess
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__) or ".").decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=os.path.dirname(__file__) or ".").decode().strip()
    except Exception:
        git = branch = "unknown"
    print(f"GIT {branch}@{git} | v5 n_src={args.n_src} n_anchor={args.n_anchor} n_tgt={args.n_tgt} vox={args.voxels} "
          f"match_w={args.match_weight} mae_w={args.mae_weight} tumor={args.tumor_frac}", flush=True)
    dev = args.device
    root = os.path.expanduser(args.data_root)

    pid2dir = {}
    for tr in args.tracks:
        d = os.path.join(root, tr)
        if os.path.isdir(d):
            found = D.find_brats_patients(d); pid2dir.update(found); print(f"{tr}: {len(found)} patients", flush=True)
    import glob as _glob
    def _complete(dd):
        return all(_glob.glob(os.path.join(dd, "**", f"*-{suf}.nii.gz"), recursive=True) for suf in D.LOCAL_SUFFIX.values())
    n0 = len(pid2dir); pid2dir = {p: dd for p, dd in pid2dir.items() if _complete(dd)}
    print(f"complete 4-modality patients: {len(pid2dir)}/{n0}", flush=True)
    assert pid2dir
    all_pids = sorted(pid2dir)
    train_pids, val_pids = H.split_patients(all_pids, seed=args.holdout_seed, val_frac=args.holdout_frac)
    assert not H.contamination_check(train_pids, val_pids)
    print(f"total {len(all_pids)} | train {len(train_pids)} | val {len(val_pids)}", flush=True)

    _seg = args.tumor_frac > 0                                          # only pay seg-loading cost for tumor-focus
    loader = lambda pid: D.load_local_cpu(pid, pid2dir[pid], with_seg=_seg)[0]   # noqa: E731
    placer = lambda raw: D.place_bundle(raw, dev)                       # noqa: E731
    cache = D.JitteredRotatingCache(train_pids, loader, size=args.cache_size, placer=placer, warmup_log_every=8)
    cache.start_prefetch(workers=args.prefetch_workers, depth=8)
    val_bundles = [D.load_local_bundle(p, pid2dir[p], device=dev)[0] for p in val_pids[:args.val_patients]]

    torch.manual_seed(args.seed)
    enc = M.Phase0Encoder(M.EncoderConfig(width=384, depth=12, heads=6, n_series=8,
                                          patch_grid=(args.voxels,) * 3)).to(dev)      # 3D cube patches
    cfg = T.TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                        compile=not args.no_compile, content_blur=args.content_blur,
                        mae_weight=args.mae_weight, match_weight=args.match_weight,
                        v5_n_src=args.n_src, v5_n_anchor=args.n_anchor, v5_n_tgt=args.n_tgt, v5_voxels=args.voxels,
                        v5_prisms=tuple(args.prisms), tumor_frac=args.tumor_frac, v5_sampler_workers=args.sampler_workers,
                        git_commit=git, git_branch=branch,
                        ckpt_dir=args.ckpt_dir, wandb=args.wandb, wandb_run=args.wandb_run)
    print(f"model {sum(p.numel() for p in enc.parameters())/1e6:.1f}M | grid {enc.grid} pv {enc.pv} | bs {args.batch_size}", flush=True)
    T.train_v5(enc, cache, val_bundles, {}, cfg, device=dev)
    cache.stop_prefetch()


if __name__ == "__main__":
    main()
