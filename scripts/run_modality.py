"""v4 modality-completion pretraining (docs/MIXED_V4_DESIGN.md).

32 co-located foreground positions in a 64mm prism; at each position ONE modality is hidden (balanced
8 each of T1/T1c/T2/FLAIR) and the other 3 are visible. Predict the hidden modality's content from the
3 co-located + surrounding anatomy: position|target-modality matching + pixel MAE. Scan-relative target
(raw/z/CDF) + EMA target encoder. No view-CLS, curriculum, panels, or registers.

    python scripts/run_modality.py --wandb xmodal-mixed --steps 70000 --ema-color --seed 0 \
        --ckpt-dir runs/modality_s0 --wandb-run modality_s0
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
    ap.add_argument("--n-pos", type=int, default=32, help="positions per bag (targets = n_pos; source = 3*n_pos)")
    ap.add_argument("--prism", type=float, default=64.0)
    ap.add_argument("--patch", type=float, default=8.0)
    ap.add_argument("--ckpt-dir", default="runs/modality")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--content-blur", type=int, default=3)
    ap.add_argument("--orient", choices=["scan", "native", "random"], default="native")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema-color", action="store_true")
    ap.add_argument("--no-scan-context", action="store_true", help="disable scan-relative channels (ablation)")
    ap.add_argument("--wandb", default=None)
    ap.add_argument("--wandb-run", default=None)
    args = ap.parse_args()
    import subprocess
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__) or ".").decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=os.path.dirname(__file__) or ".").decode().strip()
    except Exception:
        git = branch = "unknown"
    print(f"GIT branch={branch} commit={git} | v4 modality-completion n_pos={args.n_pos} prism={args.prism} "
          f"patch={args.patch} ema={args.ema_color} scan_ctx={not args.no_scan_context}", flush=True)
    dev = args.device
    root = os.path.expanduser(args.data_root)

    pid2dir = {}
    for tr in args.tracks:
        d = os.path.join(root, tr)
        if os.path.isdir(d):
            found = D.find_brats_patients(d); pid2dir.update(found); print(f"{tr}: {len(found)} patients", flush=True)
    import glob as _glob                                             # v4 requires complete 4-modality patients
    def _complete(dd):
        return all(_glob.glob(os.path.join(dd, "**", f"*-{suf}.nii.gz"), recursive=True) for suf in D.LOCAL_SUFFIX.values())
    n0 = len(pid2dir); pid2dir = {p: dd for p, dd in pid2dir.items() if _complete(dd)}
    print(f"complete 4-modality patients: {len(pid2dir)}/{n0} ({n0 - len(pid2dir)} excluded)", flush=True)
    assert pid2dir, "v4 requires complete T1/T1c/T2/FLAIR patients"
    all_pids = sorted(pid2dir)
    train_pids, val_pids = H.split_patients(all_pids, seed=args.holdout_seed, val_frac=args.holdout_frac)
    assert not H.contamination_check(train_pids, val_pids)
    print(f"total {len(all_pids)} | train {len(train_pids)} | val {len(val_pids)}", flush=True)

    loader = lambda pid: D.load_local_cpu(pid, pid2dir[pid])[0]        # noqa: E731
    placer = lambda raw: D.place_bundle(raw, dev)                       # noqa: E731
    cache = D.JitteredRotatingCache(train_pids, loader, size=args.cache_size, placer=placer, warmup_log_every=8)
    cache.start_prefetch(workers=args.prefetch_workers, depth=8)
    val_bundles = [D.load_local_bundle(p, pid2dir[p], device=dev)[0] for p in val_pids[:args.val_patients]]

    torch.manual_seed(args.seed)
    enc = M.Phase0Encoder(M.EncoderConfig(width=384, depth=12, heads=6, n_series=8,
                                          scan_context=not args.no_scan_context)).to(dev)
    cfg = T.TrainConfig(steps=args.steps, batch_size=args.batch_size, n_pos=args.n_pos, lr=args.lr, seed=args.seed,
                        compile=not args.no_compile, content_blur=args.content_blur, orient=args.orient,
                        ema_color=args.ema_color, scan_context=not args.no_scan_context, mc_prism=args.prism,
                        mc_patch=args.patch, git_commit=git, git_branch=branch,
                        ckpt_dir=args.ckpt_dir, wandb=args.wandb, wandb_run=args.wandb_run)
    print(f"model {sum(p.numel() for p in enc.parameters())/1e6:.1f}M | bs {args.batch_size} | steps {args.steps}", flush=True)
    T.train_modality(enc, cache, val_bundles, {}, cfg, device=dev)
    cache.stop_prefetch()


if __name__ == "__main__":
    main()
