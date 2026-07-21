"""Provenance pretraining (series-CLS + view-CLS ONLY, no ordering/MAE) on OpenMind via the METADATA MANIFEST:
all ~37.5k structural images / 15 contrasts, series label = (dataset, modality), contrast + acquisition-plane
BALANCED sampling, a prism-CLOSENESS curriculum (a/b prisms start far, end near), native resolution, and
orientation-correct slabs (thin along the true world through-plane). OpenMind-only (BraTS dropped).

  python scripts/run_provenance.py --wandb xmodal-prov --steps 150000 --wandb-run prov_manifest_s0
"""
from __future__ import annotations
import argparse, os, sys, time
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from xmodal import data as D, sampling as S, model as M, openmind as OM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--om-pool", type=int, default=12000, help="scans drawn (weighted) into the streaming pool")
    ap.add_argument("--contrast-alpha", type=float, default=0.5, help="contrast inverse-freq strength (0 uniform,1 full)")
    ap.add_argument("--coronal-boost", type=float, default=3.0); ap.add_argument("--sagittal-boost", type=float, default=6.0)
    ap.add_argument("--plane-cache", default="~/xmodal/openmind_planes.json", help="cached native-plane probe (json)")
    ap.add_argument("--pd-start", type=float, nargs=2, default=[32., 160.], help="a/b prism dist (mm) EARLY (easy, far)")
    ap.add_argument("--pd-end", type=float, nargs=2, default=[8., 48.], help="a/b prism dist (mm) LATE (hard, near)")
    ap.add_argument("--cache-size", type=int, default=48); ap.add_argument("--prefetch-workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150000); ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--token-count", type=int, default=128); ap.add_argument("--voxels", type=int, default=16)
    ap.add_argument("--patch-sizes", type=float, nargs="+", default=[4., 8., 16.])
    ap.add_argument("--prisms", type=float, nargs="+", default=[32., 64., 128.])
    ap.add_argument("--series-weight", type=float, default=1.0); ap.add_argument("--n-xmod", type=int, default=1)
    ap.add_argument("--rel-spatial-weight", type=float, default=0.25); ap.add_argument("--rel-window-weight", type=float, default=0.25)
    ap.add_argument("--width", type=int, default=768); ap.add_argument("--depth", type=int, default=12); ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", default="runs/prov"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--wandb", default=None); ap.add_argument("--wandb-run", default=None)
    a = ap.parse_args()
    dev = a.device
    import subprocess
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__) or ".").decode().strip()
    except Exception:
        git = "unknown"

    # ---- index OpenMind via the manifest + native-plane probe + contrast/plane-weighted pool draw ----
    recs, n_mod, n_pat = OM.index_openmind_manifest()
    print(f"manifest: {len(recs)} imgs | {n_mod} series(dataset,modality) | {n_pat} patients | GIT {git}", flush=True)
    OM.probe_planes(recs, cache_path=os.path.expanduser(a.plane_cache))
    print(f"native planes (all): {dict(Counter(r['plane'] for r in recs))}", flush=True)
    w = OM.sampling_weights(recs, contrast_alpha=a.contrast_alpha,
                            plane_boost={"axial": 1.0, "coronal": a.coronal_boost, "sagittal": a.sagittal_boost})
    rng0 = np.random.default_rng(a.seed)
    npool = min(a.om_pool, len(recs))
    items = [recs[i] for i in rng0.choice(len(recs), size=npool, replace=False, p=w)]
    print(f"POOL {len(items)} | contrasts {dict(Counter(r['modality'] for r in items).most_common(10))}", flush=True)
    print(f"POOL planes {dict(Counter(r['plane'] for r in items))}", flush=True)

    def loader(i):                             # CPU-only (prefetch threads): stream/decode one scan
        return ("om", OM.load_openmind_raw(items[i]), items[i])

    def placer(raw):                           # main thread -> LIST of CachedScan (flatten in the loop)
        return [OM.place_openmind(raw[1], raw[2], device=dev)]

    cache = D.JitteredRotatingCache(list(range(len(items))), loader, size=a.cache_size, placer=placer, warmup_log_every=8)
    cache.start_prefetch(workers=a.prefetch_workers, depth=8)

    torch.manual_seed(a.seed)
    E = M.Phase0Encoder(M.EncoderConfig(width=a.width, depth=a.depth, heads=a.heads, n_series=8,
                                        patch_voxels=a.voxels)).to(dev)              # 2.5D slab stem (V,V,1)
    print(f"model {sum(p.numel() for p in E.parameters())/1e6:.1f}M | grid {E.grid} | bs {a.batch_size}", flush=True)
    opt = torch.optim.AdamW(E.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95), fused=(dev == "cuda"))
    if not a.no_compile:
        try:
            E.encode = torch.compile(E.encode, dynamic=True)
        except Exception as e:
            print(f"[compile] skipped: {e}", flush=True)
    rng = np.random.default_rng(a.seed)
    warmup = int(0.05 * a.steps)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda"))
    wb = None
    if a.wandb:
        try:
            import wandb as wb
            wb.init(project=a.wandb, name=a.wandb_run, config={**vars(a), "git": git, "n_series": n_mod, "pool": len(items)})
        except Exception as e:
            print(f"[wandb] disabled: {e}", flush=True); wb = None

    def lr_at(step):
        if step < warmup:
            return a.lr * step / max(warmup, 1)
        p = (step - warmup) / max(a.steps - warmup, 1)
        return 0.5 * a.lr * (1 + np.cos(np.pi * min(p, 1.0)))

    def pd_at(step):                           # prism-closeness curriculum: interpolate (min,max) start -> end
        p = min(step / max(a.steps, 1), 1.0)
        return (a.pd_start[0] + (a.pd_end[0] - a.pd_start[0]) * p,
                a.pd_start[1] + (a.pd_end[1] - a.pd_start[1]) * p)

    os.makedirs(a.ckpt_dir, exist_ok=True)
    print("step   total   series  rel_sp  rel_wn  rel_acc  s_viol   pd_lo  pd_hi     lr", flush=True)
    t0 = time.time()
    for step in range(a.steps + 1):
        scans = [sc for lst in cache.resident() for sc in lst]         # flatten lists -> flat scan pool
        pdmin, pdmax = pd_at(step)
        batch = S.sample_provenance_batch(scans, batch_size=a.batch_size, token_count=a.token_count,
                                          patch_sizes=tuple(a.patch_sizes), voxels=a.voxels,
                                          prism_choices=tuple(a.prisms), orient="scan", rng=rng, device=dev,
                                          pair_dist_min=pdmin, pair_dist_max=pdmax)
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast(**amp):
            out = E.forward_provenance(batch, series_weight=a.series_weight, n_xmod=a.n_xmod,
                                       rel_spatial_weight=a.rel_spatial_weight, rel_window_weight=a.rel_window_weight)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(E.parameters(), 1.0)
        opt.step()
        cache.step()                                                  # rotate: stream fresh scans from R2
        if step % 50 == 0:
            print(f"{step:6d} {out['loss'].item():7.4f} {float(out['series']):7.4f} {float(out['rel_spatial']):7.4f} "
                  f"{float(out['rel_window']):7.4f} {out['rel_acc']:7.3f} {out['series_viol']:7.3f} {pdmin:6.1f} {pdmax:6.1f} {lr:.2e}", flush=True)
            if wb:
                wb.log({"train/total": out["loss"].item(), "train/series": float(out["series"]),
                        "train/rel_spatial": float(out["rel_spatial"]), "train/rel_window": float(out["rel_window"]),
                        "train/rel_acc": out["rel_acc"], "train/series_viol": out["series_viol"],
                        "sched/pd_min": pdmin, "sched/pd_max": pdmax, "lr": lr}, step=step)
        if step > 0 and step % 5000 == 0:
            torch.save({"model": E.state_dict(), "step": step, "cfg": vars(a)}, os.path.join(a.ckpt_dir, f"step_{step:06d}.pt"))
    cache.stop_prefetch()
    print(f"provenance run done: {a.steps} steps in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
