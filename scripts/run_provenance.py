"""Provenance pretraining (series-CLS + view-CLS ONLY, no ordering/MAE) on OpenMind (R2-streamed BIDS) +
BraTS, ViT-base. Slabs thin along each scan's native acquisition plane; patch shape embedded in PATIENT
space. series-CLS = rank-hinge (same-scan + same-modality/different-patient positives); view-CLS = 5-way
BCE (relative spatial order + window signs of two prisms from one scan).

  python scripts/run_provenance.py --wandb xmodal-prov --steps 150000 --om-pool 6000 --wandb-run prov_s0
"""
from __future__ import annotations
import argparse, glob, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from xmodal import data as D, sampling as S, model as M, openmind as OM  # noqa: E402

MODS = ["t1", "t1c", "t2", "flair"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brats-root", default="~/xmodal/data/brats26")
    ap.add_argument("--brats-tracks", nargs="+", default=["mets_train", "ped_train", "goat_gt", "goat_nogt"])
    ap.add_argument("--om-pool", type=int, default=6000, help="max OpenMind scans in the streaming pool")
    ap.add_argument("--brats-patients", type=int, default=2000, help="max BraTS patients in the pool")
    ap.add_argument("--cache-size", type=int, default=48); ap.add_argument("--prefetch-workers", type=int, default=6)
    ap.add_argument("--steps", type=int, default=150000); ap.add_argument("--batch-size", type=int, default=128)
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

    # ---- index OpenMind (R2) + BraTS (disk) into a flat item pool ----
    om_recs, n_mod, n_pat = OM.index_openmind()
    rng0 = np.random.default_rng(a.seed)
    if len(om_recs) > a.om_pool:
        om_recs = [om_recs[i] for i in rng0.permutation(len(om_recs))[:a.om_pool]]
    print(f"OpenMind: {len(om_recs)} scans in pool | {n_mod} modality classes | {n_pat} patients", flush=True)

    root = os.path.expanduser(a.brats_root); pid2dir = {}
    for tr in a.brats_tracks:
        d = os.path.join(root, tr)
        if os.path.isdir(d):
            pid2dir.update(D.find_brats_patients(d))
    def _complete(dd):
        return all(glob.glob(os.path.join(dd, "**", f"*-{suf}.nii.gz"), recursive=True) for suf in D.LOCAL_SUFFIX.values())
    pid2dir = {p: dd for p, dd in pid2dir.items() if _complete(dd)}
    bpids = sorted(pid2dir)
    if len(bpids) > a.brats_patients:
        bpids = [bpids[i] for i in sorted(rng0.permutation(len(bpids))[:a.brats_patients])]
    bpid_idx = {p: i for i, p in enumerate(bpids)}
    print(f"BraTS: {len(bpids)} complete 4-modality patients in pool", flush=True)

    # item = ("om", rec) | ("brats", pid, dir). Series/patient label spaces are OFFSET so BraTS and OpenMind
    # never form cross-domain positives (different scanners/domains).
    items = [("om", r) for r in om_recs] + [("brats", p, pid2dir[p]) for p in bpids]
    SER_OFF, PAT_OFF = n_mod, n_pat            # BraTS modality/patient ids start after OpenMind's

    def loader(i):                             # CPU-only (prefetch threads): stream/decode raw volumes
        it = items[i]
        if it[0] == "om":
            return ("om", OM.load_openmind_raw(it[1]), it[1])
        cpu, _ = D.load_local_cpu(it[1], it[2])          # {mod: (vol_np, affine, ...)}
        return ("brats", cpu, it[1])

    def placer(raw):                           # main thread: -> LIST of CachedScan (flatten in the loop)
        if raw[0] == "om":
            return [OM.place_openmind(raw[1], raw[2], device=dev)]
        cpu, pid = raw[1], raw[2]                         # cpu = {mod: _cpu_payload dict}
        bundle = D.place_bundle(cpu, dev)                 # proven BraTS GPU placement (dict -> CachedScan)
        out = []
        for mod in MODS:
            if mod not in bundle:
                continue
            sc, p = bundle[mod], cpu[mod]
            R = np.linalg.inv(np.asarray(p["affine_inv"], np.float32))
            aff = np.eye(4, dtype=np.float32); aff[:3, :3] = R; aff[:3, 3] = np.asarray(p["affine_trans"], np.float32)
            thick = S.native_or_axial_thick(p["spacing"], aff)      # BraTS axial: S-I through-plane
            sc.thick_axis = thick
            sc.world_thin_axis = int(np.argmax(np.abs(R[:, thick])))  # patient-space slab orientation
            sc.series_idx = SER_OFF + D.LOCAL_SERIES[mod]             # offset: no OpenMind/BraTS cross positives
            sc.patient = str(PAT_OFF + bpid_idx[pid])
            out.append(sc)
        return out

    cache = D.JitteredRotatingCache(list(range(len(items))), loader, size=a.cache_size, placer=placer, warmup_log_every=8)
    cache.start_prefetch(workers=a.prefetch_workers, depth=8)

    torch.manual_seed(a.seed)
    E = M.Phase0Encoder(M.EncoderConfig(width=a.width, depth=a.depth, heads=a.heads, n_series=8,
                                        patch_voxels=a.voxels)).to(dev)              # 2.5D slab stem (V,V,1)
    print(f"model {sum(p.numel() for p in E.parameters())/1e6:.1f}M | grid {E.grid} | bs {a.batch_size} | GIT {git}", flush=True)
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
            wb.init(project=a.wandb, name=a.wandb_run, config={**vars(a), "git": git})
        except Exception as e:
            print(f"[wandb] disabled: {e}", flush=True); wb = None

    def lr_at(step):
        if step < warmup:
            return a.lr * step / max(warmup, 1)
        p = (step - warmup) / max(a.steps - warmup, 1)
        return 0.5 * a.lr * (1 + np.cos(np.pi * min(p, 1.0)))

    os.makedirs(a.ckpt_dir, exist_ok=True)
    print("step   total   series  rel_sp  rel_wn  rel_acc  s_viol     lr", flush=True)
    t0 = time.time()
    for step in range(a.steps + 1):
        scans = [sc for lst in cache.resident() for sc in lst]         # flatten lists -> flat scan pool
        batch = S.sample_provenance_batch(scans, batch_size=a.batch_size, token_count=a.token_count,
                                          patch_sizes=tuple(a.patch_sizes), voxels=a.voxels,
                                          prism_choices=tuple(a.prisms), orient="scan", rng=rng, device=dev)
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
        if step % 50 == 0:
            print(f"{step:6d} {out['loss'].item():7.4f} {float(out['series']):7.4f} {float(out['rel_spatial']):7.4f} "
                  f"{float(out['rel_window']):7.4f} {out['rel_acc']:7.3f} {out['series_viol']:7.3f} {lr:.2e}", flush=True)
            if wb:
                wb.log({"train/total": out["loss"].item(), "train/series": float(out["series"]),
                        "train/rel_spatial": float(out["rel_spatial"]), "train/rel_window": float(out["rel_window"]),
                        "train/rel_acc": out["rel_acc"], "train/series_viol": out["series_viol"], "lr": lr}, step=step)
        if step > 0 and step % 5000 == 0:
            torch.save({"model": E.state_dict(), "step": step, "cfg": vars(a)}, os.path.join(a.ckpt_dir, f"step_{step:06d}.pt"))
    cache.stop_prefetch()
    print(f"provenance run done: {a.steps} steps in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
