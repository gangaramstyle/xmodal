"""Precompute a prism cache for the readout ablation (CPU array job).

For each mets_train patient (held-out val excluded), build X tumor-anchored + Y non-tumor prisms. Each prism
stores BOTH a random-4096 source sampling and a full-coverage-2048 lattice, plus the GT grid at res=1mm.
One .pt per prism -> prism_cache/<pid>/{tumor,neg}_<i>.pt. Seeds are deterministic (hash of pid:kind:idx),
so any (n_patients, n_tumor, n_neg, n_src<=4096 prefix, sampling) subset is reproducible and rebuild-free.

Gather runs on CPU (grid_sample) — slower per patient, but parallelized across a big SLURM array.
"""
import argparse, os, glob, hashlib, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xmodal import data as D, sampling as S, holdout as H

MODS = ["t1", "t1c", "t2", "flair"]


def _seed(*parts):
    return int(hashlib.md5(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _gather(b, cs, sm, size, unit, dev):
    """Gather 8^3 source cubes at world-mm centers cs with per-cube modality sm. Returns (N,8,8,8) float16."""
    sp = np.zeros((len(cs), 8, 8, 8), np.float16)
    for mi, m in enumerate(MODS):
        sel = np.nonzero(sm == mi)[0]
        if sel.size:
            sp[sel] = S._gather_cubes(b[m], torch.as_tensor(cs[sel], device=dev),
                                      torch.full((sel.size,), size, device=dev), unit, dev).half().cpu().numpy()
    return sp


def _seg_at(scan, at, ai, pts_mm, dev):
    vox = ((pts_mm - at) @ ai.T).round().long()
    shp = torch.as_tensor(scan.seg_vol.shape, device=dev)
    vox = vox.clamp(min=torch.zeros(3, device=dev, dtype=torch.long), max=shp - 1)
    return scan.seg_vol[vox[:, 0], vox[:, 1], vox[:, 2]].long()


def build_patient(pid, pdir, a, unit, dev):
    b = D.load_local_bundle(pid, pdir, device=dev, with_seg=True)[0]
    sc0 = b["t1c"]; tmm = sc0.tumor_np
    if tmm is None or len(tmm) == 0:
        return []
    fgn = sc0.foreground_np if sc0.foreground_np is not None else sc0.foreground_mm.cpu().numpy()
    at, ai = sc0.affine_trans, sc0.affine_inv
    half = a.prism_mm / 2.0
    lin = np.arange(-half, half + 1e-3, a.res, dtype=np.float32); G = len(lin)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij"); grid = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    # full-coverage lattice (shared): (prism_mm/size)^3 cells x 4 modalities
    nper = int(round(a.prism_mm / a.size))
    cl = np.arange(-half + a.size / 2, half, a.size, dtype=np.float32)[:nper]
    cgx, cgy, cgz = np.meshgrid(cl, cl, cl, indexing="ij")
    lat = np.stack([cgx, cgy, cgz], -1).reshape(-1, 3)
    cover_rel = np.tile(lat, (4, 1)).astype(np.float32); cover_sm = np.repeat(np.arange(4), len(lat))
    out = []
    for kind, npr in (("tumor", a.n_tumor), ("neg", a.n_neg)):
        for i in range(npr):
            rng = np.random.default_rng(_seed(pid, kind, i))
            if kind == "tumor":
                anch = tmm[rng.integers(len(tmm))].astype(np.float32)
            else:                                                     # hard-negative: foreground with no tumor in prism
                cand = fgn[rng.integers(len(fgn), size=128)].astype(np.float32)
                free = cand[~(np.abs(tmm[None] - cand[:, None]).max(-1) <= half).any(1)]
                if len(free) == 0:
                    continue
                anch = free[0]
            cm = fgn[np.abs(fgn - anch).max(-1) <= half]              # foreground inside the prism
            if len(cm) < a.n_src // 2:
                continue
            gpts = (grid + anch).astype(np.float32)
            cs = cm[rng.integers(len(cm), size=a.n_src)].astype(np.float32)   # random-N centers
            sm = rng.integers(4, size=a.n_src)
            sp_rand = _gather(b, cs, sm, a.size, unit, dev)
            ccs = (cover_rel + anch).astype(np.float32)
            sp_cover = _gather(b, ccs, cover_sm, a.size, unit, dev)
            gt = _seg_at(sc0, at, ai, torch.as_tensor(gpts, device=dev), dev)
            gt = torch.where(gt == 4, torch.full_like(gt, 3), gt).cpu().numpy().astype(np.int8)
            out.append(dict(
                pid=pid, kind=kind, idx=i, anch=anch.astype(np.float32), prism_mm=float(a.prism_mm),
                res=float(a.res), size=float(a.size), gdim=int(G),
                sp_rand=torch.from_numpy(sp_rand), sc_rand=torch.from_numpy((cs - anch).astype(np.float32)),
                sm_rand=torch.from_numpy(sm.astype(np.int8)),
                sp_cover=torch.from_numpy(sp_cover), sc_cover=torch.from_numpy(cover_rel),
                sm_cover=torch.from_numpy(cover_sm.astype(np.int8)),
                gt=torch.from_numpy(gt)))
    del b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/brats26/mets_train")
    ap.add_argument("--out", default="prism_cache")
    ap.add_argument("--split", default="train", choices=["train", "val"], help="which hash-split to cache")
    ap.add_argument("--n-patients", type=int, default=648, help="use first N of the chosen split")
    ap.add_argument("--n-tumor", type=int, default=12); ap.add_argument("--n-neg", type=int, default=12)
    ap.add_argument("--n-src", type=int, default=4096, help="random source patches to store (slice a prefix at train)")
    ap.add_argument("--size", type=float, default=4.0); ap.add_argument("--prism-mm", type=float, default=32.0)
    ap.add_argument("--res", type=float, default=1.0)
    ap.add_argument("--task", type=int, default=0); ap.add_argument("--ntask", type=int, default=1)
    a = ap.parse_args()
    dev = "cpu"
    unit = S._cube_unit(8, dev)
    mets = D.find_brats_patients(a.data_root)
    train, val = H.split_patients(sorted(mets), seed=0, val_frac=0.1)
    train = (val if a.split == "val" else train)[:a.n_patients]
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", a.task))
    ntask = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", a.ntask))
    mine = train[task::ntask]                                          # stride chunking across array
    print(f"[task {task}/{ntask}] {len(mine)} of {len(train)} patients | {a.n_tumor}T+{a.n_neg}N prisms, "
          f"random-{a.n_src} + cover, res={a.res}mm", flush=True)
    done = 0
    for k, pid in enumerate(mine):
        outd = os.path.join(a.out, os.path.basename(pid))
        expect = a.n_tumor + a.n_neg
        if os.path.isdir(outd) and len(glob.glob(f"{outd}/*.pt")) >= expect:
            done += 1; continue                                       # resumable: skip finished patients
        try:
            prisms = build_patient(pid, mets[pid], a, unit, dev)
        except Exception as e:
            print(f"  SKIP {pid}: {type(e).__name__} {str(e)[:100]}", flush=True); continue
        os.makedirs(outd, exist_ok=True)
        for pr in prisms:
            torch.save(pr, os.path.join(outd, f"{pr['kind']}_{pr['idx']}.pt"))
        done += 1
        if k % 5 == 0:
            print(f"  [{task}] {k+1}/{len(mine)} · {pid} · {len(prisms)} prisms", flush=True)
    print(f"[task {task}] DONE {done}/{len(mine)} patients", flush=True)


if __name__ == "__main__":
    main()
