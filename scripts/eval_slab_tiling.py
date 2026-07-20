"""Whole-brain TILED segmentation Dice for cube- vs slab-source v5 encoders (the honest metric: slide prisms
across the whole scan, classify every center, stitch votes -> voxel Dice + over-prediction). Per checkpoint:
train a readout on mets_train, then tile N held-out patients. src_shape (cube|slab) drives the source gather
only; tiling/decode/stitch identical. Reveals whether slab 'falls apart' on real segmentation vs cube.

  python scripts/eval_slab_tiling.py --checkpoints cube:cube:runs/v5_multisize_vitbase_s0/step_070000.pt \
      slabrand:slab:runs/v5_slab_random_vitbase_s0/step_070000.pt
"""
from __future__ import annotations
import argparse, glob, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "marimo"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
import infer  # noqa: E402
from xmodal import data as D  # noqa: E402

SAMP = {"random": dict(cover=False, hybrid=False), "cover": dict(cover=True, hybrid=False),
        "hybrid": dict(cover=False, hybrid=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="name:src_shape:globpath")
    ap.add_argument("--train-root", default="data/brats26/mets_train"); ap.add_argument("--eval-root", default="/tmp/ho/mets_ho")
    ap.add_argument("--n-train", type=int, default=100); ap.add_argument("--n-val-tile", type=int, default=4)
    ap.add_argument("--n-pri", type=int, default=8); ap.add_argument("--n-src", type=int, default=512)
    ap.add_argument("--sampling", choices=list(SAMP), default="random")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--epochs2", type=int, default=30)
    ap.add_argument("--unfreeze", type=int, default=12); ap.add_argument("--qsize", type=float, default=2.0)
    ap.add_argument("--stride-mm", type=float, default=16.0); ap.add_argument("--consensus", type=float, default=0.5)
    ap.add_argument("--min-vox", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--query", choices=["mm", "voxel"], default="mm")
    a = ap.parse_args()
    samp = SAMP[a.sampling]

    emap = D.find_brats_patients(os.path.expanduser(a.eval_root)); eval_pids = set(emap)
    val_dirs = [emap[p] for p in sorted(emap)][:a.n_val_tile]
    tmap = D.find_brats_patients(os.path.expanduser(a.train_root))
    train_dirs = [tmap[p] for p in sorted(tmap) if p not in eval_pids][:a.n_train]
    print(f"train {len(train_dirs)} (excl {len(eval_pids)} held-out) | tile-val {len(val_dirs)} | "
          f"sampling={a.sampling} n_src={a.n_src} stride={a.stride_mm}mm", flush=True)
    assert train_dirs and val_dirs
    rows = []
    for spec in a.checkpoints:
        nm, shape, path = spec.split(":", 2)
        g = sorted(glob.glob(path))
        if not g:
            print(f"{nm} NO_CKPT {path}", flush=True); continue
        t0 = time.time()
        E = infer.load_model(g[-1])
        print(f"[{nm}] loaded stem-grid {E.grid} src_shape={shape}; training readout...", flush=True)
        tr = infer.build_prisms(train_dirs, n_prisms=a.n_pri, n_src=a.n_src, res=1.0, neg_frac=0.3,
                                cache=False, seed=a.seed, src_shape=shape, query=a.query, **samp)
        ro = infer.train_readout(E, tr, epochs=a.epochs, epochs2=a.epochs2, unfreeze=a.unfreeze,
                                 warmstart=True, qsize=a.qsize, seed=a.seed)
        print(f"[{nm}] readout trained ({time.time()-t0:.0f}s); tiling {len(val_dirs)} val patients...", flush=True)
        dsc = {"et": [], "tc": [], "wt": []}; pv_et = []; gv_et = []
        for vd in val_dirs:
            pid = os.path.basename(vd)
            b = D.load_local_bundle(pid, vd, device="cuda", with_seg=True)[0]
            store = infer.stitch_tiles(b, ro, E, src_shape=shape, stride_mm=a.stride_mm,
                                       sampling=a.sampling, n_src=a.n_src, query=a.query, dev="cuda")
            m = infer.tile_metrics(store, consensus=a.consensus, min_vox=a.min_vox)
            for k in dsc:
                if m[f"dsc_{k}"] >= 0:
                    dsc[k].append(m[f"dsc_{k}"])
            pv_et.append(m["pred_vox_et"]); gv_et.append(m["gt_vox_et"])
            print(f"[{nm}]   {pid}: ET dsc {m['dsc_et']:.3f} (GT {m['gt_vox_et']} / pred {m['pred_vox_et']} vox) "
                  f"{store['n_tiles']} tiles", flush=True)
        mean = lambda x: float(np.mean(x)) if x else -1.0
        line = (f"TILE_RESULT {nm} ({shape}/{a.sampling}): whole-brain ET dsc {mean(dsc['et']):.3f} | "
                f"TC {mean(dsc['tc']):.3f} | WT {mean(dsc['wt']):.3f} | mean pred/gt ET vox "
                f"{int(np.mean(pv_et))}/{int(np.mean(gv_et))} | {len(val_dirs)} pts | {time.time()-t0:.0f}s")
        print(line, flush=True); rows.append(line)
    print("\n===== TILING SUMMARY =====")
    for r in rows:
        print(r, flush=True)


if __name__ == "__main__":
    main()
