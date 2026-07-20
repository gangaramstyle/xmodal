"""Dense decoder-head readout eval (the trusted 'best approach': infer.build_prisms -> train_readout ->
eval_readout -> lesionwise ET/TC/WT Dice) for cube- vs slab-source v5 encoders. Per checkpoint: train the
seg readout on mets_train prisms, eval on held-out METS. `src_shape` (cube|slab) sets the source patch
geometry to match each encoder's stem; infer.load_model auto-detects the stem grid. On-cluster (molab dies).

  python scripts/eval_slab_readout.py --checkpoints \
    cube:cube:runs/v5_multisize_vitbase_s0/step_070000.pt \
    slabrand:slab:runs/v5_slab_random_vitbase_s0/step_070000.pt
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "marimo"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import infer  # noqa: E402
from xmodal import data as D  # noqa: E402


def _patient_dirs(root, n):
    """Canonical BraTS patient finder (handles nested track dirs; a flat BraTS-* glob misses them)."""
    found = D.find_brats_patients(os.path.expanduser(root))   # {pid: dir}
    return [found[p] for p in sorted(found)][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="name:src_shape:globpath (src_shape=cube|slab)")
    ap.add_argument("--train-root", default="data/brats26/mets_train")
    ap.add_argument("--eval-root", default="/tmp/ho/mets_ho")
    ap.add_argument("--n-train", type=int, default=30); ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--n-pri", type=int, default=8); ap.add_argument("--n-src", type=int, default=2048)
    ap.add_argument("--res", type=float, default=1.0); ap.add_argument("--neg", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--epochs2", type=int, default=40)
    ap.add_argument("--unfreeze", type=int, default=12); ap.add_argument("--qsize", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    train_dirs = _patient_dirs(a.train_root, a.n_train)
    eval_dirs = _patient_dirs(a.eval_root, a.n_eval)
    print(f"train patients {len(train_dirs)} | eval patients {len(eval_dirs)} | "
          f"n_src={a.n_src} n_pri={a.n_pri} res={a.res} ep={a.epochs}/{a.epochs2} unf={a.unfreeze}", flush=True)
    assert train_dirs and eval_dirs
    rows = []
    for spec in a.checkpoints:
        nm, shape, path = spec.split(":", 2)
        g = sorted(glob.glob(path))
        if not g:
            print(f"{nm} NO_CKPT {path}", flush=True); continue
        t0 = time.time()
        E = infer.load_model(g[-1])
        print(f"[{nm}] loaded {os.path.basename(g[-1])} stem-grid {E.grid} src_shape={shape}", flush=True)
        tr = infer.build_prisms(train_dirs, n_prisms=a.n_pri, n_src=a.n_src, res=a.res, neg_frac=a.neg,
                                cache=False, seed=a.seed, src_shape=shape)
        print(f"[{nm}] built {len(tr)} train prisms ({time.time()-t0:.0f}s); training readout...", flush=True)
        ro = infer.train_readout(E, tr, epochs=a.epochs, epochs2=a.epochs2, unfreeze=a.unfreeze,
                                 warmstart=True, qsize=a.qsize, seed=a.seed,
                                 progress=lambda m: print(f"[{nm}]   {m}", flush=True))
        ev = infer.build_prisms(eval_dirs, n_prisms=a.n_pri, n_src=a.n_src, res=a.res, seed=1, src_shape=shape)
        res = infer.eval_readout(E, ro, ev)
        m = infer.leaderboard_metrics(res)
        line = (f"RESULT {nm} ({shape}): ET dsc {m['dsc_et']:.3f} nsd {m['nsd_et']:.3f} smF1 {m['sm_f1_et']:.2f} | "
                f"TC dsc {m['dsc_tc']:.3f} | WT dsc {m['dsc_wt']:.3f} | n={m['n']} | {time.time()-t0:.0f}s")
        print(line, flush=True); rows.append(line)
    print("\n======== SUMMARY ========")
    for r in rows:
        print(r, flush=True)


if __name__ == "__main__":
    main()
