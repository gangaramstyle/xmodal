"""Train + eval a seg readout from the precomputed prism cache (GPU job, fast — no rebuild/gather).

Loads N patients' cached prisms, slices the requested source sampling (random prefix or cover), fine-tunes
the fp32 two-stage readout on them, evaluates on the held-out val set (also cached or built), and appends a
row of BraTS-style metrics to a results CSV. The whole ablation is a sweep over the args below.
"""
import argparse, os, glob, csv, sys, hashlib
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "marimo"))
import infer  # reuse train_readout / eval_readout / leaderboard_metrics (fp32; amp off)
from xmodal import data as D, holdout as H


def load_prism(path, sampling, n_src):
    """Turn a cached .pt into the prism dict train_readout/eval_readout expect (sp/sc/sm/gpts/gt/imgs...)."""
    d = torch.load(path, map_location="cpu", weights_only=False)   # prism dict holds a numpy array (anch); torch>=2.6 needs this
    G = d["gdim"]; half = d["prism_mm"] / 2.0; res = d["res"]
    lin = np.arange(-half, half + 1e-3, res, dtype=np.float32)[:G]
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    gpts = np.stack([gx, gy, gz], -1).reshape(-1, 3).astype(np.float32)   # anchor-relative query grid
    if sampling == "cover":
        sp, sc, sm = d["sp_cover"].numpy(), d["sc_cover"].numpy(), d["sm_cover"].numpy().astype(np.int64)
    elif sampling == "hybrid":                                    # full cover + n_src random on top
        k = min(n_src, d["sp_rand"].shape[0])
        sp = np.concatenate([d["sp_cover"].numpy(), d["sp_rand"][:k].numpy()])
        sc = np.concatenate([d["sc_cover"].numpy(), d["sc_rand"][:k].numpy()])
        sm = np.concatenate([d["sm_cover"].numpy(), d["sm_rand"][:k].numpy()]).astype(np.int64)
    else:
        k = min(n_src, d["sp_rand"].shape[0])
        sp, sc, sm = d["sp_rand"][:k].numpy(), d["sc_rand"][:k].numpy(), d["sm_rand"][:k].numpy().astype(np.int64)
    return dict(sp=sp.astype(np.float16), sc=sc.astype(np.float32), sm=sm, gpts=gpts,
                gt=d["gt"].numpy().reshape(-1).astype(np.int8), gdim=G, pid=d["pid"], pname=d["pid"],
                anch=np.asarray(d["anch"]), prism_mm=d["prism_mm"], res=res,   # anch stored as numpy, not a tensor
                imgs=np.zeros((4, G, G, G), np.float16))   # imgs only used by the renderer, not train/eval


def gather(cache, pids, n_tumor, n_neg, sampling, n_src):
    out = []
    for pid in pids:
        d = os.path.join(cache, os.path.basename(pid))
        for kind, n in (("tumor", n_tumor), ("neg", n_neg)):
            for i in range(n):
                p = os.path.join(d, f"{kind}_{i}.pt")
                if os.path.exists(p):
                    out.append(load_prism(p, sampling, n_src))
    return out


DEFAULTS = dict(n_patients=100, n_tumor=6, n_neg=6, sampling="random", n_src=2048, n_test=51, test_tumor=8,
                epochs=20, epochs2=40, unfreeze=12, qsize=2.0, warmstart=True, src_cache_gb=None, seed=0)


def run_ablation(E, cfg, cache, val_cache, data_root):
    """Run one ablation config on an already-loaded encoder E; return the metrics row. Shared by the CLI and
    the ticket worker (which loads E once and calls this per ticket)."""
    c = {**DEFAULTS, **cfg}
    mets = D.find_brats_patients(data_root)
    train, val = H.split_patients(sorted(mets), seed=0, val_frac=0.1)
    tr_pids = [p for p in train[:c["n_patients"]] if os.path.isdir(os.path.join(cache, os.path.basename(p)))]
    te_pids = [p for p in val[:c["n_test"]] if os.path.isdir(os.path.join(val_cache, os.path.basename(p)))]
    tr = gather(cache, tr_pids, c["n_tumor"], c["n_neg"], c["sampling"], c["n_src"])
    te = gather(val_cache, te_pids, c["test_tumor"], 0, c["sampling"], c["n_src"])   # eval on tumor prisms only
    print(f"train {len(tr)} prisms / {len(tr_pids)} pt · eval {len(te)} prisms / {len(te_pids)} pt · "
          f"{c['sampling']}-{c['n_src']} · ep{c['epochs']}/{c['epochs2']} · unf{c['unfreeze']}", flush=True)
    ro = infer.train_readout(E, tr, epochs=c["epochs"], epochs2=c["epochs2"], unfreeze=c["unfreeze"],
                             warmstart=c["warmstart"], qsize=c["qsize"], src_cache_gb=c["src_cache_gb"], amp=False,
                             seed=c["seed"], progress=lambda m: print(m, flush=True))
    res = infer.eval_readout(E, ro, te)
    m = infer.leaderboard_metrics(res)
    row = dict(n_patients=len(tr_pids), n_tumor=c["n_tumor"], n_neg=c["n_neg"], sampling=c["sampling"], n_src=c["n_src"],
               epochs=c["epochs"], epochs2=c["epochs2"], unfreeze=c["unfreeze"], qsize=c["qsize"], seed=c["seed"],
               n_test=len(te), **m)
    print("RESULT " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="prism_cache"); ap.add_argument("--val-cache", default="prism_cache_val")
    ap.add_argument("--data-root", default="data/brats26/mets_train")
    ap.add_argument("--n-patients", type=int, default=100); ap.add_argument("--n-tumor", type=int, default=6)
    ap.add_argument("--n-neg", type=int, default=6); ap.add_argument("--sampling", default="random", choices=["random", "cover", "hybrid"])
    ap.add_argument("--n-src", type=int, default=2048)
    ap.add_argument("--n-test", type=int, default=51); ap.add_argument("--test-tumor", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--epochs2", type=int, default=40)
    ap.add_argument("--unfreeze", type=int, default=12); ap.add_argument("--qsize", type=float, default=2.0)
    ap.add_argument("--src-cache-gb", type=float, default=None, help="VRAM budget for the encode-once source cache; overflow is streamed (result-invariant). None=auto (half free VRAM)")
    ap.add_argument("--warmstart", action="store_true", default=True)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--enc-width", type=int, default=768); ap.add_argument("--enc-heads", type=int, default=12)
    ap.add_argument("--results", default="ablation_results.csv"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    E = infer.load_model(a.ckpt, width=a.enc_width, heads=a.enc_heads, dev="cuda")
    cfg = dict(n_patients=a.n_patients, n_tumor=a.n_tumor, n_neg=a.n_neg, sampling=a.sampling, n_src=a.n_src,
               n_test=a.n_test, test_tumor=a.test_tumor, epochs=a.epochs, epochs2=a.epochs2, unfreeze=a.unfreeze,
               qsize=a.qsize, warmstart=a.warmstart, src_cache_gb=a.src_cache_gb, seed=a.seed)
    row = run_ablation(E, cfg, a.cache, a.val_cache, a.data_root)
    exists = os.path.exists(a.results)
    with open(a.results, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
