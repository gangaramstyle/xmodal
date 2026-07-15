"""Generate ablation tickets (one JSON per config) into <root>/pending/. Idempotent: a config already present
in pending/, running/, done/ or failed/ is skipped, so re-running only adds new cells.

Three sets:
  --sweep   (default) baseline-anchored ONE-factor-at-a-time — vary each axis alone so every cell is directly
            attributable (a full factorial would be thousands of runs).
  --confirm four configs that reproduce the old molab readouts (ballpark + ordering check).
  --random N  N fully-random configs drawn from the joint space. After the sweep is done, predict each from
            baseline + summed main effects and compare to actual: a big residual ⇒ the axes interact (so OFAT
            under-describes the space and we owe some 2-D slices).

Sampling semantics: random = n_src random patches · cover = the fixed 2048 lattice (n_src ignored) · hybrid =
cover 2048 + a SMALL random add-on (n_src). canon() enforces these so hashes match the actual patches used.
"""
import os, json, hashlib, argparse, random

BASELINE = dict(n_patients=100, n_tumor=8, n_neg=2, sampling="random", n_src=2048,
                epochs=20, epochs2=40, unfreeze=12, warmstart=True, seed=0)

AXES = {                                    # vary one, hold the rest at BASELINE
    "n_patients": [40, 100, 300, 648],
    "n_src":      [256, 512, 1024, 2048, 4096],   # random-sampling patch count
    "n_tumor":    [4, 8, 12],
    "n_neg":      [0, 1, 2, 4],
    "unfreeze":   [0, 4, 8, 12],
    "epochs":     [10, 20, 30],
    "epochs2":    [20, 40],
    "seed":       [0, 1, 2],                       # error bars
}
SAMPLING = [dict(sampling="random", n_src=2048), dict(sampling="cover", n_src=2048),
            dict(sampling="hybrid", n_src=256)]    # random-2048 vs cover-2048 vs cover+256

# reproduce old molab readouts: n_train, sampling; neg≈0.1 (8 tumor : 1 neg), ep30/60 as those used
CONFIRM = [dict(n_patients=258, sampling="random", n_src=2048), dict(n_patients=512, sampling="cover", n_src=2048),
           dict(n_patients=40, sampling="random", n_src=2048), dict(n_patients=258, sampling="cover", n_src=2048)]
CONFIRM_OVR = dict(n_tumor=8, n_neg=1, epochs=30, epochs2=60, unfreeze=12)

# patient x epochs 2-D slice: does more DATA beat more EPOCHS? (epochs help most when data is scarce)
INTERACT = [dict(n_patients=p, epochs=e, epochs2=2 * e) for p in (40, 300, 648) for e in (10, 30)]

HKEYS = ["n_patients", "n_tumor", "n_neg", "sampling", "n_src", "epochs", "epochs2", "unfreeze", "warmstart", "seed"]


def canon(cfg):
    cfg = dict(cfg)
    if cfg["sampling"] == "cover":
        cfg["n_src"] = 2048                 # cover is the full lattice; n_src is moot -> canonicalize (dedup)
    return cfg


def cfg_hash(cfg):
    return hashlib.md5(json.dumps({k: cfg[k] for k in HKEYS}, sort_keys=True).encode()).hexdigest()[:12]


def mem_tier(cfg):
    return "big" if cfg["n_src"] >= 2048 else "small"    # soft hint; streaming cache bounds GPU peak regardless


def build(sweep, confirm, nrandom, interact=False):
    cfgs = {}
    def add(cfg):
        cfg = canon(cfg); cfgs[cfg_hash(cfg)] = cfg
    if interact:
        for c in INTERACT:
            add({**BASELINE, **c})
    if sweep:
        add(BASELINE)
        for axis, vals in AXES.items():
            for v in vals:
                add({**BASELINE, axis: v})
        for s in SAMPLING:
            add({**BASELINE, **s})
    if confirm:
        for c in CONFIRM:
            add({**BASELINE, **CONFIRM_OVR, **c})
    if nrandom:
        rng = random.Random(12345)          # reproducible config draws
        for _ in range(nrandom):
            cfg = {**BASELINE}
            for axis, vals in AXES.items():
                if axis == "seed":
                    continue
                cfg[axis] = rng.choice(vals)
            cfg["sampling"] = rng.choice(["random", "cover", "hybrid"])
            if cfg["sampling"] == "hybrid":
                cfg["n_src"] = rng.choice([128, 256, 512])
            add(cfg)
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/cbica/home/gangarav/xmodal/tickets")
    ap.add_argument("--sweep", action="store_true", help="the 1-D OFAT sweep (default if no set chosen)")
    ap.add_argument("--confirm", action="store_true", help="the 4 old-readout reproductions")
    ap.add_argument("--random", type=int, default=0, help="N random-joint spot-check configs")
    ap.add_argument("--interact", action="store_true", help="patient x epochs 2-D slice (data-vs-epochs interaction)")
    a = ap.parse_args()
    if not (a.sweep or a.confirm or a.random or a.interact):
        a.sweep = True
    for d in ("pending", "running", "done", "failed"):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)
    have = set()
    for d in ("pending", "running", "done", "failed"):
        have |= {f[:-5] for f in os.listdir(os.path.join(a.root, d)) if f.endswith(".json")}
    cfgs = build(a.sweep, a.confirm, a.random, a.interact); new = 0
    for h, cfg in cfgs.items():
        if h in have:
            continue
        json.dump(dict(hash=h, cfg=cfg, mem_tier=mem_tier(cfg), status="pending"),
                  open(os.path.join(a.root, "pending", f"{h}.json"), "w"), indent=2)
        new += 1
    print(f"{len(cfgs)} configs · {new} new tickets · {len(cfgs) - new} already present")


if __name__ == "__main__":
    main()
