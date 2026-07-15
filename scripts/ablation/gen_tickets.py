"""Generate ablation tickets (one JSON per config) into <root>/pending/. Idempotent: a config already present
in pending/, running/, done/ or failed/ is skipped, so re-running only adds new cells. The sweep is
baseline-anchored — vary ONE axis at a time from BASELINE, plus seed repeats for error bars. Expand AXES to
add cells; a full factorial would be thousands of runs, so we sweep 1-D slices and add 2-D interactions by hand.
"""
import os, json, hashlib, argparse

BASELINE = dict(n_patients=100, n_tumor=6, n_neg=6, sampling="random", n_src=2048,
                epochs=20, epochs2=40, unfreeze=12, warmstart=True, seed=0)

AXES = {
    "n_patients": [40, 100, 300, 648],
    "n_src":      [256, 512, 1024, 2048, 4096],
    "sampling":   ["random", "cover", "hybrid"],
    "n_tumor":    [2, 6, 12],            # n_neg mirrors n_tumor
    "unfreeze":   [0, 4, 8, 12],
    "epochs":     [10, 20, 30],
    "epochs2":    [20, 40],
    "seed":       [0, 1, 2],
}

HKEYS = ["n_patients", "n_tumor", "n_neg", "sampling", "n_src", "epochs", "epochs2", "unfreeze", "warmstart", "seed"]


def cfg_hash(cfg):
    return hashlib.md5(json.dumps({k: cfg[k] for k in HKEYS}, sort_keys=True).encode()).hexdigest()[:12]


def mem_tier(cfg):
    # with the streaming source cache, GPU peak is per-prism (bounded), so tier is a soft hint only
    return "big" if cfg["n_src"] >= 2048 else "small"


def expand():
    cfgs = {}
    def add(cfg):
        cfg = dict(cfg); cfg["n_neg"] = cfg["n_tumor"]           # keep 1:1 tumor:neg
        cfgs[cfg_hash(cfg)] = cfg
    add(BASELINE)
    for axis, vals in AXES.items():
        for v in vals:
            add({**BASELINE, axis: v})
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/cbica/home/gangarav/xmodal/tickets")
    a = ap.parse_args()
    for d in ("pending", "running", "done", "failed"):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)
    have = set()
    for d in ("pending", "running", "done", "failed"):
        have |= {f[:-5] for f in os.listdir(os.path.join(a.root, d)) if f.endswith(".json")}
    cfgs = expand(); new = 0
    for h, cfg in cfgs.items():
        if h in have:
            continue
        t = dict(hash=h, cfg=cfg, mem_tier=mem_tier(cfg), status="pending")
        with open(os.path.join(a.root, "pending", f"{h}.json"), "w") as f:
            json.dump(t, f, indent=2)
        new += 1
    print(f"{len(cfgs)} configs in sweep · {new} new tickets · {len(cfgs) - new} already present")


if __name__ == "__main__":
    main()
