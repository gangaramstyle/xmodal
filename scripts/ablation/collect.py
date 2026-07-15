"""Collect finished tickets into a CSV + print a leaderboard-style summary. Reads <root>/done and <root>/failed."""
import os, sys, json, csv, argparse

COLS = ["n_patients", "n_tumor", "n_neg", "sampling", "n_src", "epochs", "epochs2", "unfreeze", "seed"]
MET = ["dsc_et", "nsd_et", "dsc_tc", "dsc_wt", "sm_f1_et", "n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/cbica/home/gangarav/xmodal/tickets")
    ap.add_argument("--out", default="/cbica/home/gangarav/xmodal/ablation_summary.csv")
    a = ap.parse_args()
    done, failed, pend, run = (len(os.listdir(os.path.join(a.root, d))) if os.path.isdir(os.path.join(a.root, d)) else 0
                               for d in ("done", "failed", "pending", "running"))
    rows = []
    dd = os.path.join(a.root, "done")
    for f in sorted(os.listdir(dd)) if os.path.isdir(dd) else []:
        if not f.endswith(".json"):
            continue
        t = json.load(open(os.path.join(dd, f))); m = t.get("metrics", {})
        rows.append({"hash": t["hash"], **{k: t["cfg"].get(k) for k in COLS},
                     **{k: m.get(k) for k in MET}, "elapsed": t.get("elapsed", 0)})
    print(f"tickets: {done} done · {failed} failed · {run} running · {pend} pending")
    if not rows:
        return
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    rows.sort(key=lambda r: -(r["dsc_et"] or 0))
    print(f"CSV -> {a.out}\ntop by ET DSC:")
    for r in rows[:15]:
        print(f"  ET {r['dsc_et']:.3f} TC {r['dsc_tc']:.3f} WT {r['dsc_wt']:.3f} smF1 {r['sm_f1_et']:.2f} | "
              f"np{r['n_patients']} nsrc{r['n_src']} {r['sampling']} unf{r['unfreeze']} "
              f"ep{r['epochs']}/{r['epochs2']} nt{r['n_tumor']} s{r['seed']}")
    fd = os.path.join(a.root, "failed")
    fails = [json.load(open(os.path.join(fd, f))) for f in os.listdir(fd) if f.endswith(".json")] if os.path.isdir(fd) else []
    if fails:
        print("\nfailed:")
        for t in fails[:10]:
            print(f"  {t['hash']} · {t.get('error', '?')}")


if __name__ == "__main__":
    main()
