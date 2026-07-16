"""Ticket worker — atomically claims pending ablation tickets and drains them on one GPU, reusing a single
loaded encoder across tickets. Concurrency is handled purely by os.rename (POSIX atomic on one filesystem):
whoever renames pending/<h>.json -> running/<h>.json first owns it; everyone else's rename raises and they
move on. No locks, no DB. Submit many of these across whatever GPUs are free — adding workers just drains faster.

  - crash/timeout recovery: a dead worker leaves a ticket in running/; the next worker sees its SLURM jobid is
    no longer in squeue and returns it to pending/.
  - heterogeneous GPUs: a 'small' worker (p100) releases 'big' tickets back for a 'big' worker; a config that
    OOMs on a small GPU is retagged big and requeued.
"""
import os, sys, json, time, socket, argparse, subprocess, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))          # scripts/ -> train_from_cache
import train_from_cache as T                            # noqa: E402  (also sets up src/ + marimo/ paths)
import infer                                            # noqa: E402
import torch                                            # noqa: E402


def claim(root):
    pend = os.path.join(root, "pending")
    for f in sorted(os.listdir(pend)):
        if not f.endswith(".json"):
            continue
        try:
            os.rename(os.path.join(pend, f), os.path.join(root, "running", f))   # atomic claim
            return os.path.join(root, "running", f)
        except OSError:
            continue                                    # another worker won the race
    return None


def reclaim_stale(root, lease=7200):
    """Return running tickets whose lease has expired (worker died mid-run) to pending. Time-based, NOT squeue-
    based: a freshly-claimed ticket (started ~now) is never reclaimed, so concurrent workers don't fight over it.
    lease=2h comfortably exceeds any single job's runtime."""
    run = os.path.join(root, "running")
    for f in os.listdir(run):
        if not f.endswith(".json"):
            continue
        p = os.path.join(run, f)
        try:
            t = json.load(open(p))
        except Exception:
            continue
        if time.time() - t.get("started", time.time()) > lease:
            try:
                os.rename(p, os.path.join(root, "pending", f))
                print(f"reclaimed stale {f} (lease expired)", flush=True)
            except OSError:
                pass


def _write(path, t):
    json.dump(t, open(path, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/cbica/home/gangarav/xmodal/tickets")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--data-root", default="data/brats26/mets_train")
    ap.add_argument("--gpu-tier", default="big", choices=["small", "big"], help="small workers skip big-tier tickets")
    ap.add_argument("--wall", type=int, default=5400, help="worker wall-clock budget (s); stop claiming near the end")
    ap.add_argument("--reserve", type=int, default=1200, help="s reserved so a claimed job can finish before the wall")
    ap.add_argument("--ckpt-root", default="/cbica/home/gangarav/xmodal/runs")
    a = ap.parse_args()
    t0 = time.time()
    jobid = os.environ.get("SLURM_JOB_ID", "local")
    E = {"path": None, "model": None}
    def get_E(cfg):
        """Load the encoder for this ticket's checkpoint, reloading only when it changes (so a checkpoint
        sweep works; a fixed-ckpt fleet never reloads). cfg['ckpt'] is 'run/step' -> runs/run/step.pt."""
        ck = cfg.get("ckpt")
        path = os.path.join(a.ckpt_root, ck + ".pt") if ck else a.ckpt
        if path != E["path"]:
            E["model"] = None; import torch as _t; _t.cuda.empty_cache()
            E["model"] = infer.load_model(path, dev="cuda"); E["path"] = path
            print(f"loaded encoder: {path}", flush=True)
        return E["model"]
    print(f"worker {jobid} @ {socket.gethostname()} tier={a.gpu_tier} wall={a.wall}s — up", flush=True)
    reclaim_stale(a.root)
    done = 0
    while time.time() - t0 < a.wall - a.reserve:
        tk = claim(a.root)
        if tk is None:
            print("no pending tickets — exiting", flush=True)
            break
        f = os.path.basename(tk)
        t = json.load(open(tk)); h = t["hash"]; cfg = t["cfg"]
        if a.gpu_tier == "small" and t.get("mem_tier") == "big":
            os.rename(tk, os.path.join(a.root, "pending", f))    # not for us; leave for a big worker
            time.sleep(2)
            continue
        t.update(worker=jobid, host=socket.gethostname(), started=time.time()); _write(tk, t)
        print(f"=== claim {h} · {cfg} ===", flush=True)
        try:
            row = T.run_ablation(get_E(cfg), cfg, a.cache, a.val_cache, a.data_root)
            t.update(status="done", metrics=row, elapsed=round(time.time() - t["started"])); _write(tk, t)
            os.rename(tk, os.path.join(a.root, "done", f))
            done += 1
            print(f"=== done {h} · ET {row.get('dsc_et')} · {t['elapsed']}s ===", flush=True)
        except Exception as e:
            torch.cuda.empty_cache()
            oom = ("out of memory" in str(e).lower()) or ("OutOfMemory" in type(e).__name__)
            t.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}",
                     traceback=traceback.format_exc()[-1500:], retries=t.get("retries", 0) + 1)
            if oom:
                t["needs"] = "big_gpu"; t["mem_tier"] = "big"
            # OOM on a small GPU (or first big failure) -> requeue for a big worker, up to 2 tries; else park in failed
            requeue = oom and a.gpu_tier == "small" and t["retries"] < 3
            dest = "pending" if requeue else "failed"
            _write(tk, t)
            os.rename(tk, os.path.join(a.root, dest, f))
            print(f"=== FAIL {h} -> {dest} · {t['error']} ===", flush=True)
    print(f"worker {jobid} exiting · {done} tickets done this run", flush=True)


if __name__ == "__main__":
    main()
