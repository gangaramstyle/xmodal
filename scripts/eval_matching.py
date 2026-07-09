"""On-distribution matching eval + visualization for a phased checkpoint.

Loads a checkpoint, samples bags from HELD-OUT val patients (the real METS/PED/GoAT training
distribution), runs the phase-0 self matching, reports match_acc, and saves a viz montage:
for K held-out slots, [truth | model's top-matched patch], green border = correct match.

    python scripts/eval_matching.py --ckpt runs/phased/step_011000.pt --out runs/eval
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from xmodal import sampling as S, model as M, data as D, holdout as H  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="~/xmodal/data/brats26")
    ap.add_argument("--tracks", nargs="+", default=["mets_train", "ped_train", "goat_gt", "goat_nogt"])
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--val-patients", type=int, default=12)
    ap.add_argument("--token-count", type=int, default=128)
    ap.add_argument("--mask-ratio", type=float, default=0.35)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    dev = "cuda"
    root = os.path.expanduser(args.data_root)

    pid2dir = {}
    for tr in args.tracks:
        d = os.path.join(root, tr)
        if os.path.isdir(d):
            pid2dir.update(D.find_brats_patients(d))
    all_pids = sorted(pid2dir)
    _, val_pids = H.split_patients(all_pids, seed=0, val_frac=0.1)
    val_pids = val_pids[:args.val_patients]
    print(f"val patients: {len(val_pids)} (of {len(all_pids)} total)", flush=True)
    bundles = [D.load_local_bundle(p, pid2dir[p], device=dev)[0] for p in val_pids]

    ck = torch.load(os.path.expanduser(args.ckpt), map_location=dev)
    enc = M.Phase0Encoder(M.EncoderConfig(width=384, depth=12, heads=6, n_series=8)).to(dev).eval()
    enc.load_state_dict(ck["model"])
    print(f"loaded {args.ckpt} (step {ck.get('step')})", flush=True)

    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    rng = np.random.default_rng(0)
    accs = []
    viz = None
    with torch.no_grad():
        for i in range(args.eval_batches):
            b = S.sample_paired_batch(bundles, batch_size=args.batch_size, token_count=args.token_count,
                                      rng=rng, device=dev)
            with torch.autocast(**amp):
                o = enc.forward_self(b["patches_a"], b["coords_a"], b["sizes_a"], mask_ratio=args.mask_ratio)
            accs.append(float(o["match_acc"]))
            if viz is None:
                # slot->color argmax on item 0, build a [truth | matched] montage
                sim = (o["slots"][0].float() @ o["colors"][0].float().T)   # [Q,Q]
                pick = sim.argmax(1).cpu().numpy()                         # slot -> chosen color idx
                held = o["held_patches"][0, :, :, :, 0].float().cpu().numpy()  # [Q,16,16]
                viz = (held, pick, o["held_sizes"][0].cpu().numpy() if "held_sizes" in o else None)
    macc = float(np.mean(accs))
    print(f"ON-DISTRIBUTION match_acc = {macc:.3f} (+/- {np.std(accs):.3f}) over {len(accs)} bags, "
          f"chance ~ {1.0/max(1,int(round(args.token_count*args.mask_ratio))):.3f}", flush=True)

    # ---- viz montage: K slots, [truth | matched], green if correct ----
    try:
        from PIL import Image
        held, pick, _ = viz
        K = min(16, held.shape[0])
        def n8(a):
            a = a.astype(np.float32); lo, hi = a.min(), a.max()
            return ((a - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
        cell = 40
        rows = []
        for k in range(K):
            truth = np.array(Image.fromarray(n8(held[k])).resize((cell, cell)))
            matched = np.array(Image.fromarray(n8(held[pick[k]])).resize((cell, cell)))
            correct = (pick[k] == k)
            pair = np.stack([np.hstack([truth, np.full((cell, 2), 255, np.uint8), matched])] * 3, -1)  # RGB
            border = (40, 200, 40) if correct else (220, 40, 40)
            pair[0, :, :] = border; pair[-1, :, :] = border; pair[:, 0, :] = border; pair[:, -1, :] = border
            rows.append(pair)
        # 2 columns of K/2
        half = (K + 1) // 2
        left = np.vstack(rows[:half]); right = np.vstack(rows[half:half*2] if len(rows) > half else rows[:half])
        if right.shape[0] < left.shape[0]:
            right = np.vstack([right, np.zeros((left.shape[0]-right.shape[0], right.shape[1], 3), np.uint8)])
        grid = np.hstack([left, np.full((left.shape[0], 4, 3), 255, np.uint8), right])
        os.makedirs(os.path.expanduser(args.out), exist_ok=True)
        outp = os.path.join(os.path.expanduser(args.out), f"match_viz_step{ck.get('step')}.png")
        Image.fromarray(grid).save(outp)
        print(f"VIZ saved: {outp}  (each pair = [held truth | model's matched patch], green=correct)", flush=True)
    except Exception as e:
        print(f"viz skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
