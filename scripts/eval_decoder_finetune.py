#!/usr/bin/env python
"""Decoder-head fine-tune readout: reuse the PRETRAINED matching decoder as a task classifier. For each
labeled patch we build a v5-style bag (source = the patient's 4-modality context patches, frozen-encoded)
and DECODE a query at the patch's position; a small class head on the decoded rep predicts the 3-way tumor
class. We fine-tune the DECODER + class head (encoder frozen) with GroupKFold-5 over patients -> macro-F1.
Tests whether the pretraining decoder itself learned transferable structure, vs a fresh probe on encoder
features. v1 defaults (documented): source = n_src random-foreground context patches across all 4 modalities;
each target queried at modality = t1c; encoder frozen; fine-tune decoder + query embeds + head.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from xmodal import data as D, model as M, sampling as S  # noqa: E402

MODS = ["t1", "t1c", "t2", "flair"]
CLASSES = [0, 2, 3]                                                    # non-tumor, edema, ET (necrosis dropped)


def label_positions(sc0, segt, centers, LS, V, dev):
    """3-way footprint label (priority ET>NCR>ED) for each center [K,3] world-mm; necrosis -> class 1 (dropped)."""
    unit = S._cube_unit(V, dev)
    phys = unit[None] * float(LS) + centers[:, None, None, None, :]
    vi = ((phys - sc0.affine_trans) @ sc0.affine_inv.T).round().long()
    shp = torch.as_tensor(sc0.volume.shape, device=dev)
    vi = vi.clamp(min=torch.zeros(3, device=dev, dtype=torch.long), max=shp - 1)
    sl = segt[vi[..., 0], vi[..., 1], vi[..., 2]].reshape(centers.shape[0], -1)
    tf = (sl > 0).float().mean(1).cpu().numpy()
    c1 = (sl == 1).float().mean(1).cpu().numpy(); c2 = (sl == 2).float().mean(1).cpu().numpy(); c3 = (sl == 3).float().mean(1).cpu().numpy()
    tv = c1 + c2 + c3 + 1e-9; lab = np.zeros(len(tf), int); tum = tf > 0.25; tau = 0.15
    et = tum & (c3 / tv >= tau); nc = tum & ~et & (c1 / tv >= tau); ed = tum & ~et & ~nc
    lab[ed] = 2; lab[nc] = 1; lab[et] = 3
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="name=globpath ... (latest per name)")
    ap.add_argument("--data-root", default="/tmp/ho"); ap.add_argument("--tracks", nargs="+", default=["mets_ho"])
    ap.add_argument("--n-patients", type=int, default=40)
    ap.add_argument("--k-tgt", type=int, default=300, help="labeled target patches sampled per patient")
    ap.add_argument("--n-src", type=int, default=96, help="source context patches per bag")
    ap.add_argument("--voxels", type=int, default=8); ap.add_argument("--size", type=float, default=8.0)
    ap.add_argument("--query-mod", type=int, default=1, help="modality index to query targets with (1=t1c)")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bag", type=int, default=64, help="target patches decoded per step")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device; V = a.voxels; unit = S._cube_unit(V, dev)

    # load held-out patients: per patient cache the labeled target patches (coords + labels) and the scans
    dirs = []
    for tr in a.tracks:
        dirs += sorted(glob.glob(os.path.join(os.path.expanduser(a.data_root), tr, "BraTS-*")))
    pats = []
    for d in dirs:
        if len(pats) >= a.n_patients:
            break
        pid = os.path.basename(d); segp = glob.glob(f"{d}/{pid}-seg.nii.gz")
        if not segp:
            continue
        try:
            b = D.load_local_bundle(pid, d, device=dev, with_seg=True)[0]
        except Exception:
            continue
        sc0 = b["t1c"]; segt = torch.as_tensor(np.asarray(nib.load(segp[0]).get_fdata()), dtype=torch.int16, device=dev)
        c = sc0.foreground_mm[torch.randint(sc0.foreground_mm.shape[0], (a.k_tgt,), device=dev)]
        lab = label_positions(sc0, segt, c, 4, V, dev)
        pats.append({"b": b, "sc0": sc0, "coords": c, "labels": lab})
        del segt
    print(f"loaded {len(pats)} patients", flush=True)
    assert pats

    def bag_context(pat, rng):
        """source bag = n_src random-foreground patches across the 4 modalities (frozen-encoded context)."""
        sc0 = pat["sc0"]; M4 = sc0.foreground_mm.shape[0]
        cs = sc0.foreground_mm[torch.randint(M4, (a.n_src,), device=dev)]
        sers = torch.as_tensor(rng.integers(4, size=a.n_src), device=dev)
        ps = torch.empty(a.n_src, V, V, V, device=dev)
        for mi, m in enumerate(MODS):
            sel = (sers == mi)
            if sel.any():
                ps[sel] = S._gather_cubes(pat["b"][m], cs[sel], torch.full((int(sel.sum()),), a.size, device=dev), unit, dev)
        z = torch.full((a.n_src, 3), a.size, device=dev)
        return ps[None], (cs - cs.mean(0))[None], z[None], sers[None], cs.mean(0)

    def decoded_reps(E, pat, tgt_idx, rng):
        """encode source (frozen) + decode the target queries -> decoded reps [k, W] for classification."""
        ps, cs, zs, sers, ctr = bag_context(pat, rng)
        nreg = 2 + E.registers.shape[0]
        with torch.no_grad():
            x = E.encode(*E._context([E.embed(ps, zs, sers)], [cs], dev, 1))
            ctx, cc = E._context([x[:, nreg:]], [cs], dev, 1)
        tc = pat["coords"][tgt_idx] - ctr; zt = torch.full((1, len(tgt_idx), 3), a.size, device=dev)
        modt = torch.full((1, len(tgt_idx)), a.query_mod, device=dev, dtype=torch.long)
        query = (E.query_seed[None, None, :] + E._size_emb(zt) + E.series_q_embed(modt)).contiguous()
        return E._decode(query, ctx, cc, tc[None])[0]                 # [k, W] (grad flows to decoder + query embeds)

    def run(E):
        gids = list(range(len(pats))); f1s = []
        # GroupKFold over patients (round-robin, matches eval_battery)
        foldp = {g: i % 5 for i, g in enumerate(gids)}
        allY, allP = [], []
        for f in range(5):
            tr = [g for g in gids if foldp[g] != f]; te = [g for g in gids if foldp[g] == f]
            head = torch.nn.Linear(E.cfg.width, len(CLASSES)).to(dev)
            dec = list(E.decoder.parameters()) + [E.query_seed] + list(E.series_q_embed.parameters())
            opt = torch.optim.Adam(list(head.parameters()) + dec, lr=a.lr)
            cmap = {c: i for i, c in enumerate(CLASSES)}
            rng = np.random.default_rng(a.seed)
            for ep in range(a.epochs):
                for g in tr:
                    yl = pats[g]["labels"]; keep = np.where(yl != 1)[0]
                    if len(keep) == 0:
                        continue
                    sub = keep[rng.permutation(len(keep))[:a.bag]]
                    y = torch.as_tensor([cmap[int(yl[i])] for i in sub], device=dev)
                    cnt = torch.bincount(y, minlength=len(CLASSES)).float(); w = (len(y) / (len(CLASSES) * cnt.clamp(min=1)))
                    logits = head(decoded_reps(E, pats[g], sub, rng))
                    opt.zero_grad(); torch.nn.functional.cross_entropy(logits, y, weight=w).backward(); opt.step()
            with torch.no_grad():
                for g in te:
                    yl = pats[g]["labels"]; keep = np.where(yl != 1)[0]
                    if len(keep) == 0:
                        continue
                    logits = head(decoded_reps(E, pats[g], keep, np.random.default_rng(1)))
                    allP += [CLASSES[i] for i in logits.argmax(1).cpu().numpy()]; allY += [int(yl[i]) for i in keep]
        allY = np.array(allY); allP = np.array(allP); f1 = []
        for c in CLASSES:
            tp = ((allP == c) & (allY == c)).sum(); fp = ((allP == c) & (allY != c)).sum(); fn = ((allP != c) & (allY == c)).sum()
            p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9); f1.append(2 * p * r / (p + r + 1e-9))
        return np.array(f1)

    for spec in a.checkpoints:
        nm, path = spec.split("=", 1); g = sorted(glob.glob(path))
        if not g:
            print(f"{nm} NO_CKPT {path}", flush=True); continue
        ck = torch.load(g[-1], map_location=dev); step = int(ck.get("step", -1))
        E = M.Phase0Encoder(M.EncoderConfig(width=384, depth=12, heads=6, n_series=8, patch_grid=(V, V, V))).to(dev)
        E.load_state_dict(ck["model"], strict=False)
        for p in E.parameters():
            p.requires_grad_(False)                                   # encoder frozen; decoder re-enabled in run()
        for p in list(E.decoder.parameters()) + list(E.series_q_embed.parameters()):
            p.requires_grad_(True)
        E.query_seed.requires_grad_(True)
        f1 = run(E)
        print(f"{nm} step{step} decoder-FT: per-class F1 {np.round(f1,3)} enh {f1[2]:.3f} macro {f1.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
