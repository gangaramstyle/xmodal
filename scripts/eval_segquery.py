#!/usr/bin/env python
"""Seg-query decoder readout (the most in-distribution eval): treat SEGMENTATION as a 5th 'modality'.
Reuse the pretrained encoder+decoder exactly as trained -- a multi-modal SOURCE bag in a prism is encoded
(encoder FROZEN), then the decoder cross-attends a query POSITION and, instead of predicting a held
modality, a learnable SEG token + class head predict the center's tumor class. Ablate how much of the
decoder is unfrozen (--unfreeze N last blocks; 0 = frozen decoder, only seg-token+head train). GroupKFold-5
over patients -> macro-F1. 4mm patches @ 32mm prism (matches how 4mm trained). Supersedes the broken
decoder-FT v1 (that queried t1c + random context; this uses a dedicated seg query + prism source).
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
    """3-way footprint label (priority ET>NCR>ED) for centers [K,3] world-mm; NCR->1 (dropped)."""
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
    ap.add_argument("--checkpoints", nargs="+", required=True, help="name=globpath ...")
    ap.add_argument("--data-root", default="/tmp/ho"); ap.add_argument("--tracks", nargs="+", default=["mets_ho"])
    ap.add_argument("--n-patients", type=int, default=40); ap.add_argument("--n-prisms", type=int, default=16)
    ap.add_argument("--n-src", type=int, default=96); ap.add_argument("--n-tgt", type=int, default=64)
    ap.add_argument("--voxels", type=int, default=8); ap.add_argument("--size", type=float, default=4.0)
    ap.add_argument("--prism-mm", type=float, default=32.0); ap.add_argument("--prism-tumor", type=float, default=0.7)
    ap.add_argument("--unfreeze", type=int, default=0, help="last N decoder blocks to unfreeze (0=frozen decoder)")
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chunk", type=int, default=16, help="bags per fwd/bwd")
    ap.add_argument("--enc-width", type=int, default=384); ap.add_argument("--enc-heads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device; V = a.voxels; unit = S._cube_unit(V, dev); rng = np.random.default_rng(a.seed)

    # ---- build prism bags (source = all-modality context; targets = labeled positions), grouped by patient ----
    dirs = []
    for tr in a.tracks:
        dirs += sorted(glob.glob(os.path.join(os.path.expanduser(a.data_root), tr, "BraTS-*")))
    bags = []                                                          # each: dict with src/tgt arrays + patient id
    pid_ctr = 0
    for d in dirs:
        if pid_ctr >= a.n_patients:
            break
        pid = os.path.basename(d); segp = glob.glob(f"{d}/{pid}-seg.nii.gz")
        if not segp:
            continue
        try:
            b = D.load_local_bundle(pid, d, device=dev, with_seg=True)[0]
        except Exception:
            continue
        sc0 = b["t1c"]; fgn = sc0.foreground_np if sc0.foreground_np is not None else sc0.foreground_mm.cpu().numpy()
        tmm = sc0.tumor_np
        if tmm is None:
            continue
        segt = torch.as_tensor(np.asarray(nib.load(segp[0]).get_fdata()), dtype=torch.int16, device=dev)
        half = a.prism_mm / 2.0; made = 0
        for _ in range(a.n_prisms * 3):
            if made >= a.n_prisms:
                break
            anch = (tmm[rng.integers(len(tmm))] + (rng.random(3) * 2 - 1) * half) if rng.random() < a.prism_tumor \
                else fgn[rng.integers(len(fgn))]
            loc = fgn[np.abs(fgn - anch).max(-1) <= half]
            if len(loc) < a.n_src:
                continue
            src_c = loc[rng.integers(len(loc), size=a.n_src)].astype(np.float32)      # source context centers
            src_m = rng.integers(4, size=a.n_src)                                     # random modality per source patch
            tgt_c = loc[rng.integers(len(loc), size=a.n_tgt)].astype(np.float32)      # query/target positions
            lab = label_positions(sc0, segt, torch.as_tensor(tgt_c, device=dev), a.size, V, dev)
            # gather source cubes per modality (CPU numpy for the bag store)
            sp = np.zeros((a.n_src, V, V, V), np.float16)
            for mi, m in enumerate(MODS):
                sel = np.where(src_m == mi)[0]
                if sel.size:
                    sp[sel] = S._gather_cubes(b[m], torch.as_tensor(src_c[sel], device=dev),
                                              torch.full((sel.size,), a.size, device=dev), unit, dev).half().cpu().numpy()
            bags.append(dict(sp=sp, sc=(src_c - anch.astype(np.float32)), sm=src_m,
                             tc=(tgt_c - anch.astype(np.float32)), lab=lab, pid=pid_ctr))
            made += 1
        pid_ctr += 1; del segt; torch.cuda.empty_cache()
    print(f"built {len(bags)} prism bags over {pid_ctr} patients (src={a.n_src} tgt={a.n_tgt} @ {a.size}mm/{a.prism_mm}mm prism)", flush=True)
    assert bags
    cmap = {c: i for i, c in enumerate(CLASSES)}

    def slots_for(E, seg_tok, chunk_bags):                            # forward: frozen-enc source -> decode seg-queries
        nb = len(chunk_bags); nreg = 2 + E.registers.shape[0]
        sp = torch.as_tensor(np.stack([g["sp"] for g in chunk_bags]), device=dev).float()      # [nb,nS,V,V,V]
        sc = torch.as_tensor(np.stack([g["sc"] for g in chunk_bags]), device=dev).float()
        sm = torch.as_tensor(np.stack([g["sm"] for g in chunk_bags]), device=dev).long()
        tc = torch.as_tensor(np.stack([g["tc"] for g in chunk_bags]), device=dev).float()
        zs = torch.full((nb, a.n_src, 3), a.size, device=dev); zt = torch.full((nb, a.n_tgt, 3), a.size, device=dev)
        with torch.no_grad():                                         # encoder frozen
            x = E.encode(*E._context([E.embed(sp, zs, sm)], [sc], dev, nb))
            ctx, cc = E._context([x[:, nreg:]], [sc], dev, nb)
        query = (E.query_seed[None, None, :] + E._size_emb(zt) + seg_tok[None, None, :]).contiguous()  # SEG query
        return E._decode(query, ctx, cc, tc)                          # [nb, n_tgt, W] (grad -> decoder(unfrozen)+seg_tok)

    def run(E):
        pids = sorted(set(g["pid"] for g in bags)); fold_of = {p: i % 5 for i, p in enumerate(pids)}
        allY, allP = [], []
        for f in range(5):
            tr = [g for g in bags if fold_of[g["pid"]] != f]; te = [g for g in bags if fold_of[g["pid"]] == f]
            seg_tok = torch.zeros(E.cfg.width, device=dev, requires_grad=True)
            head = torch.nn.Linear(E.cfg.width, len(CLASSES)).to(dev)
            params = list(head.parameters()) + [seg_tok] + [E.query_seed]
            if a.unfreeze > 0:
                for blk in list(E.decoder)[-a.unfreeze:]:
                    for p in blk.parameters():
                        p.requires_grad_(True); params.append(p)
            opt = torch.optim.Adam(params, lr=a.lr)
            for _ in range(a.epochs):
                order = rng.permutation(len(tr))
                for c0 in range(0, len(tr), a.chunk):
                    cb = [tr[i] for i in order[c0:c0 + a.chunk]]
                    y = torch.as_tensor(np.concatenate([[cmap[int(v)] for v in g["lab"]] for g in cb]), device=dev)
                    cnt = torch.bincount(y, minlength=len(CLASSES)).float(); w = len(y) / (len(CLASSES) * cnt.clamp(min=1))
                    logits = head(slots_for(E, seg_tok, cb)).reshape(-1, len(CLASSES))
                    opt.zero_grad(); torch.nn.functional.cross_entropy(logits, y, weight=w).backward(); opt.step()
            with torch.no_grad():
                for c0 in range(0, len(te), a.chunk):
                    cb = te[c0:c0 + a.chunk]
                    pr = head(slots_for(E, seg_tok, cb)).reshape(-1, len(CLASSES)).argmax(1).cpu().numpy()
                    allP += [CLASSES[i] for i in pr]; allY += [int(v) for g in cb for v in g["lab"]]
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
        E = M.Phase0Encoder(M.EncoderConfig(width=a.enc_width, depth=12, heads=a.enc_heads, n_series=8,
                                            patch_grid=(V, V, V))).to(dev)
        E.load_state_dict(ck["model"], strict=False)
        for p in E.parameters():
            p.requires_grad_(False)                                   # encoder frozen; decoder/query re-enabled in run()
        f1 = run(E)
        print(f"{nm} step{step} segquery unfreeze{a.unfreeze}: per-class F1 {np.round(f1,3)} enh {f1[2]:.3f} macro {f1.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
