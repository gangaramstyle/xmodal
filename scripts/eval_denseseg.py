#!/usr/bin/env python
"""Dense-segmentation eval, tumor-prism-scoped (ORACLE detection). Given a tumor-containing prism, densely
query the fine-tuned seg decoder at every grid voxel -> per-voxel class -> DSC + NSD for ET/TC/WT vs GT.
Because we hand it the tumor prism, this isolates segmentation QUALITY from detection -> comparable to the
BraTS-METS lesion-wise DSC (~0.75 ET at SOTA). Encoder frozen; decoder fine-tuned on dense voxels.

Two-stage readout: (1) fine-tune decoder(last N) + seg-token + LINEAR head on scattered voxels; (2) freeze
that, cache each prism's dense slot embeddings, and train a 3D-CONV SmoothHead on top (spatial prior the
per-point cross-attn decoder lacks). Reports both heads.

READOUT TRAINING POOL: by default GroupKFold-5 within the held-out test set. With --n-train-patients>0 and
--train-tracks (e.g. mets_train), the readout instead trains on a large pool of PRETRAIN patients (their
lesions), with the held-out test IDs excluded by basename, and evaluates on the full held-out set.
Regions: ET={3}, TC={1,3}(NCR+ET), WT={1,2,3}. Query at --res mm (default 2mm) inside a --prism-mm box.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import nibabel as nib  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from xmodal import data as D, model as M, sampling as S  # noqa: E402

MODS = ["t1", "t1c", "t2", "flair"]
REGIONS = {"ET": {3}, "TC": {1, 3}, "WT": {1, 2, 3}}


class SmoothHead(torch.nn.Module):
    """3D-conv head over the regular dense-query grid. The cross-attention decoder predicts each voxel
    INDEPENDENTLY (queries attend only to the source, never to each other), so a lone flipped rim voxel
    has no way to be corrected by its neighbors. This adds the missing spatial prior: mix neighbor
    embeddings over the grid before classifying. Input [B,W,X,Y,Z] -> per-voxel logits [B,n_cls,X,Y,Z].
    Fully convolutional (same-padding) so it trains on sub-blocks and infers on the full grid."""

    def __init__(self, w, n_cls=4, hid=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(w, hid, 3, padding=1), torch.nn.GELU(),       # RF 3
            torch.nn.Conv3d(hid, hid, 3, padding=1), torch.nn.GELU(),     # RF 5 (=10mm @2mm stride)
            torch.nn.Conv3d(hid, n_cls, 1))

    def forward(self, vol):
        return self.net(vol)


def find_patient_dirs(root, tracks):
    """Patient dirs = any dir containing a *-seg.nii.gz, at ANY depth under root/track (layouts nest
    differently: /tmp/ho/mets_ho/BraTS-* is flat, mets_train/MICCAI-.../BraTS-MET-* is one deeper)."""
    dirs = []
    for tr in tracks:
        segs = glob.glob(os.path.join(os.path.expanduser(root), tr, "**", "*-seg.nii.gz"), recursive=True)
        dirs += sorted({os.path.dirname(s) for s in segs})
    return dirs


def seg_at(scan, pts_mm, dev):
    """nearest seg class at world-mm points [K,3]."""
    vox = ((pts_mm - scan.affine_trans) @ scan.affine_inv.T).round().long()
    shp = torch.as_tensor(scan.seg_vol.shape, device=dev)
    vox = vox.clamp(min=torch.zeros(3, device=dev, dtype=torch.long), max=shp - 1)
    cls = scan.seg_vol[vox[..., 0], vox[..., 1], vox[..., 2]].long()            # [K]
    return torch.where(cls == 4, torch.full_like(cls, 3), cls)                  # old BraTS label 4 (ET) -> 3 (mets_train uses 4)


def dsc(pred, gt):
    inter = (pred & gt).sum().item(); s = pred.sum().item() + gt.sum().item()
    return None if s == 0 else 2.0 * inter / s                                  # None = region absent in both (no penalty)


def nsd(pred, gt, coords, tol):
    """Normalized surface distance @ tol mm (brute force over boundary points; prism-sized so cheap)."""
    if pred.sum() == 0 or gt.sum() == 0:
        return None
    pc, gc = coords[pred], coords[gt]
    if len(pc) == 0 or len(gc) == 0:
        return None
    d_pg = torch.cdist(pc, gc).min(1).values; d_gp = torch.cdist(gc, pc).min(1).values
    return float(((d_pg <= tol).float().sum() + (d_gp <= tol).float().sum()) / (len(pc) + len(gc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data-root", default="/tmp/ho"); ap.add_argument("--tracks", nargs="+", default=["mets_ho"])
    ap.add_argument("--n-patients", type=int, default=51, help="held-out TEST patients"); ap.add_argument("--n-prisms", type=int, default=8)
    # --- readout TRAIN pool: large pretrain-patient set (test IDs excluded). Empty -> GroupKFold within test. ---
    ap.add_argument("--train-root", default="~/xmodal/data/brats26")
    ap.add_argument("--train-tracks", nargs="+", default=[], help="tracks for readout TRAIN pool, e.g. mets_train")
    ap.add_argument("--n-train-patients", type=int, default=0, help=">0 -> train readout on this many pretrain patients (test IDs excluded); 0 -> 5-fold within test")
    ap.add_argument("--n-train-prisms", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.0, help=">0: single patient split (train on 1-val_frac of held-out, test on val_frac) instead of GroupKFold-5. train/test prisms from the SAME build, split by patient.")
    ap.add_argument("--n-src", type=int, default=96); ap.add_argument("--voxels", type=int, default=8)
    ap.add_argument("--size", type=float, default=4.0, help="source patch mm (uniform, unless --src-sizes)"); ap.add_argument("--prism-mm", type=float, default=32.0)
    ap.add_argument("--src-sizes", default=None, help="mixed-scale source: comma sizes e.g. '2,3,4' — each source patch a random one (multi-scale context, matches multi-size training). None=uniform --size.")
    ap.add_argument("--src-mult", type=int, default=1, help="oversample source context by this factor (heavy multi-scale inference context, e.g. 3x the ~96 seen in training)")
    ap.add_argument("--res", type=float, default=2.0, help="dense query grid STRIDE mm (metric resolution)")
    ap.add_argument("--qsize", type=float, default=4.0, help="query SIZE embedding mm (in-distribution w/ pretrain patch size, independent of stride)")
    ap.add_argument("--tol", type=float, default=2.0, help="NSD tolerance mm")
    ap.add_argument("--epochs", type=int, default=30, help="stage-1 decoder fine-tune epochs"); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs2", type=int, default=60, help="stage-2 conv-smoothing epochs (on cached embeddings, cheap)")
    ap.add_argument("--tgt-train", type=int, default=384, help="voxels/prism for stage-1 (linear) fine-tune"); ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--head", choices=["conv", "linear"], default="conv", help="conv=3D smoothing head (spatial prior); linear=stage-1 only")
    ap.add_argument("--block", type=int, default=11, help="contiguous grid sub-block side for conv-head training (voxels)")
    ap.add_argument("--warmstart-seg", action="store_true", help="init seg-token+head from the ckpt's baked-in seg_query/seg_cls_head (in-distribution eval of seg-task arms; no-op for non-seg arms whose seg_query is untrained)")
    ap.add_argument("--unfreeze", type=int, default=12); ap.add_argument("--enc-width", type=int, default=384); ap.add_argument("--enc-heads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device; V = a.voxels; unit = S._cube_unit(V, dev); rng = np.random.default_rng(a.seed)
    src_sizes = np.array([float(x) for x in a.src_sizes.split(",")]) if a.src_sizes else None

    def build_prisms(dirs, npat, nprism, exclude, label):
        """Build tumor-anchored dense prisms from up to npat patients in dirs (skip exclude ids / no-seg / no-tumor)."""
        out = []; ctr = 0
        for d in dirs:
            if ctr >= npat:
                break
            pid = os.path.basename(d)
            if exclude and pid in exclude:
                continue
            segp = glob.glob(f"{d}/{pid}-seg.nii.gz")
            if not segp:
                continue
            try:
                b = D.load_local_bundle(pid, d, device=dev, with_seg=True)[0]
            except Exception:
                continue
            sc0 = b["t1c"]; tmm = sc0.tumor_np
            if tmm is None or len(tmm) == 0:
                continue
            fgn = sc0.foreground_np if sc0.foreground_np is not None else sc0.foreground_mm.cpu().numpy()
            half = a.prism_mm / 2.0
            lin = np.arange(-half, half + 1e-3, a.res, dtype=np.float32)         # dense grid offsets (mm)
            gdim = len(lin)                                                      # grid side (reshape [G^3,.]->[G,G,G,.])
            gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij"); grid = np.stack([gx, gy, gz], -1).reshape(-1, 3)
            for _ in range(nprism):
                anch = tmm[rng.integers(len(tmm))].astype(np.float32)            # anchor ON a tumor voxel
                gpts = (grid + anch).astype(np.float32)                         # dense query voxels (world-mm)
                cm = fgn[np.abs(fgn - anch).max(-1) <= half]                    # foreground INSIDE the prism
                if len(cm) < a.n_src // 2:
                    continue
                nsrc = a.n_src * a.src_mult                                     # heavy multi-scale context = more patches
                cs = cm[rng.integers(len(cm), size=nsrc)].astype(np.float32)     # source context sampled inside the prism
                sm = rng.integers(4, size=nsrc)
                ssz = (rng.choice(src_sizes, size=nsrc) if src_sizes is not None  # per-patch scale (mixed) or uniform --size
                       else np.full(nsrc, a.size)).astype(np.float32)
                sp = np.zeros((nsrc, V, V, V), np.float16)
                for mi, m in enumerate(MODS):
                    sel = np.nonzero(sm == mi)[0]
                    if sel.size:
                        sp[sel] = S._gather_cubes(b[m], torch.as_tensor(cs[sel], device=dev),
                                                  torch.as_tensor(ssz[sel], device=dev), unit, dev).half().cpu().numpy()
                gt = seg_at(sc0, torch.as_tensor(gpts, device=dev), dev).cpu().numpy()   # class per query voxel
                out.append(dict(sp=sp, sc=cs - anch, sm=sm, ssz=ssz, gpts=gpts - anch, gt=gt, gdim=gdim, pid=ctr))
            ctr += 1; del b; torch.cuda.empty_cache()
        print(f"built {len(out)} {label} prisms over {ctr} patients (res={a.res}mm grid, head={a.head})", flush=True)
        return out

    # ---- test pool (held-out) + optional large pretrain train pool ----
    test_dirs = find_patient_dirs(a.data_root, a.tracks)
    test_ids = {os.path.basename(d) for d in test_dirs}
    test_prisms = build_prisms(test_dirs, a.n_patients, a.n_prisms, None, "TEST")
    assert test_prisms
    train_prisms = None
    if a.n_train_patients > 0 and a.train_tracks:
        tr_dirs = find_patient_dirs(a.train_root, a.train_tracks)
        rng.shuffle(tr_dirs)                                                    # representative sample across the pool
        n_excl = sum(1 for d in tr_dirs if os.path.basename(d) in test_ids)
        print(f"train pool {a.train_tracks}: {len(tr_dirs)} dirs, excluding {n_excl} held-out test IDs", flush=True)
        train_prisms = build_prisms(tr_dirs, a.n_train_patients, a.n_train_prisms, test_ids, "TRAIN")
        assert train_prisms

    def slots(E, seg_tok, sp, sc, sm, qpts, ssz=None):                          # frozen-enc source -> decode dense queries
        nb = sp.shape[0]; nreg = 2 + E.registers.shape[0]
        zs = ssz[..., None].expand(-1, -1, 3) if ssz is not None else torch.full((nb, sp.shape[1], 3), a.size, device=dev)
        zt = torch.full((nb, qpts.shape[1], 3), a.qsize, device=dev)
        with torch.no_grad():
            x = E.encode(*E._context([E.embed(sp, zs, sm)], [sc], dev, nb)); ctx, cc = E._context([x[:, nreg:]], [sc], dev, nb)
        q = (E.query_seed[None, None, :] + E._size_emb(zt) + seg_tok[None, None, :]).contiguous()
        return E._decode(q, ctx, cc, qpts)

    def dense_slots(E, seg_tok, pr):                                            # decode EVERY grid voxel -> [G^3, W]
        sp = torch.as_tensor(pr["sp"][None], device=dev).float(); sc = torch.as_tensor(pr["sc"][None], device=dev).float()
        sm = torch.as_tensor(pr["sm"][None], device=dev).long(); coords = torch.as_tensor(pr["gpts"], device=dev).float()
        ssz = torch.as_tensor(pr["ssz"][None], device=dev).float()
        with torch.no_grad():
            sl = [slots(E, seg_tok, sp, sc, sm, coords[c0:c0 + 4096][None], ssz)[0] for c0 in range(0, coords.shape[0], 4096)]
        return torch.cat(sl)                                                    # [G^3, W]

    def grid_logits(head, sl, G):                                              # [G^3,W] -> conv over volume -> [G^3,4]
        vol = sl.reshape(G, G, G, -1).permute(3, 0, 1, 2)[None]                # [1,W,G,G,G]
        return head(vol)[0].permute(1, 2, 3, 0).reshape(-1, 4)                 # [G^3,4], voxel order matches gt flat

    def score(preds):                                                          # [(pc, gtc, coords)] -> {region:(DSC,NSD,n)}
        agg = {r: [] for r in REGIONS}; aggn = {r: [] for r in REGIONS}
        for pc, gtc, coords in preds:
            for r, cls in REGIONS.items():
                gm = torch.zeros_like(gtc, dtype=torch.bool); pm = torch.zeros_like(pc, dtype=torch.bool)
                for k in cls:
                    gm |= (gtc == k); pm |= (pc == k)
                dv = dsc(pm, gm)
                if dv is not None:
                    agg[r].append(dv); nv = nsd(pm, gm, coords, a.tol)
                    if nv is not None:
                        aggn[r].append(nv)
        return {r: (float(np.mean(agg[r])) if agg[r] else 0, float(np.mean(aggn[r])) if aggn[r] else 0, len(agg[r])) for r in REGIONS}

    def fit_eval(E, tr, te):
        """Stage-1 (decoder+seg_tok+linear on scattered voxels) -> freeze+cache embeddings -> stage-2 conv.
        Returns (linear_preds, conv_preds) on te as lists of (pred_class, gt_class, coords)."""
        if a.warmstart_seg and hasattr(E, "seg_query"):                        # give seg-task arms their baked-in head
            seg_tok = E.seg_query.detach().clone().to(dev).requires_grad_(True)
        else:
            seg_tok = torch.zeros(E.cfg.width, device=dev, requires_grad=True)
        lin = torch.nn.Linear(E.cfg.width, 4).to(dev)
        if a.warmstart_seg and hasattr(E, "seg_cls_head"):                     # seg_cls_head is Linear(width,4) -> direct copy
            with torch.no_grad():
                lin.weight.copy_(E.seg_cls_head.weight); lin.bias.copy_(E.seg_cls_head.bias)
        dec_blocks = list(E.decoder)[-a.unfreeze:]
        params = list(lin.parameters()) + [seg_tok, E.query_seed]
        for blk in dec_blocks:
            for p in blk.parameters():
                p.requires_grad_(True); params.append(p)
        opt = torch.optim.Adam(params, lr=a.lr)
        for _ in range(a.epochs):                                              # ---- Stage 1 ----
            for pr in [tr[i] for i in rng.permutation(len(tr))]:
                sub = rng.integers(len(pr["gt"]), size=a.tgt_train)
                sp = torch.as_tensor(pr["sp"][None], device=dev).float(); sc = torch.as_tensor(pr["sc"][None], device=dev).float()
                sm = torch.as_tensor(pr["sm"][None], device=dev).long(); qp = torch.as_tensor(pr["gpts"][sub][None], device=dev).float()
                ssz = torch.as_tensor(pr["ssz"][None], device=dev).float()
                y = torch.as_tensor(pr["gt"][sub], device=dev).long()
                cnt = torch.bincount(y, minlength=4).float(); w = len(y) / (4 * cnt.clamp(min=1))
                logit = lin(slots(E, seg_tok, sp, sc, sm, qp, ssz)).reshape(-1, 4)
                opt.zero_grad(); torch.nn.functional.cross_entropy(logit, y, weight=w).backward(); opt.step()
        seg_tok.requires_grad_(False); E.query_seed.requires_grad_(False)      # freeze all of stage 1
        for m in [lin, *dec_blocks]:
            for p in m.parameters():
                p.requires_grad_(False)
        emb_tr = [dense_slots(E, seg_tok, pr).half().cpu() for pr in tr]        # train cache on CPU (pool can be large)
        emb_te = [dense_slots(E, seg_tok, pr).half() for pr in te]             # test cache on GPU
        conv = None
        if a.head == "conv":                                                   # ---- Stage 2 ----
            conv = SmoothHead(E.cfg.width).to(dev); opt2 = torch.optim.Adam(conv.parameters(), lr=a.lr)
            for _ in range(a.epochs2):
                for i in rng.permutation(len(tr)):
                    pr = tr[i]; G = pr["gdim"]; b = min(a.block, G); o = rng.integers(0, G - b + 1, size=3)
                    ev = emb_tr[i].reshape(G, G, G, -1)[o[0]:o[0] + b, o[1]:o[1] + b, o[2]:o[2] + b].to(dev).float()  # [b,b,b,W]
                    y = torch.as_tensor(pr["gt"].reshape(G, G, G)[o[0]:o[0] + b, o[1]:o[1] + b, o[2]:o[2] + b].reshape(-1), device=dev).long()
                    logit = conv(ev.permute(3, 0, 1, 2)[None])[0].permute(1, 2, 3, 0).reshape(-1, 4)
                    cnt = torch.bincount(y, minlength=4).float(); w = len(y) / (4 * cnt.clamp(min=1))
                    opt2.zero_grad(); torch.nn.functional.cross_entropy(logit, y, weight=w).backward(); opt2.step()
        lp = []; cp = []                                                       # ---- eval both heads ----
        with torch.no_grad():
            for j, pr in enumerate(te):
                e = emb_te[j].float(); G = pr["gdim"]
                gtc = torch.as_tensor(pr["gt"], device=dev).long(); coords = torch.as_tensor(pr["gpts"], device=dev).float()
                lp.append((lin(e).argmax(1), gtc, coords))
                if conv is not None:
                    cp.append((grid_logits(conv, e, G).argmax(1), gtc, coords))
        return lp, cp

    def run(E):
        if train_prisms is not None:                                           # pretrain-pool mode: train->pool, test->held-out
            lp, cp = fit_eval(E, train_prisms, test_prisms)
        elif a.val_frac > 0:                                                   # single 80/20 patient split within the held-out
            pids = sorted(set(p["pid"] for p in test_prisms))
            order = np.random.default_rng(a.seed).permutation(len(pids))
            nte = max(1, int(round(len(pids) * a.val_frac))); te_pids = {pids[i] for i in order[:nte]}
            tr = [p for p in test_prisms if p["pid"] not in te_pids]; te = [p for p in test_prisms if p["pid"] in te_pids]
            print(f"single split: train {len(pids)-nte} patients / test {nte} patients (val_frac={a.val_frac})", flush=True)
            lp, cp = fit_eval(E, tr, te)
        else:                                                                  # GroupKFold-5 within the held-out test set
            pids = sorted(set(p["pid"] for p in test_prisms)); fold = {p: i % 5 for i, p in enumerate(pids)}
            lp = []; cp = []
            for f in range(5):
                tr = [p for p in test_prisms if fold[p["pid"]] != f]; te = [p for p in test_prisms if fold[p["pid"]] == f]
                a2, b2 = fit_eval(E, tr, te); lp += a2; cp += b2
        return score(lp), (score(cp) if cp else None)

    for spec in a.checkpoints:
        nm, path = spec.split("=", 1); g = sorted(glob.glob(path))
        if not g:
            print(f"{nm} NO_CKPT {path}", flush=True); continue
        ck = torch.load(g[-1], map_location=dev); step = int(ck.get("step", -1))
        E = M.Phase0Encoder(M.EncoderConfig(width=a.enc_width, depth=12, heads=a.enc_heads, n_series=8,
                                            patch_grid=(V, V, V))).to(dev)
        E.load_state_dict(ck["model"], strict=False)
        for p in E.parameters():
            p.requires_grad_(False)
        rl, rc = run(E)
        pool = f"train{len(train_prisms)}p" if train_prisms is not None else "kfold5"
        ws = "+warmstart" if a.warmstart_seg else ""
        src = f"src{a.src_sizes}x{a.src_mult}" if a.src_sizes else f"src{a.size}mm"
        tag = f"{nm} step{step} denseseg[{pool}{ws}] {src}/qsize{a.qsize}mm/stride{a.res}mm"
        print(f"{tag} [stage1 linear]: " + " | ".join(f"{k} DSC {v[0]:.3f} NSD {v[1]:.3f}(n{v[2]})" for k, v in rl.items()), flush=True)
        if rc is not None:
            print(f"{tag} [stage2 conv-smooth]: " + " | ".join(f"{k} DSC {v[0]:.3f} NSD {v[1]:.3f}(n{v[2]})" for k, v in rc.items()), flush=True)


if __name__ == "__main__":
    main()
