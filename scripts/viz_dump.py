#!/usr/bin/env python
"""Dump per-lesion inputs+outputs for the interactive molab viz (diagnose where DSC fails).

Best model (multisize ViT-base). Build tumor-anchored 32mm prisms over held-out patients; k-fold fine-tune
the seg decoder (warm-start seg-token/head + conv smoother, same as the eval), then on each TEST prism dump,
on a dense 1mm grid: the 4 modality intensities, GT class, predicted class, per-region DSC, and lesion size.
-> a single .npz the marimo notebook scrubs (case slider, slice slider, GT/pred/error overlays).
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from xmodal import data as D, model as M, sampling as S  # noqa: E402

MODS = ["t1", "t1c", "t2", "flair"]
REGIONS = {"ET": {3}, "TC": {1, 3}, "WT": {1, 2, 3}}


def sample_vol(scan_vol, affine_trans, affine_inv, pts_mm, dev):
    """nearest intensity/class at world-mm pts [K,3]."""
    vox = ((pts_mm - affine_trans) @ affine_inv.T).round().long()
    shp = torch.as_tensor(scan_vol.shape, device=dev)
    vox = vox.clamp(min=torch.zeros(3, device=dev, dtype=torch.long), max=shp - 1)
    return scan_vol[vox[..., 0], vox[..., 1], vox[..., 2]]


def dsc(pred, gt):
    inter = (pred & gt).sum().item(); s = pred.sum().item() + gt.sum().item()
    return -1.0 if s == 0 else 2.0 * inter / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="/tmp/viz_dump.npz")
    ap.add_argument("--data-root", default="/tmp/ho"); ap.add_argument("--tracks", nargs="+", default=["mets_ho"])
    ap.add_argument("--n-patients", type=int, default=16); ap.add_argument("--n-prisms", type=int, default=6)
    ap.add_argument("--n-src", type=int, default=96); ap.add_argument("--voxels", type=int, default=8)
    ap.add_argument("--size", type=float, default=4.0); ap.add_argument("--prism-mm", type=float, default=32.0)
    ap.add_argument("--res", type=float, default=1.0, help="viz grid spacing mm (crisp)")
    ap.add_argument("--qsize", type=float, default=2.0); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--unfreeze", type=int, default=12)
    ap.add_argument("--enc-width", type=int, default=768); ap.add_argument("--enc-heads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device; V = a.voxels; unit = S._cube_unit(V, dev); rng = np.random.default_rng(a.seed)

    class SmoothHead(torch.nn.Module):
        def __init__(self, w, n=4, h=128):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Conv3d(w, h, 3, padding=1), torch.nn.GELU(),
                                           torch.nn.Conv3d(h, h, 3, padding=1), torch.nn.GELU(), torch.nn.Conv3d(h, n, 1))

        def forward(self, x):
            return self.net(x)

    dirs = []
    for tr in a.tracks:
        dirs += sorted(glob.glob(os.path.join(os.path.expanduser(a.data_root), tr, "BraTS-*")))
    prisms = []; ctr = 0
    for d in dirs:
        if ctr >= a.n_patients:
            break
        pid = os.path.basename(d)
        if not glob.glob(f"{d}/{pid}-seg.nii.gz"):
            continue
        try:
            b = D.load_local_bundle(pid, d, device=dev, with_seg=True)[0]
        except Exception:
            continue
        sc0 = b["t1c"]; tmm = sc0.tumor_np
        if tmm is None or len(tmm) == 0:
            continue
        fgn = sc0.foreground_np if sc0.foreground_np is not None else sc0.foreground_mm.cpu().numpy()
        at, ai = sc0.affine_trans, sc0.affine_inv
        half = a.prism_mm / 2.0
        lin = np.arange(-half, half + 1e-3, a.res, dtype=np.float32); G = len(lin)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij"); grid = np.stack([gx, gy, gz], -1).reshape(-1, 3)
        for _ in range(a.n_prisms):
            anch = tmm[rng.integers(len(tmm))].astype(np.float32)
            gpts = (grid + anch).astype(np.float32)
            cm = fgn[np.abs(fgn - anch).max(-1) <= half]
            if len(cm) < a.n_src // 2:
                continue
            cs = cm[rng.integers(len(cm), size=a.n_src)].astype(np.float32)
            sm = rng.integers(4, size=a.n_src)
            sp = np.zeros((a.n_src, V, V, V), np.float16)
            for mi, m in enumerate(MODS):
                sel = np.nonzero(sm == mi)[0]
                if sel.size:
                    sp[sel] = S._gather_cubes(b[m], torch.as_tensor(cs[sel], device=dev),
                                              torch.full((sel.size,), a.size, device=dev), unit, dev).half().cpu().numpy()
            gp = torch.as_tensor(gpts, device=dev)
            gt = sample_vol(sc0.seg_vol, at, ai, gp, dev).long()
            gt = torch.where(gt == 4, torch.full_like(gt, 3), gt).cpu().numpy().astype(np.int8)   # ET 4->3
            imgs = np.stack([sample_vol(b[m].volume, b[m].affine_trans, b[m].affine_inv, gp, dev).float().cpu().numpy()
                             for m in MODS]).astype(np.float16)                # [4, G^3]
            prisms.append(dict(sp=sp, sc=cs - anch, sm=sm, gpts=gpts - anch, gt=gt, imgs=imgs, gdim=G, pid=ctr, pname=pid))
        ctr += 1; del b; torch.cuda.empty_cache()
    print(f"built {len(prisms)} prisms over {ctr} patients (res={a.res}mm, G={prisms[0]['gdim'] if prisms else '?'})", flush=True)
    assert prisms

    ck = torch.load(sorted(glob.glob(a.checkpoint))[-1], map_location=dev)
    E = M.Phase0Encoder(M.EncoderConfig(width=a.enc_width, depth=12, heads=a.enc_heads, n_series=8,
                                        patch_grid=(V, V, V))).to(dev)
    E.load_state_dict(ck["model"], strict=False)
    for p in E.parameters():
        p.requires_grad_(False)

    def slots(seg_tok, sp, sc, sm, qp):
        nb = sp.shape[0]; nreg = 2 + E.registers.shape[0]
        zs = torch.full((nb, sp.shape[1], 3), a.size, device=dev); zt = torch.full((nb, qp.shape[1], 3), a.qsize, device=dev)
        with torch.no_grad():
            x = E.encode(*E._context([E.embed(sp, zs, sm)], [sc], dev, nb)); ctx, cc = E._context([x[:, nreg:]], [sc], dev, nb)
        q = (E.query_seed[None, None, :] + E._size_emb(zt) + seg_tok[None, None, :]).contiguous()
        return E._decode(q, ctx, cc, qp)

    def grid_logits(head, sl, G):
        return head(sl.reshape(G, G, G, -1).permute(3, 0, 1, 2)[None])[0].permute(1, 2, 3, 0).reshape(-1, 4)

    pids = sorted(set(p["pid"] for p in prisms)); fold = {p: i % 5 for i, p in enumerate(pids)}
    dump = []
    for f in range(5):
        tr = [p for p in prisms if fold[p["pid"]] != f]; te = [p for p in prisms if fold[p["pid"]] == f]
        if not te:
            continue
        seg_tok = E.seg_query.detach().clone().to(dev).requires_grad_(True) if hasattr(E, "seg_query") \
            else torch.zeros(E.cfg.width, device=dev, requires_grad=True)
        dec = list(E.decoder)[-a.unfreeze:]
        lin = torch.nn.Linear(E.cfg.width, 4).to(dev)
        if hasattr(E, "seg_cls_head"):
            with torch.no_grad():
                lin.weight.copy_(E.seg_cls_head.weight); lin.bias.copy_(E.seg_cls_head.bias)
        params = list(lin.parameters()) + [seg_tok, E.query_seed]
        for blk in dec:
            for p in blk.parameters():
                p.requires_grad_(True); params.append(p)
        opt = torch.optim.Adam(params, lr=a.lr)
        for _ in range(a.epochs):                                              # stage-1 scattered
            for pr in [tr[i] for i in rng.permutation(len(tr))]:
                sub = rng.integers(len(pr["gt"]), size=384)
                sp = torch.as_tensor(pr["sp"][None], device=dev).float(); sc = torch.as_tensor(pr["sc"][None], device=dev).float()
                sm = torch.as_tensor(pr["sm"][None], device=dev).long(); qp = torch.as_tensor(pr["gpts"][sub][None], device=dev).float()
                y = torch.as_tensor(pr["gt"][sub], device=dev).long()
                cnt = torch.bincount(y, minlength=4).float(); w = len(y) / (4 * cnt.clamp(min=1))
                logit = lin(slots(seg_tok, sp, sc, sm, qp)).reshape(-1, 4)
                opt.zero_grad(); torch.nn.functional.cross_entropy(logit, y, weight=w).backward(); opt.step()
        seg_tok.requires_grad_(False); E.query_seed.requires_grad_(False)
        for m in [lin, *dec]:
            for p in m.parameters():
                p.requires_grad_(False)
        emb_tr = [_dense(E, slots, seg_tok, pr, dev).half().cpu() for pr in tr]
        conv = SmoothHead(E.cfg.width).to(dev); opt2 = torch.optim.Adam(conv.parameters(), lr=a.lr)
        for _ in range(60):                                                    # stage-2 conv on cached emb
            for i in rng.permutation(len(tr)):
                pr = tr[i]; G = pr["gdim"]; bb = min(11, G); o = rng.integers(0, G - bb + 1, size=3)
                ev = emb_tr[i].reshape(G, G, G, -1)[o[0]:o[0]+bb, o[1]:o[1]+bb, o[2]:o[2]+bb].to(dev).float()
                y = torch.as_tensor(pr["gt"].reshape(G, G, G)[o[0]:o[0]+bb, o[1]:o[1]+bb, o[2]:o[2]+bb].reshape(-1), device=dev).long()
                logit = conv(ev.permute(3, 0, 1, 2)[None])[0].permute(1, 2, 3, 0).reshape(-1, 4)
                cnt = torch.bincount(y, minlength=4).float(); w = len(y) / (4 * cnt.clamp(min=1))
                opt2.zero_grad(); torch.nn.functional.cross_entropy(logit, y, weight=w).backward(); opt2.step()
        with torch.no_grad():                                                  # predict + dump each test prism
            for pr in te:
                G = pr["gdim"]; sl = _dense(E, slots, seg_tok, pr, dev)
                pred = grid_logits(conv, sl, G).argmax(1).cpu().numpy().astype(np.int8)
                gt = pr["gt"]; d = {}
                for r, cls in REGIONS.items():
                    gm = np.isin(gt, list(cls)); pm = np.isin(pred, list(cls))
                    d[r] = dsc(torch.as_tensor(pm), torch.as_tensor(gm))
                dump.append(dict(pname=pr["pname"], gdim=G, imgs=pr["imgs"].reshape(4, G, G, G),
                                 gt=gt.reshape(G, G, G), pred=pred.reshape(G, G, G),
                                 et_vox=int((gt == 3).sum()), tc_vox=int(np.isin(gt, [1, 3]).sum()),
                                 dsc_et=d["ET"], dsc_tc=d["TC"], dsc_wt=d["WT"]))
        print(f"fold {f}: dumped {len(te)} prisms", flush=True)

    np.savez_compressed(a.out, cases=np.array(dump, dtype=object))
    et = [c["dsc_et"] for c in dump if c["dsc_et"] >= 0]
    print(f"wrote {len(dump)} cases -> {a.out} | ET DSC mean {np.mean(et):.3f} (n{len(et)})", flush=True)


def _dense(E, slots, seg_tok, pr, dev):
    sp = torch.as_tensor(pr["sp"][None], device=dev).float(); sc = torch.as_tensor(pr["sc"][None], device=dev).float()
    sm = torch.as_tensor(pr["sm"][None], device=dev).long(); coords = torch.as_tensor(pr["gpts"], device=dev).float()
    with torch.no_grad():
        sl = [slots(seg_tok, sp, sc, sm, coords[c0:c0 + 4096][None])[0] for c0 in range(0, coords.shape[0], 4096)]
    return torch.cat(sl)


if __name__ == "__main__":
    main()
