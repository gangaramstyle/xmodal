#!/usr/bin/env python
"""Cross-modal reconstruction GIF (standalone port of the notebook _cross_gif cell).

For a held-out patient with a focal enhancing met, scroll 1mm/frame through the lesion and show:
  top:    full native-plane t1c | t1n slices (radiologist context; green box = model's patch FOV)
  bottom: the model's patch grid -> GT t1c | cross-MAE prediction (blue=predicted) | ordering jigsaw
          (each masked slot filled with the matcher's assigned patch; green=correct, red=wrong)
Predict target modality (t1c) from source (t1n) via the cross decoder. Use a checkpoint whose PIXEL
decoder is well-trained (end of the cross phase, e.g. native@50k) — the latent phase degrades it.

Run on HELD-OUT data to judge overfitting vs representation quality: sharp generalizing reconstruction
=> representation is fine; blurry/wrong on held-out => genuinely weak (or overfit if train >> held-out).
"""
import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xmodal import data as D, model as M, sampling as S  # noqa: E402
from xmodal.matching import blur_contents as _blur  # noqa: E402


def u8(a):
    return (np.clip(np.asarray(a, np.float32), 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="/tmp/heldout")
    ap.add_argument("--tracks", nargs="+", default=["mets_ho"])
    ap.add_argument("--src", default="t1")
    ap.add_argument("--tgt", default="t1c")
    ap.add_argument("--out", default="/tmp/cross_recon.gif")
    ap.add_argument("--target-vox", type=int, default=2000, help="pick the enhancing met closest to this size")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    ck = torch.load(a.checkpoint, map_location=dev)
    E = M.Phase0Encoder(M.EncoderConfig(width=384, depth=12, heads=6, n_series=8)).to(dev)
    E.load_state_dict(ck["model"]); E.eval(); step = int(ck.get("step", -1))

    dirs = []
    for tr in a.tracks:
        dirs += sorted(glob.glob(os.path.join(os.path.expanduser(a.data_root), tr, "BraTS-*")))
    best = None
    for i, d in enumerate(dirs):
        pid = os.path.basename(d)
        sp = f"{d}/{pid}-seg.nii.gz"
        if not os.path.exists(sp):
            continue
        seg = np.asarray(nib.load(sp).get_fdata()).astype(int)
        lid, nl = ndi.label(seg == 3)
        for c in range(1, nl + 1):
            szc = int((lid == c).sum())
            if szc < 200:
                continue
            sc_ = abs(szc - a.target_vox)
            if best is None or sc_ < best[0]:
                best = (sc_, d, lid == c, szc)
    if best is None:
        print("no focal enhancing met found"); return
    _, bd, mask, csz = best
    pid = os.path.basename(bd)
    b = D.load_local_bundle(pid, bd, device=dev)[0]
    sc = b[a.tgt]
    ax = int(np.argmax(np.abs(np.array(sc.volume.shape, float) - np.median(sc.volume.shape))))
    inpl = [x for x in range(3) if x != ax]
    ctrd = np.argwhere(mask).mean(0).astype(np.float32)
    anchor_mm = (np.linalg.inv(sc.affine_inv.cpu().numpy()) @ ctrd) + sc.affine_trans.cpu().numpy()
    av = (sc.affine_inv.cpu().numpy() @ (anchor_mm - sc.affine_trans.cpu().numpy()))
    volc = sc.volume.cpu().numpy(); voln = b[a.src].volume.cpu().numpy()

    G, pm, p, NF, zstep = 6, 8.0, 16, 28, 1.0
    spacing = pm * 15 / 16; hw = int(round(G * spacing / 2))
    lin = (np.arange(G) - (G - 1) / 2) * spacing
    gi, gj = np.meshgrid(lin, lin, indexing="ij")
    baseoff = np.zeros((G * G, 3), np.float32); baseoff[:, inpl[0]] = gi.ravel(); baseoff[:, inpl[1]] = gj.ravel()
    unit = S.slab_unit_offsets(ax, 16, dev); N = G * G
    sz = torch.full((1, N), pm, device=dev); sz3 = S.size_to_extent(sz, ax)
    rng = np.random.default_rng(0); perm = rng.permutation(N); na = max(2, int(round(N * 0.10)))
    anc = np.sort(perm[:na]); rec = np.sort(perm[na:]); anct = torch.as_tensor(anc, device=dev); rect = torch.as_tensor(rec, device=dev)
    rset = {int(m): k for k, m in enumerate(rec)}

    def pgrid(vals, bfn=None):
        can = np.zeros((G * p, G * p, 3), np.uint8)
        for n in range(N):
            i, j = n // G, n % G; can[i * p:(i + 1) * p, j * p:(j + 1) * p] = np.stack([vals[n]] * 3, -1)
            if bfn is not None:
                bc = bfn(n)
                if bc is not None:
                    can[i * p, j * p:(j + 1) * p] = bc; can[(i + 1) * p - 1, j * p:(j + 1) * p] = bc
                    can[i * p:(i + 1) * p, j * p] = bc; can[i * p:(i + 1) * p, (j + 1) * p - 1] = bc
        return can

    def full(vol, z):
        sl = np.take(vol, int(np.clip(z, 0, vol.shape[ax] - 1)), axis=ax); img = np.stack([u8(sl)] * 3, -1)
        H, W = img.shape[:2]; g = np.array([40, 220, 90], np.uint8); th = max(2, H // 140)   # thick enough to survive resize
        y0, y1 = int(np.clip(av[inpl[0]] - hw, 0, H - 1)), int(np.clip(av[inpl[0]] + hw, 0, H - 1))
        x0, x1 = int(np.clip(av[inpl[1]] - hw, 0, W - 1)), int(np.clip(av[inpl[1]] + hw, 0, W - 1))
        for d in range(th):
            img[np.clip(y0 + d, 0, H - 1), x0:x1 + 1] = g; img[np.clip(y1 - d, 0, H - 1), x0:x1 + 1] = g
            img[y0:y1 + 1, np.clip(x0 + d, 0, W - 1)] = g; img[y0:y1 + 1, np.clip(x1 - d, 0, W - 1)] = g
        return img

    frames = []
    for f in range(NF):
        zoff = (f - NF / 2) * zstep; shf = np.zeros(3, np.float32); shf[ax] = zoff; zc = int(round(av[ax] + zoff))
        ctr = torch.as_tensor(anchor_mm + shf + baseoff, device=dev, dtype=torch.float32)
        co = torch.as_tensor(baseoff, device=dev, dtype=torch.float32)[None]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            src = S.sample_patches_group(b[a.src].volume, S.mixed_bag_vox(b[a.src], ctr[None], sz, unit))
            tgt = S.sample_patches_group(b[a.tgt].volume, S.mixed_bag_vox(b[a.tgt], ctr[None], sz, unit))
            ss, _ = E.teacher_readout(src, co, sz3); ts, _ = E.teacher_readout(tgt, co, sz3)
            sf = E.fuse_series(E._encode_prism(src, co, sz3), ss)
            af = E.fuse_series(E._gather(E._encode_prism(tgt, co, sz3), anct[None], E.cfg.width), ts)
            cx, cc = E._context([sf, af], [co, co[:, anct]], dev, 1)
            q = (E.query_seed[None, None, :] + ts[:, None, :] + E._size_emb(sz3[:, rect])).contiguous()
            q = E._decode(q, cx, cc, co[:, rect]); pred = E.dec_pixel_head(q)[0].reshape(-1, 16, 16).float().cpu().numpy()
            slt = F.normalize(E.match_slot_proj(q), dim=-1); clr = F.normalize(E.color_head(_blur(tgt[:, rect], 3)), dim=-1)
            pick = (slt @ clr.transpose(1, 2))[0].argmax(1).cpu().numpy()
        tnp = tgt[0, :, :, :, 0].float().cpu().numpy(); snp = src[0, :, :, :, 0].float().cpu().numpy()
        anc_set = set(int(x) for x in anc); black = np.zeros((p, p), np.uint8)
        P = 200
        def Rz(a):
            return np.array(Image.fromarray(a).resize((P, P), Image.NEAREST))
        # 4 rows x 2 cols (SOURCE t1 | TARGET t1c): full scan / full prism / model context / model output
        p11 = Rz(full(voln, zc)); p12 = Rz(full(volc, zc))
        p21 = Rz(pgrid([u8(snp[n]) for n in range(N)])); p22 = Rz(pgrid([u8(tnp[n]) for n in range(N)]))
        p31 = Rz(pgrid([u8(snp[n]) for n in range(N)]))                                            # source: ALL patches are context
        p32 = Rz(pgrid([u8(tnp[n]) if n in anc_set else black for n in range(N)], lambda n: (45, 220, 45) if n in anc_set else None))  # target: only anchors
        p41 = Rz(pgrid([u8(pred[rset[n]]) if n in rset else u8(tnp[n]) for n in range(N)], lambda n: (70, 150, 255) if n in rset else None))
        p42 = Rz(pgrid([u8(tnp[rec[pick[rset[n]]]]) if n in rset else u8(tnp[n]) for n in range(N)],
                       lambda n: ((45, 220, 45) if pick[rset[n]] == rset[n] else (235, 45, 45)) if n in rset else None))
        vs = np.full((P, 6, 3), 50, np.uint8); W = 2 * P + 6
        hs = np.full((6, W, 3), 50, np.uint8)
        rows = [np.hstack([p11, vs, p12]), np.hstack([p21, vs, p22]), np.hstack([p31, vs, p32]), np.hstack([p41, vs, p42])]
        grid = rows[0]
        for r in rows[1:]:
            grid = np.vstack([grid, hs, r])
        LM, TM = 66, 18; canvas = Image.new("RGB", (LM + W, TM + grid.shape[0]), (15, 15, 15))
        canvas.paste(Image.fromarray(grid), (LM, TM)); d = ImageDraw.Draw(canvas)
        d.text((LM + P // 2 - 28, 4), "SOURCE t1", fill=(210, 210, 210)); d.text((LM + P + 6 + P // 2 - 30, 4), "TARGET t1c", fill=(210, 210, 210))
        for k, lb in enumerate(["scan", "prism", "context", "predict"]):
            d.text((6, TM + k * (P + 6) + P // 2 - 4), lb, fill=(210, 210, 210))
        frames.append(canvas)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=110, loop=0, format="GIF")
    print(f"WROTE {a.out} | patient {pid} | {csz}-vox enhancing met | src {a.src}->tgt {a.tgt} | ckpt step {step}")


if __name__ == "__main__":
    main()
