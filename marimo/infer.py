"""In-molab inference: build tumor prisms, fine-tune the seg decoder readout, predict per lesion.
Reuses the xmodal repo's data loader + sampling (identical to the CUBIC eval). Options exposed."""
import sys, os, io, json, hashlib, contextlib
sys.path.insert(0, "/marimo/xmodal/src")
import numpy as np, torch
try:
    from PIL import Image   # only needed by the render helpers (molab); train/eval don't use it
except ImportError:
    Image = None
from xmodal import data as D, model as M, sampling as S

MODS = ["t1", "t1c", "t2", "flair"]
REGIONS = {"ET": {3}, "TC": {1, 3}, "WT": {1, 2, 3}}


class SmoothHead(torch.nn.Module):
    def __init__(self, w, n=4, h=128):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Conv3d(w, h, 3, padding=1), torch.nn.GELU(),
                                        torch.nn.Conv3d(h, h, 3, padding=1), torch.nn.GELU(), torch.nn.Conv3d(h, n, 1))

    def forward(self, x):
        return self.net(x)


def _at(vol, at, ai, pts, dev):
    vox = ((pts - at) @ ai.T).round().long()
    shp = torch.as_tensor(vol.shape, device=dev)
    vox = vox.clamp(min=torch.zeros(3, device=dev, dtype=torch.long), max=shp - 1)
    return vol[vox[..., 0], vox[..., 1], vox[..., 2]]


def load_model(ckpt, width=768, heads=12, dev="cuda"):
    ck = torch.load(ckpt, map_location=dev)
    E = M.Phase0Encoder(M.EncoderConfig(width=width, depth=12, heads=heads, n_series=8, patch_grid=(8, 8, 8))).to(dev)
    E.load_state_dict(ck["model"], strict=False)
    for p in E.parameters():
        p.requires_grad_(False)
    return E


def build_prisms(patient_dirs, n_prisms=6, n_src=96, size=4.0, prism_mm=32.0, res=1.0, seed=0,
                 cover=False, neg_frac=0.0, cache=True, dev="cuda", progress=None):
    unit = S._cube_unit(8, dev); rng = np.random.default_rng(seed); out = []
    for ci, pd in enumerate(patient_dirs):
        pid = os.path.basename(pd)
        if progress:
            progress(f"prisms {ci+1}/{len(patient_dirs)}: {pid}")
        try:
            if pd in _BCACHE:
                b = _BCACHE[pd]
            else:
                b = D.load_local_bundle(pid, pd, device=dev, with_seg=True)[0]
                if cache:
                    _BCACHE[pd] = b        # cache small reused pools (eval); skip for the large train pool to avoid OOM
        except Exception:
            continue
        sc0 = b["t1c"]; tmm = sc0.tumor_np
        if tmm is None or len(tmm) == 0:
            continue
        fgn = sc0.foreground_np if sc0.foreground_np is not None else sc0.foreground_mm.cpu().numpy()
        at, ai = sc0.affine_trans, sc0.affine_inv; half = prism_mm / 2.0
        lin = np.arange(-half, half + 1e-3, res, dtype=np.float32); G = len(lin)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij"); grid = np.stack([gx, gy, gz], -1).reshape(-1, 3)
        if cover:                                          # full-coverage lattice, shared across prisms
            nper = int(round(prism_mm / size))             # 32mm / 4mm = 8 cubes per axis (non-overlapping tiling)
            cl = np.arange(-half + size / 2, half, size, dtype=np.float32)[:nper]
            cgx, cgy, cgz = np.meshgrid(cl, cl, cl, indexing="ij")
            lat = np.stack([cgx, cgy, cgz], -1).reshape(-1, 3)      # (512, 3) centers rel. anchor
            cover_rel = np.tile(lat, (4, 1)).astype(np.float32)     # (2048, 3): 512 cells x 4 modalities
            cover_sm = np.repeat(np.arange(4), len(lat))           # modality per patch
        for _ in range(n_prisms):
            if rng.random() < neg_frac:                                    # hard-negative: a prism with NO tumor
                cand = fgn[rng.integers(len(fgn), size=64)].astype(np.float32)
                free = cand[~(np.abs(tmm[None] - cand[:, None]).max(-1) <= half).any(1)]  # no tumor within the prism box
                if len(free) == 0:
                    continue
                anch = free[0]
            else:
                anch = tmm[rng.integers(len(tmm))].astype(np.float32)       # anchor ON a tumor voxel
            gpts = (grid + anch).astype(np.float32)
            if cover:
                cs = (cover_rel + anch).astype(np.float32); sm = cover_sm   # 8^3 x 4mod = 2048, voxel-complete
            else:
                cm = fgn[np.abs(fgn - anch).max(-1) <= half]
                if len(cm) < n_src // 2:
                    continue
                cs = cm[rng.integers(len(cm), size=n_src)].astype(np.float32)
                sm = rng.integers(4, size=n_src)
            sp = np.zeros((len(cs), 8, 8, 8), np.float16)
            for mi, m in enumerate(MODS):
                sel = np.nonzero(sm == mi)[0]
                if sel.size:
                    sp[sel] = S._gather_cubes(b[m], torch.as_tensor(cs[sel], device=dev),
                                              torch.full((sel.size,), size, device=dev), unit, dev).half().cpu().numpy()
            gp = torch.as_tensor(gpts, device=dev)
            gt = _at(sc0.seg_vol, at, ai, gp, dev).long()
            gt = torch.where(gt == 4, torch.full_like(gt, 3), gt).cpu().numpy().astype(np.int8)
            imgs = np.stack([_at(b[m].volume, b[m].affine_trans, b[m].affine_inv, gp, dev).float().cpu().numpy()
                             for m in MODS]).astype(np.float16)
            out.append(dict(sp=sp, sc=cs - anch, sm=sm, gpts=gpts - anch, gt=gt, imgs=imgs, gdim=G, pid=ci, pname=pid,
                            anch=anch.copy(), prism_mm=prism_mm, res=res))
        torch.cuda.empty_cache()       # b stays in _BCACHE for reuse
    return out


_BCACHE = {}
_PCACHE = {}


def _load_patient_cached(pdir, dev="cuda"):
    if pdir not in _PCACHE:
        b = D.load_local_bundle(os.path.basename(pdir), pdir, device=dev, with_seg=True)[0]
        vols = {m: b[m].volume.cpu().numpy() for m in MODS}
        sc = b["t1c"]
        _PCACHE[pdir] = (vols, sc.seg_vol.cpu().numpy(), sc.affine_trans.cpu().numpy(), sc.affine_inv.cpu().numpy())
    return _PCACHE[pdir]


def native_render(result, psl, label=3, patients_root="/marimo/assets/patients"):
    """All 4 native sequences at this prism slice, each with the 1px red prism box + shared TP/FP/FN overlay
    for class `label` (3=ET, 2=edema, 1=NCR). The box interior is filled DENSELY from the prism grid (inverse
    of the diagonal affine), so the anisotropic grid→voxel scaling doesn't leave gaps.
    Returns ({modality: uint8 RGB}, {modality: raw RGB}, brain_z, class_peak_slice)."""
    vols, seg, at, ai = _load_patient_cached(f"{patients_root}/{result['pname']}")
    seg = np.where(seg == 4, 3, seg)
    G = result["gdim"]; half = result["prism_mm"] / 2.0; res = result["res"]; anch = result["anch"]
    lin = np.arange(-half, half + 1e-3, res, dtype=np.float32); psl = int(min(psl, G - 1))
    shp = vols["t1c"].shape
    # box footprint in voxels: map the 4 grid corners + centre plane through the (diagonal) affine
    gx, gy = np.meshgrid(lin, lin, indexing="ij")
    W = np.stack([anch[0] + gx, anch[1] + gy, np.full_like(gx, anch[2] + lin[psl])], -1).reshape(-1, 3)
    V = np.round((W - at) @ ai.T).astype(int)
    bz = int(np.clip(np.median(V[:, 2]), 0, shp[2] - 1))
    Vx = np.clip(V[:, 0], 0, shp[0] - 1); Vy = np.clip(V[:, 1], 0, shp[1] - 1)
    x0, x1, y0, y1 = int(Vx.min()), int(Vx.max()), int(Vy.min()), int(Vy.max())
    # DENSE fill: for each brain voxel row/col in the box, look up its nearest prism-grid index
    aix, aiy = ai[0, 0], ai[1, 1]
    xs = np.arange(x0, x1 + 1); ys = np.arange(y0, y1 + 1)
    ii = np.clip(np.round(((xs / aix + at[0]) - anch[0] + half) / res).astype(int), 0, G - 1)
    jj = np.clip(np.round(((ys / aiy + at[1]) - anch[1] + half) / res).astype(int), 0, G - 1)
    gt2 = result["gt"][:, :, psl]; pred2 = result["pred"][:, :, psl]
    box_gt = gt2[np.ix_(ii, jj)] == label; box_pred = pred2[np.ix_(ii, jj)] == label   # dense box masks for this class
    inbox = np.zeros(shp[:2], bool); inbox[x0:x1 + 1, y0:y1 + 1] = True
    tp = np.zeros(shp[:2], bool); fp = np.zeros(shp[:2], bool); fn = np.zeros(shp[:2], bool)
    tp[x0:x1 + 1, y0:y1 + 1] = box_gt & box_pred
    fp[x0:x1 + 1, y0:y1 + 1] = box_pred & ~box_gt
    fn[x0:x1 + 1, y0:y1 + 1] = box_gt & ~box_pred
    outside = (seg[:, :, bz] == label) & ~inbox                     # GT of this class elsewhere in the slice (context)
    peak = int((result["gt"] == label).sum((0, 1)).argmax())
    panels = {}; clean = {}
    for name, key in [("T1", "t1"), ("T1c", "t1c"), ("T2", "t2"), ("FLAIR", "flair")]:
        sl = vols[key][:, :, bz].astype(np.float32); lo, hi = np.percentile(sl, 1), np.percentile(sl, 99.5)
        base = np.stack([np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1)] * 3, -1)
        clean[name] = (np.rot90(base) * 255).astype(np.uint8)              # raw, no overlay
        rgb = base.copy()
        rgb[outside] = 0.72 * rgb[outside] + 0.28 * np.array([1, .85, 0])
        rgb[tp] = 0.45 * rgb[tp] + 0.55 * np.array([0, .9, .25])
        rgb[fp] = 0.35 * rgb[fp] + 0.65 * np.array([1, .12, .12])
        rgb[fn] = 0.35 * rgb[fn] + 0.65 * np.array([.2, .5, 1])
        rgb[x0:x1 + 1, y0] = [1, 0, 0]; rgb[x0:x1 + 1, y1] = [1, 0, 0]
        rgb[x0, y0:y1 + 1] = [1, 0, 0]; rgb[x1, y0:y1 + 1] = [1, 0, 0]
        panels[name] = (np.rot90(rgb) * 255).astype(np.uint8)
    return panels, clean, bz, peak


def context_fill(result, psl, size=4.0, patients_root="/marimo/assets/patients", zt=2.0):
    """A black canvas the size of the scan, with ONLY the sampled context-patch footprints filled in with
    the native anatomy at those locations, per modality. Reads the display volume directly at each patch's
    voxel footprint (sized from the patient's affine), so it stays exactly registered to the scan — no flip,
    gap, or offset regardless of the acquisition grid. Black = the model never sampled there. Returns (panels, n_near)."""
    vols, seg, at, ai = _load_patient_cached(f"{patients_root}/{result['pname']}")
    G = result["gdim"]; half = result["prism_mm"] / 2.0; res = result["res"]; anch = result["anch"]
    lin = np.arange(-half, half + 1e-3, res, dtype=np.float32); psl = int(min(psl, G - 1))
    shp = vols["t1c"].shape; world_z = anch[2] + lin[psl]
    # brain slice for this prism plane (identical to native_render)
    gx, gy = np.meshgrid(lin, lin, indexing="ij")
    W = np.stack([anch[0] + gx, anch[1] + gy, np.full_like(gx, world_z)], -1).reshape(-1, 3)
    bz = int(np.clip(np.median(np.round((W - at) @ ai.T)[:, 2]), 0, shp[2] - 1))
    sm = result["sm"]
    SW = anch[None, :] + result["sc"]; SV = np.round((SW - at) @ ai.T).astype(int)
    near = np.abs(SW[:, 2] - world_z) <= zt                                # patches whose extent hits this slice
    rx = max(1, int(round(size * abs(ai[0, 0]) / 2))); ry = max(1, int(round(size * abs(ai[1, 1]) / 2)))  # footprint in voxels
    panels = {}
    for mi, (name, key) in enumerate([("T1", "t1"), ("T1c", "t1c"), ("T2", "t2"), ("FLAIR", "flair")]):
        sl = vols[key][:, :, bz].astype(np.float32); lo, hi = np.percentile(sl, 1), np.percentile(sl, 99.5)
        norm = np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1)                  # same normalisation as the raw row
        canvas = np.zeros(shp[:2], np.float32); filled = np.zeros(shp[:2], bool)
        for k in np.nonzero(near & (sm == mi))[0]:
            vx, vy = int(SV[k, 0]), int(SV[k, 1])
            xa, xb, ya, yb = vx - rx, vx + rx, vy - ry, vy + ry
            if xa < 0 or ya < 0 or xb > shp[0] or yb > shp[1]:
                continue
            canvas[xa:xb, ya:yb] = norm[xa:xb, ya:yb]; filled[xa:xb, ya:yb] = True   # native anatomy, aligned
        rgb = np.stack([canvas] * 3, -1); rgb[~filled] = 0
        panels[name] = (np.rot90(rgb) * 255).astype(np.uint8)
    return panels, int(near.sum())


_LBL = [("NCR", 1, [0, .45, 1]), ("edema", 2, [0, .85, .3]), ("ET", 3, [1, .25, .15])]  # blue / green / red


def _box_map(result, psl, vols, at, ai):
    """Dense map from the prism grid onto brain-voxel box coords at this slice (inverse of the diagonal
    affine). Returns (bz, x0,x1,y0,y1, box_gt_labels, box_pred_labels) — full label maps (0..3), box-shaped."""
    G = result["gdim"]; half = result["prism_mm"] / 2.0; res = result["res"]; anch = result["anch"]
    lin = np.arange(-half, half + 1e-3, res, dtype=np.float32); psl = int(min(psl, G - 1))
    shp = vols["t1c"].shape
    gx, gy = np.meshgrid(lin, lin, indexing="ij")
    W = np.stack([anch[0] + gx, anch[1] + gy, np.full_like(gx, anch[2] + lin[psl])], -1).reshape(-1, 3)
    V = np.round((W - at) @ ai.T).astype(int)
    bz = int(np.clip(np.median(V[:, 2]), 0, shp[2] - 1))
    Vx = np.clip(V[:, 0], 0, shp[0] - 1); Vy = np.clip(V[:, 1], 0, shp[1] - 1)
    x0, x1, y0, y1 = int(Vx.min()), int(Vx.max()), int(Vy.min()), int(Vy.max())
    xs = np.arange(x0, x1 + 1); ys = np.arange(y0, y1 + 1)
    ii = np.clip(np.round(((xs / ai[0, 0] + at[0]) - anch[0] + half) / res).astype(int), 0, G - 1)
    jj = np.clip(np.round(((ys / ai[1, 1] + at[1]) - anch[1] + half) / res).astype(int), 0, G - 1)
    bg = result["gt"][:, :, psl][np.ix_(ii, jj)]; bp = result["pred"][:, :, psl][np.ix_(ii, jj)]
    return bz, x0, x1, y0, y1, bg, bp, ii, jj


def _dsc_lab(a, b):
    inter = int((a & b).sum()); s = int(a.sum()) + int(b.sum())
    return -1.0 if s == 0 else 2.0 * inter / s


def multiclass_render(result, psl, bases=("T1c", "FLAIR"), patients_root="/marimo/assets/patients"):
    """GT vs Pred with ALL tumor classes colour-coded (NCR blue, edema green, ET red) on the native
    base modalities, dense box fill. Returns ({f'{base} {GT|Pred}': uint8 RGB}, bz, per-class dice)."""
    vols, seg, at, ai = _load_patient_cached(f"{patients_root}/{result['pname']}")
    bz, x0, x1, y0, y1, bg, bp, _ii, _jj = _box_map(result, psl, vols, at, ai)
    keymap = {"T1": "t1", "T1c": "t1c", "T2": "t2", "FLAIR": "flair"}
    def paint(base_rgb, labels):
        rgb = base_rgb.copy()
        for _, lab, col in _LBL:
            m = np.zeros(rgb.shape[:2], bool); m[x0:x1 + 1, y0:y1 + 1] = labels == lab
            rgb[m] = 0.35 * rgb[m] + 0.65 * np.array(col)
        return (np.rot90(rgb) * 255).astype(np.uint8)
    panels = {}
    for base in bases:
        sl = vols[keymap[base]][:, :, bz].astype(np.float32); lo, hi = np.percentile(sl, 1), np.percentile(sl, 99.5)
        base_rgb = np.stack([np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1)] * 3, -1)
        panels[f"{base} GT"] = paint(base_rgb, bg); panels[f"{base} Pred"] = paint(base_rgb, bp)
    g3, p3 = result["gt"], result["pred"]
    dice = {n: _dsc_lab(p3 == lab, g3 == lab) for n, lab, _ in _LBL}
    dice["TC"] = _dsc_lab(np.isin(p3, [1, 3]), np.isin(g3, [1, 3]))
    dice["WT"] = _dsc_lab(np.isin(p3, [1, 2, 3]), np.isin(g3, [1, 2, 3]))
    return panels, bz, dice


def proj_render(result, label=3, base="T1c", patients_root="/marimo/assets/patients"):
    """Z-max projection of GT vs Pred for `label` (default ET) over the WHOLE prism, on the base modality —
    collapses diffuse over-prediction spread across slices into one view. Returns (gt_rgb, pred_rgb, gt_vox, pred_vox)."""
    vols, seg, at, ai = _load_patient_cached(f"{patients_root}/{result['pname']}")
    G = result["gdim"]; keymap = {"T1": "t1", "T1c": "t1c", "T2": "t2", "FLAIR": "flair"}
    bz, x0, x1, y0, y1, _bg, _bp, ii, jj = _box_map(result, G // 2, vols, at, ai)   # mid-plane base image + box map
    gproj = (result["gt"] == label).any(2); pproj = (result["pred"] == label).any(2)  # (G,G) any-slice
    sl = vols[keymap[base]][:, :, bz].astype(np.float32); lo, hi = np.percentile(sl, 1), np.percentile(sl, 99.5)
    base_rgb = np.stack([np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1)] * 3, -1)
    col = [c for n, l, c in _LBL if l == label][0]
    def paint(proj):
        rgb = base_rgb.copy(); m = np.zeros(rgb.shape[:2], bool); m[x0:x1 + 1, y0:y1 + 1] = proj[np.ix_(ii, jj)]
        rgb[m] = 0.3 * rgb[m] + 0.7 * np.array(col)
        rgb[x0:x1 + 1, y0] = [1, 0, 0]; rgb[x0:x1 + 1, y1] = [1, 0, 0]
        rgb[x0, y0:y1 + 1] = [1, 0, 0]; rgb[x1, y0:y1 + 1] = [1, 0, 0]
        return (np.rot90(rgb) * 255).astype(np.uint8)
    return paint(gproj), paint(pproj), int(gproj.sum()), int(pproj.sum())


def _encode_src(E, sp, sc, sm, size, dev):
    """Encode a prism's source bag once. Encoder is frozen, so the result is a constant that can be
    reused across every query chunk and every fine-tune epoch. Returns (ctx, cc)."""
    nb = sp.shape[0]; nreg = 2 + E.registers.shape[0]
    zs = torch.full((nb, sp.shape[1], 3), size, device=dev)
    with torch.no_grad():
        x = E.encode(*E._context([E.embed(sp, zs, sm)], [sc], dev, nb))
        ctx, cc = E._context([x[:, nreg:]], [sc], dev, nb)
    return ctx, cc


def _decode_q(E, seg_tok, ctx, cc, qp, qsize, dev):
    """Decode query points against a pre-encoded source context. Gradients flow through the decoder
    (and seg_tok / query_seed), never the frozen ctx."""
    nb = qp.shape[0]; zt = torch.full((nb, qp.shape[1], 3), qsize, device=dev)
    q = (E.query_seed[None, None, :] + E._size_emb(zt) + seg_tok[None, None, :]).contiguous()
    return E._decode(q, ctx, cc, qp)


def _prism_src(E, pr, size, dev):
    sp = torch.as_tensor(pr["sp"][None], device=dev).float(); sc = torch.as_tensor(pr["sc"][None], device=dev).float()
    sm = torch.as_tensor(pr["sm"][None], device=dev).long()
    return _encode_src(E, sp, sc, sm, size, dev)


def _dense(E, seg_tok, pr, size, qsize, dev, src=None):
    ctx, cc = _prism_src(E, pr, size, dev) if src is None else src   # encode source ONCE, reuse per chunk
    coords = torch.as_tensor(pr["gpts"], device=dev).float()
    with torch.no_grad():
        sl = [_decode_q(E, seg_tok, ctx, cc, coords[c0:c0 + 4096][None], qsize, dev)[0]
              for c0 in range(0, coords.shape[0], 4096)]
    return torch.cat(sl)


def _gl(head, sl, G):
    return head(sl.reshape(G, G, G, -1).permute(3, 0, 1, 2)[None])[0].permute(1, 2, 3, 0).reshape(-1, 4)


def _dsc(pred, gt):
    inter = (pred & gt).sum().item(); s = pred.sum().item() + gt.sum().item()
    return -1.0 if s == 0 else 2.0 * inter / s


def _nsd(pred, gt, coords, tol=2.0):
    """Normalized surface distance @ tol mm (same impl as the CUBIC eval, so molab matches it). coords in mm.
    Returns None when either mask is empty."""
    if pred.sum() == 0 or gt.sum() == 0:
        return None
    pc, gc = coords[pred], coords[gt]
    d_pg = torch.cdist(pc, gc).min(1).values; d_gp = torch.cdist(gc, pc).min(1).values
    return float(((d_pg <= tol).float().sum() + (d_gp <= tol).float().sum()) / (len(pc) + len(gc)))


def leaderboard_metrics(results, small_vox=50):
    """Aggregate results into BraTS-leaderboard-style metrics: lesionwise mean DSC & NSD per region, plus
    small-instance TP/FN/FP/F1 (small = GT region < small_vox voxels; oracle-eval analogue of the challenge
    detection metric — a lesion 'detected' if pred overlaps GT, FP = pred region present with no GT region)."""
    m = {"n": len(results)}
    for rg in ("et", "tc", "wt"):
        ds = [r[f"dsc_{rg}"] for r in results if r.get(f"dsc_{rg}", -1) >= 0]
        ns = [r[f"nsd_{rg}"] for r in results if r.get(f"nsd_{rg}") is not None]
        m[f"dsc_{rg}"] = round(float(np.mean(ds)), 4) if ds else 0.0
        m[f"nsd_{rg}"] = round(float(np.mean(ns)), 4) if ns else 0.0
        tp = fn = fp = 0
        for r in results:
            gv, pv = r.get(f"gt_vox_{rg}", 0), r.get(f"pred_vox_{rg}", 0)
            if 0 < gv <= small_vox:                      # small GT lesion present
                if r.get(f"dsc_{rg}", 0) > 0: tp += 1
                else: fn += 1
            elif gv == 0 and 0 < pv <= small_vox:        # spurious small prediction, no GT
                fp += 1
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        m[f"sm_tp_{rg}"], m[f"sm_fn_{rg}"], m[f"sm_fp_{rg}"], m[f"sm_f1_{rg}"] = tp, fn, fp, round(f1, 3)
    return m


def _nbytes(x):
    if torch.is_tensor(x):
        return x.element_size() * x.nelement()
    if isinstance(x, (list, tuple)):
        return sum(_nbytes(y) for y in x)
    return 0


class _SrcStore:
    """Encode-once source cache with a VRAM budget. Prisms up to the budget stay encoded on-GPU (reused every
    epoch, exactly like before); prisms beyond it are re-encoded on demand each step. The encoder is frozen, so
    re-encoding is BIT-IDENTICAL to caching — results are INVARIANT to the budget; only speed/memory change.
    Lets a small GPU (p100) run any config by streaming the overflow, while a big GPU (a100) caches it all."""

    def __init__(self, E, prisms, size, dev, budget_gb=None):
        self.E, self.prisms, self.size, self.dev = E, prisms, size, dev
        self.cache = {}
        if not prisms:
            self.kmax = self.per_gb = 0
            return
        if budget_gb is None and str(dev) == "cuda":
            free, _ = torch.cuda.mem_get_info(); budget_gb = free * 0.5 / 1e9   # half of what's free after load
        c0 = _prism_src(E, prisms[0], size, dev); per = _nbytes(c0) / 1e9
        kmax = len(prisms) if not budget_gb else min(len(prisms), max(1, int(budget_gb / max(per, 1e-9))))
        self.cache[0] = c0
        for i in range(1, kmax):
            self.cache[i] = _prism_src(E, prisms[i], size, dev)
        self.kmax, self.per_gb = kmax, per

    def get(self, i):
        c = self.cache.get(i)
        return c if c is not None else _prism_src(self.E, self.prisms[i], self.size, self.dev)


def train_readout(E, train_prisms, epochs=30, epochs2=60, unfreeze=12, warmstart=True,
                  size=4.0, qsize=2.0, seed=0, conv_cache_gb=50.0, src_cache_gb=None, amp=False, dev="cuda", progress=None):
    """Fine-tune the seg readout on train_prisms. Stage 1 (decoder+seg-token+query_seed+linear) trains on ALL
    prisms; stage 2 (conv SmoothHead) caches dense embeddings for only as many prisms as fit in conv_cache_gb
    of RAM (a light head — a subset suffices), so res=1mm at large scale doesn't OOM.
    NOTE: amp=True (bf16 autocast) is ~2x faster but SILENTLY COLLAPSES training to all-background at large
    n_src (>~2900 source patches) — the cross-attention loses too much precision. Left OFF by default; only
    enable for small source bags. Returns a portable (CPU) readout dict."""
    rng = np.random.default_rng(seed); tr = train_prisms
    torch.manual_seed(seed)                               # seed init (linear head + conv)
    if str(dev) == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    try:                                                  # cut avoidable backward nondeterminism (atomicAdd order);
        torch.use_deterministic_algorithms(True, warn_only=True)   # warn_only: ops w/o a det impl still run
    except Exception:
        pass
    def ac():
        return torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()
    src_tr = _SrcStore(E, tr, size, dev, budget_gb=src_cache_gb)   # cache-or-stream; INVARIANT to budget
    if progress and tr:
        progress(f"src cache: {src_tr.kmax}/{len(tr)} prisms on GPU ({src_tr.per_gb*1e3:.0f}MB each), rest streamed")
    seg_tok = (E.seg_query.detach().clone().to(dev) if (warmstart and hasattr(E, "seg_query"))
               else torch.zeros(E.cfg.width, device=dev)).requires_grad_(True)
    dec = list(E.decoder)[-unfreeze:]; lin = torch.nn.Linear(E.cfg.width, 4).to(dev)
    if warmstart and hasattr(E, "seg_cls_head"):
        with torch.no_grad():
            lin.weight.copy_(E.seg_cls_head.weight); lin.bias.copy_(E.seg_cls_head.bias)
    params = list(lin.parameters()) + [seg_tok, E.query_seed]
    for blk in dec:
        for p in blk.parameters():
            p.requires_grad_(True); params.append(p)
    opt = torch.optim.Adam(params, lr=1e-3)
    for e in range(epochs):
        if progress:
            progress(f"fine-tune · decoder epoch {e+1}/{epochs}")
        for i in rng.permutation(len(tr)):
            pr = tr[i]; ctx, cc = src_tr.get(i)
            sub = rng.integers(len(pr["gt"]), size=384)
            qp = torch.as_tensor(pr["gpts"][sub][None], device=dev).float()
            y = torch.as_tensor(pr["gt"][sub], device=dev).long()
            cnt = torch.bincount(y, minlength=4).float(); w = len(y) / (4 * cnt.clamp(min=1))
            with ac():
                logit = lin(_decode_q(E, seg_tok, ctx, cc, qp, qsize, dev)).reshape(-1, 4)
            opt.zero_grad(); torch.nn.functional.cross_entropy(logit.float(), y, weight=w).backward(); opt.step()
    seg_tok.requires_grad_(False); E.query_seed.requires_grad_(False)
    for m in [lin, *dec]:
        for p in m.parameters():
            p.requires_grad_(False)
    # stage 2: cache dense embeddings for only as many prisms as fit conv_cache_gb (conv is a light head)
    G0 = tr[0]["gdim"]; per_mb = (G0 ** 3) * E.cfg.width * 2 / 1e6
    kmax = max(1, min(len(tr), int(conv_cache_gb * 1000 / max(per_mb, 1e-6))))
    cidx = list(rng.permutation(len(tr))[:kmax])
    if progress:
        progress(f"caching {kmax}/{len(tr)} prism embeddings for conv (~{kmax*per_mb/1000:.0f}GB)")
    emb = {}
    with ac():
        for i in cidx:
            emb[i] = _dense(E, seg_tok, tr[i], size, qsize, dev, src=src_tr.get(i)).half().cpu()
    conv = SmoothHead(E.cfg.width).to(dev); opt2 = torch.optim.Adam(conv.parameters(), lr=1e-3)
    for e in range(epochs2):
        if progress:
            progress(f"fine-tune · conv epoch {e+1}/{epochs2}")
        for i in rng.permutation(cidx):
            pr = tr[i]; G = pr["gdim"]; bb = min(11, G); o = rng.integers(0, G - bb + 1, size=3)
            ev = emb[i].reshape(G, G, G, -1)[o[0]:o[0] + bb, o[1]:o[1] + bb, o[2]:o[2] + bb].to(dev).float()
            y = torch.as_tensor(pr["gt"].reshape(G, G, G)[o[0]:o[0] + bb, o[1]:o[1] + bb, o[2]:o[2] + bb].reshape(-1), device=dev).long()
            cnt = torch.bincount(y, minlength=4).float(); w = len(y) / (4 * cnt.clamp(min=1))
            with ac():
                logit = conv(ev.permute(3, 0, 1, 2)[None])[0].permute(1, 2, 3, 0).reshape(-1, 4)
            opt2.zero_grad(); torch.nn.functional.cross_entropy(logit.float(), y, weight=w).backward(); opt2.step()
    cpu = lambda sd: {k: v.detach().cpu() for k, v in sd.items()}
    return {"seg_tok": seg_tok.detach().cpu(), "query_seed": E.query_seed.detach().cpu(),
            "dec": [cpu(blk.state_dict()) for blk in dec], "lin": cpu(lin.state_dict()),
            "conv": cpu(conv.state_dict()), "unfreeze": unfreeze, "size": size, "qsize": qsize}


def eval_readout(E, readout, eval_prisms, dev="cuda", progress=None):
    """Apply a (trained or R2-loaded) readout onto E's frozen encoder and score eval_prisms per lesion."""
    unfreeze, size, qsize = readout["unfreeze"], readout["size"], readout["qsize"]
    seg_tok = readout["seg_tok"].to(dev)
    with torch.no_grad():
        E.query_seed.copy_(readout["query_seed"].to(dev))
    dec = list(E.decoder)[-unfreeze:]
    for blk, sd in zip(dec, readout["dec"]):
        blk.load_state_dict({k: v.to(dev) for k, v in sd.items()})
    lin = torch.nn.Linear(E.cfg.width, 4).to(dev); lin.load_state_dict({k: v.to(dev) for k, v in readout["lin"].items()})
    conv = SmoothHead(E.cfg.width).to(dev); conv.load_state_dict({k: v.to(dev) for k, v in readout["conv"].items()})
    results = []
    with torch.no_grad():
        for j, pr in enumerate(eval_prisms):
            if progress:
                progress(f"evaluating {j+1}/{len(eval_prisms)} lesions")
            G = pr["gdim"]; sl = _dense(E, seg_tok, pr, size, qsize, dev)
            pred = _gl(conv, sl, G).argmax(1).cpu().numpy().astype(np.int8)
            gt = pr["gt"]; coords = torch.as_tensor(pr["gpts"], device=dev)   # mm coords for NSD
            rec = dict(pname=pr["pname"], gdim=G, imgs=pr["imgs"].reshape(4, G, G, G),
                       gt=gt.reshape(G, G, G), pred=pred.reshape(G, G, G),
                       anch=pr["anch"], prism_mm=pr["prism_mm"], res=pr["res"],
                       sp=pr["sp"], sc=pr["sc"], sm=pr["sm"])
            for name, cls in REGIONS.items():
                pm = np.isin(pred, list(cls)); gm = np.isin(gt, list(cls)); k = name.lower()
                rec[f"dsc_{k}"] = _dsc(torch.as_tensor(pm), torch.as_tensor(gm))
                rec[f"nsd_{k}"] = _nsd(torch.as_tensor(pm, device=dev), torch.as_tensor(gm, device=dev), coords)
                rec[f"gt_vox_{k}"] = int(gm.sum()); rec[f"pred_vox_{k}"] = int(pm.sum())
            rec["et_vox"] = rec["gt_vox_et"]; rec["tc_vox"] = rec["gt_vox_tc"]   # back-compat for table/filter
            results.append(rec)
    return results


def fit_predict(E, train_prisms, eval_prisms, epochs=30, epochs2=60, unfreeze=12, warmstart=True,
                size=4.0, qsize=2.0, seed=0, dev="cuda", progress=None):
    """Train the readout then evaluate (backward-compatible one-shot)."""
    if not train_prisms or not eval_prisms:
        return []
    ro = train_readout(E, train_prisms, epochs, epochs2, unfreeze, warmstart, size, qsize, seed, dev, progress)
    return eval_readout(E, ro, eval_prisms, dev, progress)


def lin_vs_conv(E, readout, result, dev="cuda"):
    """From the SAME decoder embeddings, return the stage-1 linear prediction (pre-conv) and the stage-2 conv
    prediction (post-smoothing) for a prism, so you can see whether the conv head helps or hurts.
    Returns (lin_pred_3d, conv_pred_3d)."""
    seg_tok, lin, conv, size, qsize = _apply_readout(E, readout, dev)
    sp = torch.as_tensor(result["sp"][None], device=dev).float()
    sc = torch.as_tensor(result["sc"][None], device=dev).float()
    sm = torch.as_tensor(result["sm"][None], device=dev).long()
    ctx, cc = _encode_src(E, sp, sc, sm, size, dev)
    G = result["gdim"]; half = result["prism_mm"] / 2.0; res = result["res"]
    lin_c = np.arange(-half, half + 1e-3, res, dtype=np.float32)[:G]
    gx, gy, gz = np.meshgrid(lin_c, lin_c, lin_c, indexing="ij")
    coords = torch.as_tensor(np.stack([gx, gy, gz], -1).reshape(-1, 3), device=dev).float()
    with torch.no_grad():
        sl = torch.cat([_decode_q(E, seg_tok, ctx, cc, coords[c0:c0 + 4096][None], qsize, dev)[0]
                        for c0 in range(0, coords.shape[0], 4096)])
        lin_pred = lin(sl).argmax(1).cpu().numpy().astype(np.int8).reshape(G, G, G)   # pre-conv (linear)
        conv_pred = _gl(conv, sl, G).argmax(1).cpu().numpy().astype(np.int8).reshape(G, G, G)  # post-conv
    return lin_pred, conv_pred


def _apply_readout(E, readout, dev="cuda"):
    """Load a readout dict onto E's frozen encoder; return (seg_tok, lin, conv, size, qsize)."""
    seg_tok = readout["seg_tok"].to(dev)
    with torch.no_grad():
        E.query_seed.copy_(readout["query_seed"].to(dev))
    for blk, sd in zip(list(E.decoder)[-readout["unfreeze"]:], readout["dec"]):
        blk.load_state_dict({k: v.to(dev) for k, v in sd.items()})
    lin = torch.nn.Linear(E.cfg.width, 4).to(dev); lin.load_state_dict({k: v.to(dev) for k, v in readout["lin"].items()})
    conv = SmoothHead(E.cfg.width).to(dev); conv.load_state_dict({k: v.to(dev) for k, v in readout["conv"].items()})
    return seg_tok, lin, conv, readout["size"], readout["qsize"]


def consensus_predict(E, readout, result, shift=1.0, agg="intersection", dev="cuda"):
    """Test-time augmentation for one prism: probe the query grid from small +/- shifts (mm) along each axis,
    realign, and aggregate. agg='intersection' keeps voxels ET under EVERY shift (kills position-sensitive
    diffuse FP), 'majority' keeps >half, 'union' keeps any. Returns (consensus_pred_3d, single_shot_pred_3d)."""
    seg_tok, lin, conv, size, qsize = _apply_readout(E, readout, dev)
    sp = torch.as_tensor(result["sp"][None], device=dev).float()
    sc = torch.as_tensor(result["sc"][None], device=dev).float()
    sm = torch.as_tensor(result["sm"][None], device=dev).long()
    ctx, cc = _encode_src(E, sp, sc, sm, size, dev)                       # encode source ONCE
    G = result["gdim"]; half = result["prism_mm"] / 2.0; res = result["res"]
    lin = np.arange(-half, half + 1e-3, res, dtype=np.float32)[:G]         # anchor-relative grid (matches build_prisms)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    base = np.stack([gx, gy, gz], -1).reshape(-1, 3).astype(np.float32)
    offs = [(0, 0, 0), (shift, 0, 0), (-shift, 0, 0), (0, shift, 0), (0, -shift, 0), (0, 0, shift), (0, 0, -shift)]
    masks = []
    with torch.no_grad():
        for d in offs:
            coords = torch.as_tensor(base + np.asarray(d, np.float32), device=dev)
            sl = torch.cat([_decode_q(E, seg_tok, ctx, cc, coords[c0:c0 + 4096][None], qsize, dev)[0]
                            for c0 in range(0, coords.shape[0], 4096)])
            pred = lin(sl) if False else _gl(conv, sl, G)   # conv head over the grid
            masks.append((pred.argmax(1).cpu().numpy().astype(np.int8).reshape(G, G, G) == 3))
    stk = np.stack(masks)                                                 # (n_shifts, G, G, G) ET masks
    if agg == "intersection":
        cons = stk.all(0)
    elif agg == "union":
        cons = stk.any(0)
    else:
        cons = stk.sum(0) >= (len(masks) // 2 + 1)                        # majority
    return cons, masks[0]                                                 # consensus ET mask, single-shot (0-offset) ET mask


# ---- R2 (Cloudflare) readout cache: config (ckpt + settings) -> trained readout in molab-scratch ----
R2_PREFIX = "readouts"


def _r2_store():
    from obstore.store import S3Store
    return S3Store(os.environ.get("R2_BUCKET", "molab-scratch"),
                   access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                   secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                   endpoint=os.environ["R2_ENDPOINT"], region="auto",
                   virtual_hosted_style_request=False)


def readout_hash(cfg):
    """Stable id for a (checkpoint + notebook settings) combo."""
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _r2_get_bytes(store, key):
    import obstore
    try:
        return bytes(obstore.get(store, key).bytes())
    except Exception:
        return None


def _r2_ready():
    return all(os.environ.get(k) for k in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"))


def r2_index():
    """The tracking table: {hash: {'cfg':..., 'metrics':..., 'key':...}}. Empty if creds not set yet."""
    if not _r2_ready():
        return {}
    try:
        b = _r2_get_bytes(_r2_store(), f"{R2_PREFIX}/index.json")
        return json.loads(b) if b else {}
    except Exception:
        return {}


def r2_load_readout(cfg):
    """Return the cached readout dict for this config, or None if not present / creds unset."""
    if not _r2_ready():
        return None
    b = _r2_get_bytes(_r2_store(), f"{R2_PREFIX}/{readout_hash(cfg)}.pt")
    return None if b is None else torch.load(io.BytesIO(b), map_location="cuda")


def r2_save_readout(readout, cfg, metrics=None):
    """Upload the trained readout + register it in the index (metrics filled in later by eval). Returns hash."""
    import obstore
    h = readout_hash(cfg); store = _r2_store(); key = f"{R2_PREFIX}/{h}.pt"
    buf = io.BytesIO(); torch.save(readout, buf)
    obstore.put(store, key, buf.getvalue())
    idx = r2_index(); idx[h] = {"cfg": cfg, "metrics": metrics or {}, "key": key}
    obstore.put(store, f"{R2_PREFIX}/index.json", json.dumps(idx).encode())
    return h


def r2_load_by_hash(h):
    """Load a cached readout by its index hash (for eval driven by table selection)."""
    if not _r2_ready():
        return None
    b = _r2_get_bytes(_r2_store(), f"{R2_PREFIX}/{h}.pt")
    return None if b is None else torch.load(io.BytesIO(b), map_location="cuda")


def r2_list_prisms(prefix="prism_cache_val"):
    """List cached prism keys under `prefix` on R2 (e.g. the uploaded val cache). Returns sorted
    '<prefix>/<pid>/<kind>_<i>.pt' keys. Empty if creds unset."""
    if not _r2_ready():
        return []
    import obstore
    keys = []
    for batch in obstore.list(_r2_store(), prefix=prefix):
        for o in batch:
            k = o["path"] if isinstance(o, dict) else o.path
            if k.endswith(".pt"):
                keys.append(k)
    return sorted(keys)


def cache_result(src, sampling="random", n_src=None):
    """Build a renderer-ready result dict from ONE cached prism — for confirming the cache visually (no model).
    `src` is an R2 key (str), raw .pt bytes, or an already-loaded torch dict. `pred` is all-zeros: the point is
    to see the GT + the source coverage the trainer would slice. sampling='random' shows the deterministic
    random-N bag (prefix of n_src, default all stored); 'cover' shows the full-coverage lattice. Feed the
    result straight into multiclass_render / context_fill / proj_render."""
    if isinstance(src, str):
        src = _r2_get_bytes(_r2_store(), src)
    if isinstance(src, (bytes, bytearray)):
        src = torch.load(io.BytesIO(bytes(src)), map_location="cpu", weights_only=False)
    G = src["gdim"]
    if sampling == "cover":
        sc, sm = src["sc_cover"].numpy(), src["sm_cover"].numpy().astype(int)
    else:
        k = src["sp_rand"].shape[0] if n_src is None else min(int(n_src), src["sp_rand"].shape[0])
        sc, sm = src["sc_rand"][:k].numpy(), src["sm_rand"][:k].numpy().astype(int)
    return dict(pname=src["pid"], kind=src["kind"], idx=int(src["idx"]),
                gdim=G, prism_mm=float(src["prism_mm"]), res=float(src["res"]), anch=np.asarray(src["anch"]),
                sc=sc.astype(np.float32), sm=sm,
                gt=src["gt"].numpy().reshape(G, G, G).astype(int), pred=np.zeros((G, G, G), int))


def r2_set_metrics(h, metrics, eval_cfg=None):
    """Attach eval metrics to an already-cached readout's index entry (no re-upload of the .pt)."""
    import obstore
    store = _r2_store(); idx = r2_index()
    if h in idx:
        idx[h]["metrics"] = metrics
        if eval_cfg is not None:
            idx[h]["eval"] = eval_cfg
        obstore.put(store, f"{R2_PREFIX}/index.json", json.dumps(idx).encode())
    return h
