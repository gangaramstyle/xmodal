"""BraTS 2026 lesion-wise segmentation metrics — reflective of the challenge task.

Task-1 (METS) scoring per the official wiki:
- **Subject-wise:** Dice (DSC) + Normalized Surface Distance (NSD, with a tolerance in mm).
- **Lesion-wise:** F1 (harmonic mean of precision/recall over matched lesions).
- Segmentation metrics applied only to lesions **> 27 mm^3**.
- **No worst-case penalty** for missing predictions (unlike the older HD95=374 convention).
- Regions: ET / TC / WT (post-tx adds NETC / SNFH / RC).

Label convention (BraTS 2023+ / METS): 1=NETC (non-enh tumor core), 2=SNFH (edema),
3=ET (enhancing), 4=RC (resection cavity, post-tx). Region unions below.
"""
from __future__ import annotations

import numpy as np

# region -> set of label integers that make up that region's binary mask
REGIONS = {
    "ET": {3},
    "TC": {1, 3, 4},          # tumor core = NETC + ET (+ RC post-tx)
    "WT": {1, 2, 3, 4},       # whole tumor = everything
    "NETC": {1},
    "SNFH": {2},
    "RC": {4},
}


def region_mask(seg, region):
    labs = REGIONS[region]
    m = np.zeros(seg.shape, dtype=bool)
    for l in labs:
        m |= (seg == l)
    return m


def dice(pred, gt):
    """Binary Dice. Empty/empty -> 1.0; empty-vs-nonempty -> 0.0."""
    p, g = pred.astype(bool), gt.astype(bool)
    ps, gs = p.sum(), g.sum()
    if ps == 0 and gs == 0:
        return 1.0
    inter = np.logical_and(p, g).sum()
    return float(2 * inter / (ps + gs)) if (ps + gs) > 0 else 0.0


def _surface_voxels(mask):
    """Boundary voxels of a binary mask (voxel is surface if it has a background 6-neighbor)."""
    from scipy.ndimage import binary_erosion
    return mask & ~binary_erosion(mask)


def normalized_surface_distance(pred, gt, *, spacing=(1.0, 1.0, 1.0), tolerance_mm=1.0):
    """NSD: fraction of surface voxels (both directions) within `tolerance_mm` of the other
    surface. 1.0 = perfect boundary agreement. Empty/empty -> 1.0."""
    from scipy.ndimage import distance_transform_edt
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    sp, sg = _surface_voxels(p), _surface_voxels(g)
    # distance (mm) to nearest gt-surface / pred-surface voxel
    dt_to_g = distance_transform_edt(~sg, sampling=spacing)
    dt_to_p = distance_transform_edt(~sp, sampling=spacing)
    p_ok = (dt_to_g[sp] <= tolerance_mm).sum()
    g_ok = (dt_to_p[sg] <= tolerance_mm).sum()
    denom = sp.sum() + sg.sum()
    return float((p_ok + g_ok) / denom) if denom > 0 else 0.0


def lesion_f1(pred, gt, *, spacing=(1.0, 1.0, 1.0), min_lesion_mm3=27.0, overlap_thresh=0.1):
    """Lesion-wise F1. GT and pred split into 26-connected components; a GT lesion is a TP if
    some pred component overlaps it (IoU-ish over the GT lesion) above `overlap_thresh`; leftover
    pred components (above the size floor) are FP; unmatched GT lesions are FN. Lesions below
    `min_lesion_mm3` are ignored. Returns (f1, precision, recall, tp, fp, fn)."""
    from scipy.ndimage import label
    vox_mm3 = float(np.prod(spacing))
    struct = np.ones((3, 3, 3), dtype=int)               # 26-connectivity
    gl, ng = label(gt.astype(bool), structure=struct)
    pl, npd = label(pred.astype(bool), structure=struct)
    gt_ok = [i for i in range(1, ng + 1) if (gl == i).sum() * vox_mm3 >= min_lesion_mm3]
    pr_ok = [i for i in range(1, npd + 1) if (pl == i).sum() * vox_mm3 >= min_lesion_mm3]
    if not gt_ok and not pr_ok:
        return 1.0, 1.0, 1.0, 0, 0, 0
    matched_pred = set()
    tp = 0
    for gi in gt_ok:
        gmask = gl == gi
        hit = False
        for pi in pr_ok:
            if pi in matched_pred:
                continue
            inter = np.logical_and(gmask, pl == pi).sum()
            if inter / max(gmask.sum(), 1) >= overlap_thresh:
                matched_pred.add(pi); hit = True; break
        tp += int(hit)
    fn = len(gt_ok) - tp
    fp = len(pr_ok) - len(matched_pred)
    prec = tp / (tp + fp) if (tp + fp) else (1.0 if not gt_ok else 0.0)
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return float(f1), float(prec), float(rec), tp, fp, fn


def score_case(pred_seg, gt_seg, *, spacing=(1.0, 1.0, 1.0), regions=("ET", "TC", "WT"),
               nsd_tol_mm=1.0, min_lesion_mm3=27.0):
    """Full BraTS-style per-case scorecard over regions: Dice, NSD, lesion-F1 each."""
    out = {}
    for r in regions:
        p, g = region_mask(pred_seg, r), region_mask(gt_seg, r)
        f1, prec, rec, tp, fp, fn = lesion_f1(p, g, spacing=spacing, min_lesion_mm3=min_lesion_mm3)
        out[r] = dict(
            dice=dice(p, g),
            nsd=normalized_surface_distance(p, g, spacing=spacing, tolerance_mm=nsd_tol_mm),
            lesion_f1=f1, precision=prec, recall=rec, tp=tp, fp=fp, fn=fn,
        )
    return out
