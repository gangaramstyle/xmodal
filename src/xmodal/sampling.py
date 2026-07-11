"""GPU-efficient physical-mm patch sampling for CT + MR, with 2.5D (slab) patches.

Distilled from brats2026's `cls_naflex`/`ct_naflex` sampler (the efficiency-forward path),
stripped of the band/organ/window-jitter CT specifics. The whole point is throughput:
patch centers are drawn and gathered in **batched tensor ops grouped by scan**, never a
per-prism Python loop of tiny kernels.

Core ideas kept:
- **Physical-mm coordinates**: everything (foreground, prism, patch offsets) lives in patient
  world-mm, so anisotropic/mixed-orientation clinical scans are handled uniformly.
- **2.5D / thick-axis**: a patch can be a 3D cube (`cube_spec`) or a thin in-plane slab
  (`slice_spec`) whose thin dimension is the scan's `thick_axis` (the acquisition/through-plane
  axis, derived from the affine). This is how CT and MR share one sampler across major axes.
- **Vectorized cross-modal batch**: `sample_cross_batch` groups prisms by scan and draws all
  centers + gathers all patches for a group in one `grid_sample`, for both source and (co-
  registered) target modality.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Patch specs (cube = 3D, slice = 2.5D slab)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatchSpec:
    key: str
    mode: str              # "cube" | "slice"
    mm: tuple              # physical extent per patch axis (mm); slice: (m, m, 0.0)
    voxels: tuple          # sample-grid resolution per axis;    slice: (v, v, 1)


def cube_spec(patch_mm: float = 4.0, voxels: int = 16) -> PatchSpec:
    return PatchSpec(f"{patch_mm:g}mm", "cube", (patch_mm,) * 3, (voxels,) * 3)


def slice_spec(patch_mm: float = 4.0, voxels: int = 16) -> PatchSpec:
    """2.5D: a v×v in-plane slab, 1 sample thick along the scan's thick_axis."""
    return PatchSpec(f"slice_{patch_mm:g}mm", "slice", (patch_mm, patch_mm, 0.0), (voxels, voxels, 1))


def _axis_offsets(size_mm: float, voxels: int) -> np.ndarray:
    if voxels == 1 or size_mm == 0:
        return np.zeros(voxels, dtype=np.float32)
    spacing = float(size_mm) / float(voxels)
    return (np.arange(voxels, dtype=np.float32) - (voxels - 1) / 2) * spacing


def patch_offsets(spec: PatchSpec, *, thick_axis: int) -> np.ndarray:
    """Local patch sample offsets in patient physical-mm. Shape [v0, v1, v2, 3].

    For `slice` mode the two in-plane offsets go on the two non-thick axes and the thin
    (collapsed) axis is aligned with `thick_axis` — so the slab is perpendicular to the
    acquisition/through-plane direction regardless of scan orientation.
    """
    if thick_axis not in (0, 1, 2):
        raise ValueError(f"thick_axis must be 0/1/2, got {thick_axis}")
    axes = [_axis_offsets(spec.mm[a], spec.voxels[a]) for a in range(3)]
    gi, gj, gk = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    local = np.stack([gi, gj, gk], axis=-1).astype(np.float32)
    if spec.mode == "cube":
        return local
    if spec.mode != "slice":
        raise ValueError(f"unknown mode {spec.mode!r}")
    phys = np.zeros((*spec.voxels, 3), dtype=np.float32)
    inplane = [a for a in (0, 1, 2) if a != thick_axis]
    phys[..., inplane[0]] = local[..., 0]
    phys[..., inplane[1]] = local[..., 1]
    phys[..., thick_axis] = local[..., 2]
    return phys


def patch_offsets_tensor(spec: PatchSpec, *, thick_axis: int, device):
    import torch
    return torch.as_tensor(patch_offsets(spec, thick_axis=thick_axis), device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Mixed-size 2.5D sampling: per-patch physical size {4,8,16} + per-bag prism {32,64,128}.
# Every patch is a V×V×1 slab regardless of size (only the mm spacing changes), so a bag of
# mixed sizes still stacks; physical scale rides along as `sizes` for the model's size embedding.
# ---------------------------------------------------------------------------

def slab_unit_offsets(thick_axis: int, voxels: int, device):
    """Unit (1 mm in-plane) 2.5D slab offsets [V,V,1,3], oriented by thick_axis. Scale by a
    per-patch size (mm) to get that patch's physical footprint."""
    return patch_offsets_tensor(slice_spec(1.0, voxels), thick_axis=thick_axis, device=device)


def draw_patch_sizes(rng, G: int, n: int, choices, device, per_bag: bool = False):
    """Per-patch physical sizes (mm) [G,n] drawn from `choices` (e.g. (4,8,16)). Default: i.i.d. per
    patch (mixed within a bag). `per_bag=True` draws ONE size per bag (homogeneous bag; still varies
    across bags) — the ablation that removes within-bag scale mixing."""
    import torch
    arr = np.asarray(choices, dtype=np.float32)
    if per_bag:
        s = arr[rng.integers(len(arr), size=(G, 1))]                 # one size per bag
        return torch.as_tensor(np.broadcast_to(s, (G, n)).copy(), device=device)
    return torch.as_tensor(arr[rng.integers(len(arr), size=(G, n))], device=device)


def draw_prism_half(rng, G: int, choices, device):
    """Per-item prism half-extent [G,1,1] (mm) drawn from `choices` (e.g. (32,64,128)); a cubic
    prism per bag. Broadcasts over (n, 3) when placing centers."""
    import torch
    arr = np.asarray(choices, dtype=np.float32)
    return torch.as_tensor(arr[rng.integers(len(arr), size=G)], device=device)[:, None, None] / 2.0


def resolve_thick_axis(sc: "CachedScan", orient: str, rng):
    """Pick the 2.5D slab's thin (through-plane) voxel axis. `orient`:
      'scan'   -> sc.thick_axis (from geometry; = argmax spacing, arbitrary for isotropic data)
      'native' -> the scan's OUTLIER axis (the dim most different from the others, e.g. 240x240x155 -> 2)
                  = the native through-plane. Robust even when spacing is isotropic. (This is the plane
                  the scan was acquired in; it need not be axial -- could be sagittal/coronal natively.)
      'random' -> a random voxel axis (per call) so the model sees all planes.
    Isotropic BraTS defaults 'scan' to axis 0 (sagittal); use 'native'/'random' to fix that."""
    if orient == "random":
        return int(rng.integers(3))
    if orient == "native":
        shp = np.asarray(sc.volume.shape, dtype=np.float32)
        return int(np.argmax(np.abs(shp - np.median(shp))))   # outlier-shaped axis = native through-plane
    return sc.thick_axis


def size_to_extent(sizes, thick_axis: int, thin_mm: float = 1.0):
    """Scalar in-plane sizes [G,n] (mm) -> per-axis extent [G,n,3]: the two in-plane axes = the size,
    the thin (through-plane) `thick_axis` = `thin_mm`. Feeds the model's size embedding both the scale
    AND the slab orientation (which axis is thin)."""
    ext = sizes[..., None].repeat(1, 1, 3)
    ext[..., thick_axis] = thin_mm
    return ext


def mixed_bag_vox(sc: "CachedScan", centers, sizes, unit_off):
    """centers [G,n,3], sizes [G,n] (mm), unit_off [V,V,1,3] -> voxel coords [G,n,V,V,1,3].
    Per-patch offsets = unit_off * size, so each patch samples its own physical footprint."""
    off_pp = unit_off[None, None] * sizes[:, :, None, None, None, None]        # [G,n,V,V,1,3]
    phys = off_pp + centers[:, :, None, None, None, :]
    return (phys - sc.affine_trans) @ sc.affine_inv.T


# ---------------------------------------------------------------------------
# CachedScan: a volume + geometry resident on the GPU
# ---------------------------------------------------------------------------

@dataclass
class CachedScan:
    volume: object          # torch [D, H, W] float32 on device (intensity-normalized)
    affine_inv: object      # torch [3, 3]  R^-1  (world->voxel rotation)
    affine_trans: object    # torch [3]     t     (world origin)
    foreground_mm: object   # torch [M, 3]  world-mm anchor cloud (patch/prism centers)
    thick_axis: int         # acquisition/through-plane voxel axis (thin axis for 2.5D)
    plane_id: int = 0       # 0=axial 1=coronal 2=sagittal (from geometry)
    modality: str = "?"     # e.g. "t1","t1c","t2","flair","CT"
    series_idx: int = 0
    patient: str = ""
    spacing: tuple = (1.0, 1.0, 1.0)
    stats: object = None    # [SCAN_STATS_DIM] float32 scan-calibration vector (v3 scan-context); None if unset


SCAN_STATS_DIM = 28


def scan_stats(voln, keep=None):
    """Position-free per-scan calibration vector [SCAN_STATS_DIM] from the NORMALIZED foreground
    (v3 §2): 9 percentiles [1,5,10,25,50,75,90,95,99] + mean + std + foreground-fraction + 16-bin
    histogram over [0,1]. Captures the tissue-histogram SHAPE (CSF/GM/WM/tumor modes, contrast) for
    scan-relative appearance conditioning — carries no spatial information."""
    fg = voln[keep] if keep is not None else np.asarray(voln).reshape(-1)
    if fg.size == 0:
        return np.zeros(SCAN_STATS_DIM, dtype=np.float32)
    pcts = np.percentile(fg, [1, 5, 10, 25, 50, 75, 90, 95, 99]).astype(np.float32)      # 9
    frac = np.float32(float(keep.mean())) if keep is not None else np.float32(1.0)        # 1
    hist, _ = np.histogram(fg, bins=16, range=(0.0, 1.0))                                 # 16
    hist = (hist / max(hist.sum(), 1)).astype(np.float32)
    return np.concatenate([pcts, [np.float32(fg.mean()), np.float32(fg.std()), frac], hist]).astype(np.float32)


def _thick_and_plane(affine_R: np.ndarray, thick_axis):
    spacing = np.linalg.norm(affine_R, axis=0)          # voxel size per axis (mm)
    if thick_axis is None:
        thick_axis = int(np.argmax(spacing))            # largest spacing = through-plane
    dom = int(np.argmax(np.abs(affine_R[:, thick_axis])))  # world axis the thick voxel-axis points along
    plane_id = {2: 0, 1: 1, 0: 2}.get(dom, 0)           # RAS: z->axial, y->coronal, x->sagittal
    return int(thick_axis), int(plane_id), tuple(float(s) for s in spacing)


def to_device_scan(volume_np, affine, *, modality, device, series_idx=0, patient="",
                   thick_axis=None, fg_thresh=0.02, max_fg=200_000, normalize=True, seed=0) -> CachedScan:
    """Build a CachedScan on `device` from a numpy volume + 4x4 affine. Derives thick_axis
    and plane_id from geometry; builds the world-mm foreground anchor cloud."""
    import torch
    vol_np = np.ascontiguousarray(volume_np, dtype=np.float32)
    affine = np.asarray(affine, np.float32)
    R, t = affine[:3, :3], affine[:3, 3]
    thick, plane, spacing = _thick_and_plane(R, thick_axis)
    # foreground anchors: voxels above a fraction of max intensity -> world mm
    hi = float(vol_np.max()) or 1.0
    vox = np.argwhere(vol_np > fg_thresh * hi).astype(np.float32)
    if len(vox) == 0:
        vox = np.argwhere(np.ones_like(vol_np)).astype(np.float32)
    if len(vox) > max_fg:
        vox = vox[np.random.default_rng(seed).choice(len(vox), max_fg, replace=False)]
    fg = vox @ R.T + t
    keep = vol_np > fg_thresh * hi
    if normalize:  # simple per-scan robust scale to ~[0,1] (percentile window)
        lo, hiP = np.percentile(vol_np[keep], [0.5, 99.5]) if keep.any() else (0.0, hi)
        vol_np = np.clip((vol_np - lo) / max(hiP - lo, 1e-6), 0.0, 1.0)
    return CachedScan(
        volume=torch.as_tensor(np.ascontiguousarray(vol_np, dtype=np.float32), device=device, dtype=torch.float32),
        affine_inv=torch.as_tensor(np.linalg.inv(R).copy(), device=device),
        affine_trans=torch.as_tensor(t.copy(), device=device),
        foreground_mm=torch.as_tensor(fg, device=device),
        thick_axis=thick, plane_id=plane, modality=modality, series_idx=series_idx,
        patient=patient, spacing=spacing, stats=scan_stats(vol_np, keep),
    )


# ---------------------------------------------------------------------------
# GPU primitives: world-mm -> voxel -> bilinear gather
# ---------------------------------------------------------------------------

def phys_to_vox_bag_gpu(scan: CachedScan, offsets_mm, centers_mm):
    """centers_mm [n,3], offsets_mm [v0,v1,v2,3] -> voxel coords [n,v0,v1,v2,3]."""
    phys = offsets_mm[None] + centers_mm[:, None, None, None, :]
    return (phys - scan.affine_trans) @ scan.affine_inv.T


def _grid_sample(volume, vox_norm):
    import torch
    import torch.nn.functional as F
    return F.grid_sample(volume, vox_norm, mode="bilinear", padding_mode="zeros", align_corners=True)


def sample_patches(volume, vox):
    """volume [D,H,W], vox [n,v0,v1,v2,3] -> patches [n,v0,v1,v2] (single scan)."""
    import torch
    D, H, W = volume.shape
    n, v0, v1, v2 = vox.shape[:4]
    size = torch.tensor([D, H, W], device=volume.device, dtype=torch.float32)
    norm = 2.0 * vox / (size - 1).clamp(min=1) - 1.0
    grid = torch.stack([norm[..., 2], norm[..., 1], norm[..., 0]], -1).reshape(1, n * v0, v1, v2, 3)
    out = _grid_sample(volume[None, None], grid)
    return out.reshape(n, v0, v1, v2)


def sample_patches_group(volume, vox):
    """volume [D,H,W], vox [G,n,v0,v1,v2,3] -> patches [G,n,v0,v1,v2] in ONE grid_sample."""
    import torch
    D, H, W = volume.shape
    G, n, v0, v1, v2 = vox.shape[:5]
    size = torch.tensor([D, H, W], device=volume.device, dtype=torch.float32)
    norm = 2.0 * vox / (size - 1).clamp(min=1) - 1.0
    grid = torch.stack([norm[..., 2], norm[..., 1], norm[..., 0]], -1).reshape(G, n * v0, v1, v2, 3)
    out = _grid_sample(volume[None, None].expand(G, -1, -1, -1, -1), grid)
    return out.reshape(G, n, v0, v1, v2)


def draw_centers_prism_gpu(foreground_mm, n, prism_mm):
    """Pick a random anchor in the foreground cloud, then n patch centers uniformly in a
    prism box around it (the cheap body path — continuous uniform-in-box, no multinomial).
    Returns (centers[n,3], coords[n,3] = centers-anchor, anchor[3])."""
    import torch
    M = foreground_mm.shape[0]
    anchor = foreground_mm[torch.randint(M, (1,), device=foreground_mm.device)][0]
    half = torch.as_tensor(prism_mm, device=foreground_mm.device, dtype=foreground_mm.dtype) / 2.0
    centers = anchor[None] + (torch.rand(n, 3, device=foreground_mm.device) * 2 - 1) * half
    return centers, centers - anchor[None], anchor


# ---------------------------------------------------------------------------
# Vectorized cross-modal batch sampler
# ---------------------------------------------------------------------------

def sample_cross_batch(bundles, *, batch_size, token_count, patch_spec, prism_mm, rng, device,
                       pairs_per_patient=1):
    """Draw a cross-modal batch. `bundles` is a list of dicts {modality: CachedScan} (one per
    co-registered patient). For each item: pick a random (source, target) modality pair from a
    bundle, draw ONE prism, and gather matched source+target patches (co-registered -> same
    voxel coords). Prisms sharing a (bundle, pair) are grouped so each group's patches come from
    one grid_sample per modality.

    Returns dict with source/target patches [B,n,v0,v1,v2], coords [B,n,3], and per-item
    source/target series indices [B].
    """
    import torch
    src_p, tgt_p, coord_l, ssrc, stgt = [], [], [], [], []
    prism = tuple(prism_mm) if not np.isscalar(prism_mm) else (float(prism_mm),) * 3
    while len(src_p) < batch_size:
        bundle = bundles[int(rng.integers(len(bundles)))]
        mods = list(bundle.keys())
        if len(mods) < 2:
            continue
        for _ in range(pairs_per_patient):
            if len(src_p) >= batch_size:
                break
            i, j = rng.choice(len(mods), size=2, replace=False)
            s, t = bundle[mods[i]], bundle[mods[j]]
            off_s = patch_offsets_tensor(patch_spec, thick_axis=s.thick_axis, device=device)
            off_t = patch_offsets_tensor(patch_spec, thick_axis=t.thick_axis, device=device)
            centers, coords, _ = draw_centers_prism_gpu(s.foreground_mm, token_count, prism)
            src_p.append(sample_patches(s.volume, phys_to_vox_bag_gpu(s, off_s, centers)))
            tgt_p.append(sample_patches(t.volume, phys_to_vox_bag_gpu(t, off_t, centers)))
            coord_l.append(coords)
            ssrc.append(s.series_idx); stgt.append(t.series_idx)
    return dict(
        source=torch.stack(src_p[:batch_size]).float(),
        target=torch.stack(tgt_p[:batch_size]).float(),
        coords=torch.stack(coord_l[:batch_size]).float(),
        source_series=torch.tensor(ssrc[:batch_size], device=device, dtype=torch.long),
        target_series=torch.tensor(stgt[:batch_size], device=device, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Mixed-modality conditioned sampling (docs/MIXED_MODAL_DESIGN.md)
# Per-patch series drawn from a mixture; self<->cross is a stochastic dominant-series ALIGNMENT ramp.
# ---------------------------------------------------------------------------

def sample_mixes(rng, n_series, present, step, total, *, dom_lo=0.7, dom_hi=0.95, floor=0.1, ramp_frac=0.8,
                 force_align=None):
    """Per-item source/target series distributions [n_series] over `present` (the bundle's series ids).
    Both have ONE dominant series with a sampled share in [dom_lo,dom_hi]; the rest is Dirichlet spread.
    Curriculum: early the target-dominant == source-dominant (aligned/easy same-modal); an alignment
    prob ramps 1->`floor` over `ramp_frac*total` so late the dominants diverge (cross-modal). All
    proportions + the aligned coin are stochastic per item. Floors ensure no series cold-starts.

    `force_align` overrides the curriculum for FIXED validation panels (step-independent, comparable
    across training): 'aligned' (tgt_dom==src_dom), 'cross' (tgt_dom!=src_dom), 'balanced' (uniform
    mix over present for both). None = the training curriculum."""
    present = list(present)
    if force_align == "balanced":
        u = np.zeros(n_series, dtype=np.float64)
        for s in present:
            u[s] = 1.0
        u = u / u.sum()
        return u.copy(), u.copy()
    s_dom = int(rng.choice(present))
    others = [s for s in present if s != s_dom]
    if force_align == "aligned":
        aligned = True
    elif force_align == "cross":
        aligned = not others            # can't be cross with one series present
    else:
        align_p = max(floor, 1.0 - step / max(1.0, ramp_frac * total))
        aligned = (rng.random() < align_p) or (not others)
    t_dom = s_dom if aligned else int(rng.choice(others))

    def mix(dom):
        p = np.zeros(n_series, dtype=np.float64)
        if len(present) == 1:
            p[dom] = 1.0
            return p
        w = rng.dirichlet(np.ones(len(present)))
        share = float(rng.uniform(dom_lo, dom_hi))
        for s, wi in zip(present, w):
            p[s] = (1.0 - share) * wi
        p[dom] += share
        return p / p.sum()

    return mix(s_dom), mix(t_dom)


def _draw_sizes_np(rng, k, choices, per_bag=False):
    arr = np.asarray(choices, dtype=np.float32)
    if per_bag:
        return np.full(k, float(arr[rng.integers(len(arr))]), dtype=np.float32)
    return arr[rng.integers(len(arr), size=k)].astype(np.float32)


def _draw_series_np(rng, k, mix):
    p = np.asarray(mix, dtype=np.float64); p = p / p.sum()
    return rng.choice(len(p), size=k, p=p).astype(np.int64)


def _apply_exclusion(oh, sh, sdh, oa, sa, sda, half, thick_axis, frac, rng, max_tries=8):
    """Enforce a target<->source EXCLUSION radius (design doc §4): a held target must not overlap the
    in-plane footprint of any SAME-SERIES source patch (else it's reconstructable by local copy /
    interpolation). Overlap = in-plane center distance < frac*(s_held+s_src)/2. Cross-series overlaps
    are allowed (different modality appearance carries no copy). Redraws violating held centers in-place
    (best effort; dense small prisms may not fully satisfy it). frac<=0 disables. oh [m,3] world-mm off."""
    if frac <= 0 or len(oa) == 0:
        return oh
    ip = [a for a in (0, 1, 2) if a != thick_axis]
    for _ in range(max_tries):
        d = np.sqrt(((oh[:, None][:, :, ip] - oa[None, :][:, :, ip]) ** 2).sum(-1))   # [m,n] in-plane dist
        thr = frac * (sh[:, None] + sa[None, :]) / 2.0                                 # [m,n] overlap radius
        viol = ((d < thr) & (sdh[:, None] == sda[None, :])).any(1)                     # [m] same-series overlap
        if not viol.any():
            break
        oh[viol] = (rng.random((int(viol.sum()), 3)) * 2 - 1) * half
    return oh


def _apply_hardneg(oh, sh, sdh, present, frac, rng):
    """Paired same-position/different-series HARD NEGATIVES (design doc §5 / review pt 3): for `frac` of
    held slots, place two targets at the IDENTICAL center + size but DISTINCT series, so the InfoNCE
    prism contains same-anatomy/different-modality negatives — a direct test of whether series_q_embed
    routes to modality-specific appearance vs mere anatomy. In-place; frac<=0 or <2 series disables."""
    present = list(present)
    if frac <= 0 or len(present) < 2:
        return
    npair = int(frac * len(oh) // 2)
    for p in range(npair):
        i0, i1 = 2 * p, 2 * p + 1
        oh[i1] = oh[i0]; sh[i1] = sh[i0]                          # identical anatomy + scale
        s0, s1 = rng.choice(present, size=2, replace=False)
        sdh[i0], sdh[i1] = int(s0), int(s1)                       # differ ONLY in modality


PRISM_SIZE_MAP = {32.0: (4.0, 8.0), 64.0: (4.0, 8.0, 16.0)}   # v3: per-prism source sizes (16mm only in 64mm)


def _common_fg(sid2, cap=50000):
    """Union of the co-registered modalities' foreground clouds -> common brain-ish mask [K,3] world-mm
    (subsampled to `cap`). Sampling target/source centers from THIS (not uniform-in-box) is what makes
    patches land on anatomy instead of skull-stripped background (audit: uniform gave tgt_ok ~50% and
    40-60% background at 64/128 mm)."""
    import torch
    clouds = [sc.foreground_mm for sc in sid2.values()]
    fg = torch.cat(clouds, 0) if len(clouds) > 1 else clouds[0]
    if fg.shape[0] > cap:
        fg = fg[torch.randint(fg.shape[0], (cap,), device=fg.device)]
    return fg


def _local_fg(common_fg, anchor, half):
    """Points of `common_fg` inside the prism box (L-inf <= half of anchor) -> [K,3]."""
    d = (common_fg - anchor[None]).abs().amax(-1)
    return common_fg[d <= half]


def _pick_targets(points, k, min_sep, rng):
    """Pick k target indices from foreground `points` [K,3], GREEDILY enforcing mutual >= min_sep (a
    Poisson-disk accept in random order). Interior-weighted (random, not FPS which parks targets on the
    brain rim -> low-foreground patches, audit PED 62%). If the region can't fit k separated points,
    the shortfall is filled with random remaining points (rare on a 64mm prism; caller should log the
    achieved min-dist). Returns [k] indices."""
    import torch
    K = points.shape[0]
    if K <= k:
        return torch.arange(K, device=points.device)
    perm = rng.permutation(K)
    ms2 = float(min_sep) ** 2
    sel = [int(perm[0])]
    for idx in perm[1:]:
        if len(sel) >= k:
            break
        if min_sep <= 0 or bool(((points[torch.as_tensor(sel, device=points.device)] - points[int(idx)]).pow(2).sum(-1) >= ms2).all()):
            sel.append(int(idx))
    if len(sel) < k:                                                     # region too tight: fill remainder randomly
        rest = [int(x) for x in perm if int(x) not in set(sel)]
        sel += rest[: k - len(sel)]
    return torch.as_tensor(sel[:k], device=points.device)


def _band_anchor(fg, a_a, dmin, dmax, rng):
    """View-b anchor: a foreground point ~log-uniform(dmin,dmax) from a_a (for the view-CLS pair)."""
    import torch
    dwant = float(np.exp(rng.uniform(np.log(dmin), np.log(dmax))))
    dist = (fg - a_a[None]).norm(dim=-1)
    band = torch.nonzero((dist >= 0.7 * dwant) & (dist <= 1.3 * dwant), as_tuple=False).flatten()
    return fg[int(band[int(rng.integers(band.numel()))])] if band.numel() > 0 else fg[int((dist - dwant).abs().argmin())]


def _fg_source(local_np, n, sizes_pool, present, share, n_hard_min, dropout, tgt_rel, tsize, thick, excl_frac, rng):
    """Build the source bag from prism foreground offsets `local_np` [K,3] (RELATIVE to anchor). Returns
    per-slot (offsets [n,3], valid_mask [n] bool, series [n], sizes [n], overlap_rate). Real patches
    occupy n_real = min(K,n) slots (minus random dropout, floored at n_hard_min); the rest are REGISTER
    slots (valid=False). Real source-modality counts are EXACT over real slots; real patches are SCATTERED
    into random slots (so register count isn't a positional signal). Source overlapping a target slab is
    redrawn from `local_np`."""
    K = len(local_np)
    n_real = min(K, n)
    if dropout > 0 and n_real > n_hard_min:
        n_real = max(n_hard_min, n_real - int(rng.random() * dropout * n_real))
    cen = local_np[rng.choice(K, size=n_real, replace=(K < n_real))].astype(np.float32)   # [n_real,3] rel anchor
    sizes_r = np.asarray(sizes_pool, np.float32)[rng.integers(len(sizes_pool), size=n_real)]
    orate = 0.0
    if excl_frac > 0 and tgt_rel is not None and n_real:                        # redraw source overlapping a target slab
        ip = [a for a in (0, 1, 2) if a != thick]
        viol = np.zeros(n_real, dtype=bool)
        for _ in range(8):
            d = np.abs(cen[:, None, :] - tgt_rel[None, :, :])                    # [n_real,P,3]
            viol = ((d[:, :, ip[0]] < (sizes_r[:, None] + tsize) / 2) &
                    (d[:, :, ip[1]] < (sizes_r[:, None] + tsize) / 2) & (d[:, :, thick] < 1.0)).any(1)
            if not viol.any():
                break
            cen[viol] = local_np[rng.choice(K, size=int(viol.sum()))]
        orate = float(viol.mean())
    off = np.zeros((n, 3), np.float32); valid = np.zeros(n, bool)
    series = np.zeros(n, np.int64); sizes = np.zeros(n, np.float32)
    slots = rng.choice(n, size=n_real, replace=False)                           # SCATTER real into random slots
    off[slots] = cen; valid[slots] = True; sizes[slots] = sizes_r
    series[slots] = _exact_source_series(rng, present, n_real, share)
    return off, valid, series, sizes, orate


def _exact_source_series(rng, present, n, share):
    """v3 structured curriculum (review pt 4): EXACT per-modality source counts, not Dirichlet+share.
    The dominant modality gets exactly round(share*n); the remainder is split as evenly as possible over
    the others (so every modality is present until share is very high). Order shuffled. Returns [n] ids."""
    present = list(present)
    dom = int(rng.choice(present))
    others = [s for s in present if s != dom]
    dom_c = int(round(share * n))
    rem = n - dom_c
    base, extra = divmod(rem, max(1, len(others)))
    counts = {dom: dom_c}
    for j, s in enumerate(others):
        counts[s] = base + (1 if j < extra else 0)
    arr = np.concatenate([np.full(c, s, dtype=np.int64) for s, c in counts.items()]) if n else np.zeros(0, np.int64)
    rng.shuffle(arr)
    return arr


def _reject_source_overlap(oa, sa, pos, tsize, thick_axis, half, rng, max_tries=8):
    """v3 exclusion, INVERTED + slab geometry (review pt 3): targets placed first (few); redraw the SOURCE
    patches whose AXIS-ALIGNED slab footprint overlaps ANY target's. Overlap needs both in-plane axes AND
    the thin through-plane axis within half-extent sums (a 2.5D slab is thin, so thick-separated patches
    don't overlap even when in-plane-close). Returns (oa, residual_overlap_fraction)."""
    if len(oa) == 0 or len(pos) == 0:
        return oa, 0.0
    ip = [a for a in (0, 1, 2) if a != thick_axis]
    viol = np.zeros(len(oa), dtype=bool)
    for _ in range(max_tries):
        d = np.abs(oa[:, None, :] - pos[None, :, :])                        # [n,P,3]
        ovu = d[:, :, ip[0]] < (sa[:, None] + tsize) / 2.0
        ovv = d[:, :, ip[1]] < (sa[:, None] + tsize) / 2.0
        ovw = d[:, :, thick_axis] < 1.0                                     # thin extents ~1mm -> sum/2 ~1mm
        viol = (ovu & ovv & ovw).any(1)                                     # [n]
        if not viol.any():
            break
        oa[viol] = (rng.random((int(viol.sum()), 3)) * 2 - 1) * half
    return oa, float(viol.mean())


def _gather_slabs(scan, centers, sizes, thick_axis, voxels, device):
    """centers [K,3] world-mm, sizes [K] (mm) -> slab patches [K,V,V,1] from ONE scan (one grid_sample)."""
    import torch
    unit = slab_unit_offsets(thick_axis, voxels, device)                     # [V,V,1,3]
    off = unit[None] * sizes[:, None, None, None, None]                       # [K,V,V,1,3]
    phys = off + centers[:, None, None, None, :]
    vox = (phys - scan.affine_trans) @ scan.affine_inv.T
    return sample_patches(scan.volume, vox)                                  # [K,V,V,1]


def sample_mixed_paired_batch(bundles, *, batch_size, token_count, held_count, n_series, step, total,
                              patch_sizes=(4., 8., 16.), voxels=16, prism_choices=(32., 64.),
                              size_per_bag=False, orient="native", rng, device,
                              pair_dist_min=16.0, pair_dist_max=96.0, win_center_std=0.1, win_width_log_std=0.1,
                              align_floor=0.1, dom_lo=0.7, dom_hi=0.95, ramp_frac=0.8,
                              held_excl_frac=1.0, hardneg_frac=0.0, force_align=None,
                              structured=False, n_pos=12, target_size=8.0, src_share_lo=0.3, src_share_hi=0.9,
                              scan_context=False, n_hard_min=64, source_dropout=0.1, prism_size_map=None):
    """Mixed-modality paired batch. Per item: a fully-visible source bag `a` + `held_count` disjoint
    held targets (bag a) + a 2nd view `b` (view-CLS only). Per-patch series ~ Categorical(source_mix)
    for a,b and Categorical(target_mix) for held, from `sample_mixes(step,total)` on the item's bundle.
    Gather is grouped by distinct (scan,thick) — ONE grid_sample each — then scattered (vectorized).

    Returns patches_a/_b [B,n,V,V,1], held_patches [B,m,V,V,1], coords_a/_b/held [B,*,3],
    sizes_a/_b/held [B,*,3], source_series_a/_b [B,n], target_series [B,m], rel_targets [B,5]."""
    import torch
    n, m, V = token_count, held_count, voxels
    scan_list, gid = [], {}                                                  # distinct scans -> global index
    for bnd in bundles:
        for sc in bnd.values():
            if id(sc) not in gid:
                gid[id(sc)] = len(scan_list); scan_list.append(sc)

    A_src = torch.empty(batch_size, n, V, V, 1, device=device)
    A_held = torch.empty(batch_size, m, V, V, 1, device=device)
    B_src = torch.empty(batch_size, n, V, V, 1, device=device)
    ca = torch.empty(batch_size, n, 3, device=device); cb = torch.empty(batch_size, n, 3, device=device)
    ch = torch.empty(batch_size, m, 3, device=device)
    za = torch.empty(batch_size, n, 3, device=device); zb = torch.empty(batch_size, n, 3, device=device)
    zh = torch.empty(batch_size, m, 3, device=device)
    ssa = torch.zeros(batch_size, n, dtype=torch.long, device=device)
    ssb = torch.zeros(batch_size, n, dtype=torch.long, device=device)
    tsr = torch.zeros(batch_size, m, dtype=torch.long, device=device)
    sva = torch.ones(batch_size, n, dtype=torch.bool, device=device)          # source-valid mask (register where False)
    svb = torch.ones(batch_size, n, dtype=torch.bool, device=device)
    a_anch = torch.empty(batch_size, 3, device=device); b_anch = torch.empty(batch_size, 3, device=device)
    smap_map = prism_size_map or PRISM_SIZE_MAP
    common_cache = {}                                                         # id(bundle) -> common foreground cloud
    st_a = st_b = st_h = None
    if scan_context:                                                        # per-patch scan-stats [B,*,D]
        st_a = torch.zeros(batch_size, n, SCAN_STATS_DIM, device=device)
        st_b = torch.zeros(batch_size, n, SCAN_STATS_DIM, device=device)
        st_h = torch.zeros(batch_size, m, SCAN_STATS_DIM, device=device)
    share = src_share_lo + (src_share_hi - src_share_lo) * min(1.0, step / max(1.0, ramp_frac * total))
    if structured:                                                          # prefilter complete {0,1,2,3} bundles (review pt 5)
        req = {0, 1, 2, 3}
        elig = [b for b in bundles if req.issubset({sc.series_idx for sc in b.values()})]
        if not elig:
            raise RuntimeError(f"Structured v3 needs complete T1/T1c/T2/FLAIR bundles; none of {len(bundles)} qualify")
        pool = elig
    else:
        pool = bundles
    overlap_rates = []

    G_ctr, G_siz, G_key, G_buf, G_i, G_k = [], [], [], [], [], []            # flat gather accumulators
    for i in range(batch_size):
        bnd = pool[int(rng.integers(len(pool)))]
        sid2 = {sc.series_idx: sc for sc in bnd.values()}
        present = sorted(sid2.keys())
        rep = sid2[present[0]]
        thick = resolve_thick_axis(rep, orient, rng)
        fg = rep.foreground_mm; Mf = fg.shape[0]
        prism = float(np.asarray(prism_choices)[rng.integers(len(prism_choices))]); half = prism / 2.0
        sh_share = {"aligned": src_share_lo, "balanced": 1.0 / len(present), "cross": src_share_hi}.get(force_align, share)
        if structured:
            # sample BOTH targets and source from the common foreground (union of the 4 modalities) inside the
            # prism -> patches land on anatomy, not skull-stripped background (audit). Variable real source
            # count; missing slots become register tokens (valid=False).
            bid = id(bnd)
            if bid not in common_cache:
                common_cache[bid] = _common_fg(sid2)
            cfg = common_cache[bid]; sizes_pool = smap_map.get(prism, patch_sizes)
            for _t in range(8):                                          # resample anchor until >=12 valid targets
                a_a = fg[int(rng.integers(Mf))]; local = _local_fg(cfg, a_a, half)
                if local.shape[0] >= n_pos:
                    break
            local_np = (local - a_a[None]).cpu().numpy().astype(np.float32)
            pos = (local[_pick_targets(local, n_pos, target_size, rng)] - a_a[None]).cpu().numpy().astype(np.float32)  # [n_pos,3]
            oa_np, va_np, sda, sa, orate = _fg_source(local_np, n, sizes_pool, present, sh_share, n_hard_min,
                                                      source_dropout, pos, target_size, thick, held_excl_frac, rng)
            overlap_rates.append(orate)
            b_a = _band_anchor(fg, a_a, pair_dist_min, pair_dist_max, rng)
            lb = _local_fg(cfg, b_a, half); lb_np = (lb - b_a[None]).cpu().numpy().astype(np.float32)
            ob_np, vb_np, sdb, sb, _ = _fg_source(lb_np, n, sizes_pool, present, sh_share, n_hard_min,
                                                  source_dropout, None, target_size, thick, held_excl_frac, rng)
            sva[i] = torch.as_tensor(va_np, device=device); svb[i] = torch.as_tensor(vb_np, device=device)
            oh_np = np.repeat(pos, 4, axis=0); sdh = np.tile(np.asarray(present[:4], np.int64), n_pos)
            sh = np.full(n_pos * 4, float(target_size), np.float32)
        else:
            a_a = fg[int(rng.integers(Mf))]; b_a = _band_anchor(fg, a_a, pair_dist_min, pair_dist_max, rng)
            sa = _draw_sizes_np(rng, n, patch_sizes, size_per_bag); sb = _draw_sizes_np(rng, n, patch_sizes, size_per_bag)
            oa_np = (rng.random((n, 3)) * 2 - 1) * half; ob_np = (rng.random((n, 3)) * 2 - 1) * half
            smix, tmix = sample_mixes(rng, n_series, present, step, total, dom_lo=dom_lo, dom_hi=dom_hi,
                                      floor=align_floor, ramp_frac=ramp_frac, force_align=force_align)
            sda = _draw_series_np(rng, n, smix); sdb = _draw_series_np(rng, n, smix)
            sh = _draw_sizes_np(rng, m, patch_sizes, size_per_bag); sdh = _draw_series_np(rng, m, tmix)
            oh_np = (rng.random((m, 3)) * 2 - 1) * half
            oh_np = _apply_exclusion(oh_np, sh, sdh, oa_np, sa, sda, half, thick, held_excl_frac, rng)
            _apply_hardneg(oh_np, sh, sdh, present, hardneg_frac, rng)
        a_anch[i] = a_a; b_anch[i] = b_a
        oa = torch.as_tensor(oa_np, device=device, dtype=fg.dtype)
        ob = torch.as_tensor(ob_np, device=device, dtype=fg.dtype)
        oh = torch.as_tensor(oh_np, device=device, dtype=fg.dtype)
        ca[i] = oa; cb[i] = ob; ch[i] = oh
        za[i] = size_to_extent(torch.as_tensor(sa[None], device=device), thick)[0]
        zb[i] = size_to_extent(torch.as_tensor(sb[None], device=device), thick)[0]
        zh[i] = size_to_extent(torch.as_tensor(sh[None], device=device), thick)[0]
        ssa[i] = torch.as_tensor(sda, device=device); ssb[i] = torch.as_tensor(sdb, device=device)
        tsr[i] = torch.as_tensor(sdh, device=device)
        bmap = np.zeros(n_series, dtype=np.int64)                            # series id -> global scan index (this bundle)
        for sid, sc in sid2.items():
            bmap[sid] = gid[id(sc)]
        if scan_context:
            smap = np.zeros((n_series, SCAN_STATS_DIM), dtype=np.float32)     # series id -> its scan's stats
            for sid, sc in sid2.items():
                assert sc.stats is not None and np.isfinite(sc.stats).all() and sc.stats[12:28].sum() > 0, \
                    f"scan_context: bad/missing stats for {getattr(sc, 'patient', '?')} series {sid}"
                smap[sid] = sc.stats
            st_a[i] = torch.as_tensor(smap[sda], device=device)
            st_b[i] = torch.as_tensor(smap[sdb], device=device)
            st_h[i] = torch.as_tensor(smap[sdh], device=device)
        for ctr, siz, sids, buf in ((a_a[None] + oa, sa, sda, 0), (a_a[None] + oh, sh, sdh, 1), (b_a[None] + ob, sb, sdb, 2)):
            G_ctr.append(ctr); G_siz.append(torch.as_tensor(siz, device=device))
            G_key.append(bmap[sids] * 3 + thick); G_buf.append(np.full(len(sids), buf, np.int64))
            G_i.append(np.full(len(sids), i, np.int64)); G_k.append(np.arange(len(sids), dtype=np.int64))

    allctr = torch.cat(G_ctr); allsiz = torch.cat(G_siz)
    allkey = np.concatenate(G_key); allbuf = np.concatenate(G_buf); alli = np.concatenate(G_i); allk = np.concatenate(G_k)
    bufs = {0: A_src, 1: A_held, 2: B_src}
    for u in np.unique(allkey):
        sel = np.nonzero(allkey == u)[0]
        scan = scan_list[int(u) // 3]; th = int(u) % 3
        sel_t = torch.as_tensor(sel, device=device)
        patches = _gather_slabs(scan, allctr[sel_t], allsiz[sel_t], th, V, device)   # [len,V,V,1]
        bsel, isel, ksel = allbuf[sel], alli[sel], allk[sel]
        for b, buf in bufs.items():
            mb = np.nonzero(bsel == b)[0]
            if mb.size:
                buf[torch.as_tensor(isel[mb], device=device), torch.as_tensor(ksel[mb], device=device)] = \
                    patches[torch.as_tensor(mb, device=device)]

    def _jit(std_c, std_w):                                                   # per-item window jitter params
        c = torch.as_tensor(rng.normal(0.0, std_c, size=batch_size), device=device, dtype=A_src.dtype) if std_c > 0 \
            else torch.zeros(batch_size, device=device, dtype=A_src.dtype)
        w = torch.as_tensor(np.clip(rng.normal(0.0, std_w, size=batch_size), -3 * std_w, 3 * std_w), device=device,
                            dtype=A_src.dtype) if std_w > 0 else torch.zeros(batch_size, device=device, dtype=A_src.dtype)
        return c, w
    ac, aw = _jit(win_center_std, win_width_log_std); bc, bw = _jit(win_center_std, win_width_log_std)
    ea = aw.exp()[:, None, None, None, None]; eb = bw.exp()[:, None, None, None, None]
    # Window jitter is a STUDENT-side augmentation. Keep the CLEAN gathers as canonical references and
    # produce augmented presentations separately (review): the raw source channel + the pixel-recon target
    # are view-augmented; the EMA SEMANTIC target (held_semantic) and the z/CDF channels stay CLEAN.
    A_src_aug = (A_src - ac[:, None, None, None, None]) / ea                  # view-A window on source-A raw
    B_src_aug = (B_src - bc[:, None, None, None, None]) / eb                  # view-B window on source-B raw
    A_held_pixel = (A_held - ac[:, None, None, None, None]) / ea             # pixel target: held in view-A domain
    spatial = (b_anch - a_anch > 0).float()                                  # [B,3] relative anchor order
    window = torch.stack([(bc > ac).float(), (bw > aw).float()], dim=1)      # [B,2] relative window sign
    rel = torch.cat([spatial, window], dim=1)                               # [B,5]
    return dict(patches_a_raw=A_src_aug, patches_a_reference=A_src,          # student source: augmented raw + clean ref
                patches_b_raw=B_src_aug, patches_b_reference=B_src,
                held_semantic=A_held, held_pixel_target=A_held_pixel,        # clean EMA target vs view-A pixel target
                coords_a=ca, coords_b=cb, held_coords=ch,
                sizes_a=za, sizes_b=zb, held_sizes=zh,
                source_series_a=ssa, source_series_b=ssb, target_series=tsr,
                source_valid_a=sva, source_valid_b=svb,        # False slots -> active register tokens
                stats_a=st_a, stats_b=st_b, held_stats=st_h,   # None unless scan_context
                overlap_rate=float(np.mean(overlap_rates)) if overlap_rates else 0.0,
                reg_frac=float((~sva).float().mean()),         # fraction of source slots that are registers
                rel_targets=rel)


# ---------------------------------------------------------------------------
# v4 modality completion (docs/MIXED_V4_DESIGN.md): 32 co-located positions, ONE hidden modality each.
# ---------------------------------------------------------------------------

def sample_modality_completion_batch(bundles, *, batch_size, n_pos=32, prism=64.0, patch_size=8.0,
                                     voxels=16, orient="native", rng, device, scan_context=True,
                                     win_center_std=0.0, win_width_log_std=0.0):
    """Per item: `n_pos` foreground positions in a `prism`; at each position ONE modality is the hidden
    TARGET (balanced: n_pos/4 per modality) and the other 3 are VISIBLE source. -> 3*n_pos visible +
    n_pos targets. Positions are >= patch_size apart, so no source footprint overlaps a target (modality-
    aware exclusion is automatic; no registers/exclusion). Targets ordered MODALITY-major for the loss.
    Returns patches_src_raw/_ref [B,3P,V,V,1], held_semantic/_pixel [B,P,V,V,1], coords_src/_tgt,
    series_src [B,3P], mod_tgt [B,P], stats_src/_tgt (None unless scan_context)."""
    import torch
    P, V = n_pos, voxels; nS = 3 * P; half = prism / 2.0
    assert P % 4 == 0, "n_pos must be divisible by 4 for balanced target modalities"
    req = {0, 1, 2, 3}
    elig = [b for b in bundles if req.issubset({sc.series_idx for sc in b.values()})]
    if not elig:
        raise RuntimeError(f"v4 needs complete T1/T1c/T2/FLAIR bundles; none of {len(bundles)} qualify")
    scan_list, gid = [], {}
    for bnd in elig:
        for sc in bnd.values():
            if id(sc) not in gid:
                gid[id(sc)] = len(scan_list); scan_list.append(sc)

    S_raw = torch.empty(batch_size, nS, V, V, 1, device=device); T_all = torch.empty(batch_size, P, V, V, 1, device=device)
    csrc = torch.empty(batch_size, nS, 3, device=device); ctgt = torch.empty(batch_size, P, 3, device=device)
    zsrc = torch.empty(batch_size, nS, 3, device=device); ztgt = torch.empty(batch_size, P, 3, device=device)
    ser_src = torch.zeros(batch_size, nS, dtype=torch.long, device=device)
    mod_tgt = torch.zeros(batch_size, P, dtype=torch.long, device=device)
    a_anch = torch.empty(batch_size, 3, device=device)
    st_src = torch.zeros(batch_size, nS, SCAN_STATS_DIM, device=device) if scan_context else None
    st_tgt = torch.zeros(batch_size, P, SCAN_STATS_DIM, device=device) if scan_context else None
    common_cache = {}
    G_ctr, G_key, G_buf, G_i, G_k = [], [], [], [], []
    for i in range(batch_size):
        bnd = elig[int(rng.integers(len(elig)))]
        sid2 = {sc.series_idx: sc for sc in bnd.values()}; present = sorted(sid2.keys())
        rep = sid2[present[0]]; thick = resolve_thick_axis(rep, orient, rng); fg = rep.foreground_mm
        bid = id(bnd)
        if bid not in common_cache:
            common_cache[bid] = _common_fg(sid2)
        cfg = common_cache[bid]
        for _t in range(8):
            a_a = fg[int(rng.integers(fg.shape[0]))]; local = _local_fg(cfg, a_a, half)
            if local.shape[0] >= P:
                break
        posw = local[_pick_targets(local, P, patch_size, rng)]           # [P,3] world foreground positions
        posr = (posw - a_a[None]).cpu().numpy().astype(np.float32)
        posw_np = posw.cpu().numpy().astype(np.float32)
        tmod = np.tile(np.arange(4, dtype=np.int64), P // 4); rng.shuffle(tmod)   # balanced hidden modality per position
        order = np.argsort(tmod, kind="stable")                          # MODALITY-major target order
        tmod_o = tmod[order]; ctgt[i] = torch.as_tensor(posr[order], device=device); mod_tgt[i] = torch.as_tensor(tmod_o, device=device)
        vis_m, vis_p = np.where(np.arange(4)[:, None] != tmod[None, :])   # [3P] visible (modality, position)
        csrc[i] = torch.as_tensor(posr[vis_p], device=device); ser_src[i] = torch.as_tensor(vis_m, device=device)
        ztgt[i] = size_to_extent(torch.full((1, P), float(patch_size), device=device), thick)[0]
        zsrc[i] = size_to_extent(torch.full((1, nS), float(patch_size), device=device), thick)[0]
        bmap = np.zeros(8, dtype=np.int64)
        for sid, sc in sid2.items():
            bmap[sid] = gid[id(sc)]
        if scan_context:
            smap = np.zeros((8, SCAN_STATS_DIM), dtype=np.float32)
            for sid, sc in sid2.items():
                assert sc.stats is not None and np.isfinite(sc.stats).all(), f"v4 bad stats {sid}"
                smap[sid] = sc.stats
            st_tgt[i] = torch.as_tensor(smap[tmod_o], device=device); st_src[i] = torch.as_tensor(smap[vis_m], device=device)
        for ctr, sids, buf in ((posw_np[order], tmod_o, 1), (posw_np[vis_p], vis_m, 0)):   # 1=target, 0=source
            G_ctr.append(torch.as_tensor(ctr, device=device, dtype=fg.dtype))
            G_key.append(bmap[sids] * 3 + thick); G_buf.append(np.full(len(sids), buf, np.int64))
            G_i.append(np.full(len(sids), i, np.int64)); G_k.append(np.arange(len(sids), dtype=np.int64))
        a_anch[i] = a_a

    allctr = torch.cat(G_ctr); allsiz = torch.full((allctr.shape[0],), float(patch_size), device=device)
    allkey = np.concatenate(G_key); allbuf = np.concatenate(G_buf); alli = np.concatenate(G_i); allk = np.concatenate(G_k)
    bufs = {0: S_raw, 1: T_all}
    for u in np.unique(allkey):
        sel = np.nonzero(allkey == u)[0]; scan = scan_list[int(u) // 3]; th = int(u) % 3
        patches = _gather_slabs(scan, allctr[torch.as_tensor(sel, device=device)], allsiz[torch.as_tensor(sel, device=device)], th, V, device)
        bsel, isel, ksel = allbuf[sel], alli[sel], allk[sel]
        for bb, buf in bufs.items():
            mb = np.nonzero(bsel == bb)[0]
            if mb.size:
                buf[torch.as_tensor(isel[mb], device=device), torch.as_tensor(ksel[mb], device=device)] = patches[torch.as_tensor(mb, device=device)]

    # optional window jitter (student-side): source raw + pixel target augmented; SEMANTIC target clean
    def _jit(std_c, std_w):
        c = torch.as_tensor(rng.normal(0.0, std_c, size=batch_size), device=device, dtype=S_raw.dtype) if std_c > 0 else torch.zeros(batch_size, device=device, dtype=S_raw.dtype)
        w = torch.as_tensor(np.clip(rng.normal(0.0, std_w, size=batch_size), -3 * std_w, 3 * std_w), device=device, dtype=S_raw.dtype) if std_w > 0 else torch.zeros(batch_size, device=device, dtype=S_raw.dtype)
        return c, w
    ac, aw = _jit(win_center_std, win_width_log_std); ea = aw.exp()[:, None, None, None, None]
    S_aug = (S_raw - ac[:, None, None, None, None]) / ea
    T_pixel = (T_all - ac[:, None, None, None, None]) / ea
    return dict(patches_src_raw=S_aug, patches_src_ref=S_raw, held_semantic=T_all, held_pixel_target=T_pixel,
                coords_src=csrc, coords_tgt=ctgt, sizes_src=zsrc, sizes_tgt=ztgt,
                series_src=ser_src, mod_tgt=mod_tgt, stats_src=st_src, stats_tgt=st_tgt, n_pos=P)


def apply_window_jitter(patches, *, center_std, width_log_std):
    """Per-scan intensity window jitter on a patch bag (ported). Returns (jittered, target[B,2])
    where target = [center_shift, log_width]."""
    import torch
    B = patches.shape[0]
    if center_std <= 0 and width_log_std <= 0:
        return patches, patches.new_zeros((B, 2))
    center = torch.zeros(B, 1, 1, 1, 1, device=patches.device, dtype=patches.dtype)
    if center_std > 0:
        center.normal_(mean=0.0, std=float(center_std))
    log_width = torch.zeros(B, 1, 1, 1, 1, device=patches.device, dtype=patches.dtype)
    if width_log_std > 0:
        log_width.normal_(mean=0.0, std=float(width_log_std)).clamp_(min=-3 * width_log_std, max=3 * width_log_std)
    jittered = (patches - center) / log_width.exp()
    return jittered, torch.cat([center.flatten(1), log_width.flatten(1)], dim=1)


def sample_paired_batch(bundles, *, batch_size, token_count, patch_sizes=(4., 8., 16.), voxels=16,
                        prism_choices=(32., 64., 128.), size_per_bag=False, orient="scan", rng, device,
                        pair_dist_min=16.0, pair_dist_max=96.0, win_center_std=0.1, win_width_log_std=0.1):
    """Phase-0 paired-view batch (for series-CLS + view-CLS). Per item: two prisms (a, b) from one
    scan whose anchors sit ~log-uniform(pair_dist_min,max) apart. MIXED-size patches (per-patch size
    from `patch_sizes`) and a per-item prism scale (from `prism_choices`). Returns patches_a/_b
    [B,n,V,V,1], coords_a/_b [B,n,3], sizes_a/_b [B,n,3] (per-axis mm extent; thin axis ~= 1mm encodes
    slab orientation), rel_targets [B,5], series [B], patient [B]."""
    import torch
    from collections import defaultdict
    n = token_count
    refs = []
    while len(refs) < batch_size:
        bi = int(rng.integers(len(bundles))); mods = list(bundles[bi].keys())
        refs.append((bi, mods[int(rng.integers(len(mods)))]))
    groups = defaultdict(list)
    for k, r in enumerate(refs):
        groups[r].append(k)
    pa = [None] * batch_size; pb = [None] * batch_size; ca = [None] * batch_size; cb = [None] * batch_size
    za = [None] * batch_size; zb = [None] * batch_size
    spat = [None] * batch_size; ser = [0] * batch_size; pat = [0] * batch_size
    for (bi, m), ks in groups.items():
        sc = bundles[bi][m]; G = len(ks); fg = sc.foreground_mm; Mf = fg.shape[0]
        _thick = resolve_thick_axis(sc, orient, rng)
        unit = slab_unit_offsets(_thick, voxels, device)
        half = draw_prism_half(rng, G, prism_choices, device)                            # [G,1,1] per item
        anchors_a = fg[torch.randint(Mf, (G,), device=device)]                           # [G,3]
        d = np.exp(rng.uniform(np.log(pair_dist_min), np.log(pair_dist_max), size=G)).astype(np.float32)
        dt = torch.as_tensor(d, device=device)
        dvec = (fg[None] - anchors_a[:, None]).norm(dim=-1)                              # [G,M]
        band = ((dvec >= 0.7 * dt[:, None]) & (dvec <= 1.3 * dt[:, None])).float()
        near = (dvec - dt[:, None]).abs().argmin(1)
        add = torch.zeros_like(band); add[torch.arange(G, device=device), near] = (band.sum(1) == 0).float()
        anchors_b = fg[torch.multinomial(band + add, 1)[:, 0]]                           # [G,3]
        ctr_a = anchors_a[:, None] + (torch.rand(G, n, 3, device=device) * 2 - 1) * half
        ctr_b = anchors_b[:, None] + (torch.rand(G, n, 3, device=device) * 2 - 1) * half
        sizes_a = draw_patch_sizes(rng, G, n, patch_sizes, device, size_per_bag)                       # [G,n]
        sizes_b = draw_patch_sizes(rng, G, n, patch_sizes, device, size_per_bag)
        pat_a = sample_patches_group(sc.volume, mixed_bag_vox(sc, ctr_a, sizes_a, unit))
        pat_b = sample_patches_group(sc.volume, mixed_bag_vox(sc, ctr_b, sizes_b, unit))
        st = (anchors_b - anchors_a > 0).float()                                         # [G,3]
        ext_a = size_to_extent(sizes_a, _thick); ext_b = size_to_extent(sizes_b, _thick)  # [G,n,3]
        for gi, k in enumerate(ks):
            pa[k], pb[k] = pat_a[gi], pat_b[gi]
            ca[k] = ctr_a[gi] - anchors_a[gi][None]; cb[k] = ctr_b[gi] - anchors_b[gi][None]
            za[k], zb[k] = ext_a[gi], ext_b[gi]
            spat[k] = st[gi]; ser[k] = sc.series_idx; pat[k] = bi
    Pa = torch.stack(pa).float(); Pb = torch.stack(pb).float()
    Pa, ta = apply_window_jitter(Pa, center_std=win_center_std, width_log_std=win_width_log_std)
    Pb, tb = apply_window_jitter(Pb, center_std=win_center_std, width_log_std=win_width_log_std)
    win_t = (tb - ta > 0).float()                                                        # [B,2]
    rel = torch.cat([torch.stack(spat), win_t], dim=1)                                   # [B,5]
    return dict(patches_a=Pa, patches_b=Pb, coords_a=torch.stack(ca).float(), coords_b=torch.stack(cb).float(),
                sizes_a=torch.stack(za).float(), sizes_b=torch.stack(zb).float(),
                rel_targets=rel, series=torch.tensor(ser, device=device, dtype=torch.long),
                patient=torch.tensor(pat, device=device, dtype=torch.long))


def sample_self_batch(bundles, *, batch_size, token_count, patch_sizes=(4., 8., 16.), voxels=16,
                      prism_choices=(32., 64., 128.), size_per_bag=False, orient="scan", rng, device):
    """Phase-0 (self) batch: one bag of patches per item from a single random scan. Vectorized
    (grouped by scan). Mixed-size patches + per-item prism scale. Returns patches [B,n,V,V,1],
    coords [B,n,3], sizes [B,n,3] (per-axis mm extent; thin axis ~= 1mm encodes slab orientation), series [B]."""
    import torch
    from collections import defaultdict
    n = token_count
    refs = []
    while len(refs) < batch_size:
        bi = int(rng.integers(len(bundles))); mods = list(bundles[bi].keys())
        refs.append((bi, mods[int(rng.integers(len(mods)))]))
    groups = defaultdict(list)
    for k, r in enumerate(refs):
        groups[r].append(k)
    patch_l = [None] * batch_size; coord_l = [None] * batch_size; size_l = [None] * batch_size; ser = [0] * batch_size
    for (bi, m), ks in groups.items():
        sc = bundles[bi][m]; G = len(ks)
        _thick = resolve_thick_axis(sc, orient, rng)
        unit = slab_unit_offsets(_thick, voxels, device)
        half = draw_prism_half(rng, G, prism_choices, device)                            # [G,1,1]
        M = sc.foreground_mm.shape[0]
        anchors = sc.foreground_mm[torch.randint(M, (G,), device=device)]
        centers = anchors[:, None] + (torch.rand(G, n, 3, device=device) * 2 - 1) * half
        coords = centers - anchors[:, None]
        sizes = draw_patch_sizes(rng, G, n, patch_sizes, device, size_per_bag)                         # [G,n]
        p = sample_patches_group(sc.volume, mixed_bag_vox(sc, centers, sizes, unit))
        ext = size_to_extent(sizes, _thick)                                              # [G,n,3]
        for gi, k in enumerate(ks):
            patch_l[k], coord_l[k], size_l[k], ser[k] = p[gi], coords[gi], ext[gi], sc.series_idx
    return dict(patches=torch.stack(patch_l).float(), coords=torch.stack(coord_l).float(),
                sizes=torch.stack(size_l).float(), series=torch.tensor(ser, device=device, dtype=torch.long))


def sample_cross_batch_vec(bundles, *, batch_size, token_count, patch_sizes=(4., 8., 16.), voxels=16,
                           prism_choices=(32., 64., 128.), size_per_bag=False, orient="scan", rng, device, pairs_per_patient=1):
    """Vectorized cross-modal batch: items sharing a (bundle, source, target) are drawn and gathered
    together (one batched draw + ONE grid_sample per group per modality). Mixed-size patches (same
    per-patch sizes + centers for source & target so they stay co-registered), per-item prism scale.
    Returns source/target [B,n,V,V,1], coords [B,n,3], sizes [B,n,3] (per-axis mm extent), src/tgt series [B]."""
    import torch
    from collections import defaultdict
    n = token_count

    items = []                                     # (bundle_idx, src_mod, tgt_mod)
    while len(items) < batch_size:
        bi = int(rng.integers(len(bundles))); mods = list(bundles[bi].keys())
        if len(mods) < 2:
            continue
        for _ in range(pairs_per_patient):
            if len(items) >= batch_size:
                break
            i, j = rng.choice(len(mods), size=2, replace=False)
            items.append((bi, mods[i], mods[j]))
    items = items[:batch_size]

    groups = defaultdict(list)
    for k, it in enumerate(items):
        groups[it].append(k)

    src_p = [None] * batch_size; tgt_p = [None] * batch_size; coord_l = [None] * batch_size
    size_l = [None] * batch_size; ssrc = [0] * batch_size; stgt = [0] * batch_size
    for (bi, sm, tm), ks in groups.items():
        s = bundles[bi][sm]; t = bundles[bi][tm]; G = len(ks)
        _thick = resolve_thick_axis(s, orient, rng)      # co-registered s/t share orientation
        unit_s = slab_unit_offsets(_thick, voxels, device)
        unit_t = slab_unit_offsets(_thick, voxels, device)
        half = draw_prism_half(rng, G, prism_choices, device)                            # [G,1,1]
        M = s.foreground_mm.shape[0]
        anchors = s.foreground_mm[torch.randint(M, (G,), device=device)]                 # [G,3]
        centers = anchors[:, None] + (torch.rand(G, n, 3, device=device) * 2 - 1) * half  # [G,n,3]
        coords = centers - anchors[:, None]
        sizes = draw_patch_sizes(rng, G, n, patch_sizes, device, size_per_bag)                         # [G,n] shared src/tgt
        ps = sample_patches_group(s.volume, mixed_bag_vox(s, centers, sizes, unit_s))    # [G,n,V,V,1]
        pt = sample_patches_group(t.volume, mixed_bag_vox(t, centers, sizes, unit_t))
        ext = size_to_extent(sizes, _thick)                                             # [G,n,3]
        for gi, k in enumerate(ks):
            src_p[k], tgt_p[k], coord_l[k], size_l[k] = ps[gi], pt[gi], coords[gi], ext[gi]
            ssrc[k], stgt[k] = s.series_idx, t.series_idx
    return dict(
        source=torch.stack(src_p).float(),
        target=torch.stack(tgt_p).float(),
        coords=torch.stack(coord_l).float(),
        sizes=torch.stack(size_l).float(),
        source_series=torch.tensor(ssrc, device=device, dtype=torch.long),
        target_series=torch.tensor(stgt, device=device, dtype=torch.long),
    )
