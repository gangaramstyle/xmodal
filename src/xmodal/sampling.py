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
    tumor_mm: object = None   # torch [T,3] world-mm cloud of segmented tumor voxels (v5 tumor-focus); None if no seg
    foreground_np: object = None   # CPU numpy [M,3] mirror of foreground_mm -> sampler geometry never syncs the GPU
    tumor_np: object = None        # CPU numpy [T,3] mirror of tumor_mm (v5 tumor-focus); None if no seg


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
    if normalize:  # simple per-scan robust scale to ~[0,1] (percentile window)
        lo, hiP = np.percentile(vol_np[vol_np > fg_thresh * hi], [0.5, 99.5]) if (vol_np > fg_thresh * hi).any() else (0.0, hi)
        vol_np = np.clip((vol_np - lo) / max(hiP - lo, 1e-6), 0.0, 1.0)
    return CachedScan(
        volume=torch.as_tensor(np.ascontiguousarray(vol_np, dtype=np.float32), device=device, dtype=torch.float32),
        affine_inv=torch.as_tensor(np.linalg.inv(R).copy(), device=device),
        affine_trans=torch.as_tensor(t.copy(), device=device),
        foreground_mm=torch.as_tensor(fg, device=device),
        foreground_np=np.ascontiguousarray(fg, dtype=np.float32),   # CPU mirror -> no per-batch GPU->CPU sync
        thick_axis=thick, plane_id=plane, modality=modality, series_idx=series_idx,
        patient=patient, spacing=spacing,
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


def _gather_slabs(scan, centers, sizes, thick_axis, voxels, device):
    """centers [K,3] world-mm, sizes [K] (mm) -> slab patches [K,V,V,1] from ONE scan (one grid_sample)."""
    import torch
    unit = slab_unit_offsets(thick_axis, voxels, device)                     # [V,V,1,3]
    off = unit[None] * sizes[:, None, None, None, None]                       # [K,V,V,1,3]
    phys = off + centers[:, None, None, None, :]
    vox = (phys - scan.affine_trans) @ scan.affine_inv.T
    return sample_patches(scan.volume, vox)                                  # [K,V,V,1]


def sample_mixed_paired_batch(bundles, *, batch_size, token_count, held_count, n_series, step, total,
                              patch_sizes=(4., 8., 16.), voxels=16, prism_choices=(32., 64., 128.),
                              size_per_bag=False, orient="native", rng, device,
                              pair_dist_min=16.0, pair_dist_max=96.0, win_center_std=0.1, win_width_log_std=0.1,
                              align_floor=0.1, dom_lo=0.7, dom_hi=0.95, ramp_frac=0.8,
                              held_excl_frac=1.0, hardneg_frac=0.0, force_align=None):
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
    a_anch = torch.empty(batch_size, 3, device=device); b_anch = torch.empty(batch_size, 3, device=device)

    G_ctr, G_siz, G_key, G_buf, G_i, G_k = [], [], [], [], [], []            # flat gather accumulators
    for i in range(batch_size):
        bnd = bundles[int(rng.integers(len(bundles)))]
        sid2 = {sc.series_idx: sc for sc in bnd.values()}
        present = sorted(sid2.keys())
        rep = sid2[present[0]]
        thick = resolve_thick_axis(rep, orient, rng)
        fg = rep.foreground_mm; Mf = fg.shape[0]
        half = float(np.asarray(prism_choices)[rng.integers(len(prism_choices))]) / 2.0
        a_a = fg[int(rng.integers(Mf))]
        dwant = float(np.exp(rng.uniform(np.log(pair_dist_min), np.log(pair_dist_max))))
        dist = (fg - a_a[None]).norm(dim=-1)
        band = torch.nonzero((dist >= 0.7 * dwant) & (dist <= 1.3 * dwant), as_tuple=False).flatten()
        b_a = fg[int(band[int(rng.integers(band.numel()))])] if band.numel() > 0 else fg[int((dist - dwant).abs().argmin())]
        a_anch[i] = a_a; b_anch[i] = b_a
        smix, tmix = sample_mixes(rng, n_series, present, step, total, dom_lo=dom_lo, dom_hi=dom_hi,
                                  floor=align_floor, ramp_frac=ramp_frac, force_align=force_align)
        sa = _draw_sizes_np(rng, n, patch_sizes, size_per_bag)
        sb = _draw_sizes_np(rng, n, patch_sizes, size_per_bag)
        sh = _draw_sizes_np(rng, m, patch_sizes, size_per_bag)
        sda = _draw_series_np(rng, n, smix); sdb = _draw_series_np(rng, n, smix); sdh = _draw_series_np(rng, m, tmix)
        oa_np = (rng.random((n, 3)) * 2 - 1) * half
        ob_np = (rng.random((n, 3)) * 2 - 1) * half
        oh_np = (rng.random((m, 3)) * 2 - 1) * half
        oh_np = _apply_exclusion(oh_np, sh, sdh, oa_np, sa, sda, half, thick, held_excl_frac, rng)  # pt2: no same-series copy
        _apply_hardneg(oh_np, sh, sdh, present, hardneg_frac, rng)                                   # pt3: same-pos/diff-series negs
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
    A_src = (A_src - ac[:, None, None, None, None]) / ea                      # view a: source + held share a's window
    A_held = (A_held - ac[:, None, None, None, None]) / ea
    B_src = (B_src - bc[:, None, None, None, None]) / eb
    spatial = (b_anch - a_anch > 0).float()                                  # [B,3] relative anchor order
    window = torch.stack([(bc > ac).float(), (bw > aw).float()], dim=1)      # [B,2] relative window sign
    rel = torch.cat([spatial, window], dim=1)                               # [B,5]
    return dict(patches_a=A_src, patches_b=B_src, held_patches=A_held,
                coords_a=ca, coords_b=cb, held_coords=ch,
                sizes_a=za, sizes_b=zb, held_sizes=zh,
                source_series_a=ssa, source_series_b=ssb, target_series=tsr,
                rel_targets=rel)


# ---------------------------------------------------------------------------
# v5 cross-modal ordering (docs/MIXED_V5_DESIGN.md): 3D CUBE patches; context = 3 modalities (+ a few
# anchors of the target modality D); recreate the ordering of held modality-D patches.
# ---------------------------------------------------------------------------

def _fg_np(sc):
    """CPU numpy foreground for a scan, preferring the pre-computed mirror (no GPU->CPU sync)."""
    return sc.foreground_np if sc.foreground_np is not None else sc.foreground_mm.detach().cpu().numpy()


def _v5_common_fg_np(sid2, cap=50000):
    """Common foreground (union of the 4 modalities) as a CPU numpy cloud [K,3] (subsampled). All the
    per-item geometry runs on this in numpy -> no per-item GPU sync (throughput)."""
    clouds = [_fg_np(sc) for sc in sid2.values()]
    fg = (np.concatenate(clouds, 0) if len(clouds) > 1 else clouds[0]).astype(np.float32)
    if len(fg) > cap:
        fg = fg[np.random.default_rng(0).choice(len(fg), cap, replace=False)]
    return fg


def _v5_local_np(fg, anchor, half):
    return fg[np.abs(fg - anchor[None]).max(-1) <= half]


def _v5_pick_np(points, k, min_sep, rng):
    """Greedy min-separation pick of k target positions (numpy; no GPU). Distinct D targets -> clean order."""
    K = len(points)
    if K <= k:
        return np.arange(K)
    perm = rng.permutation(K); ms2 = float(min_sep) ** 2; sel = [int(perm[0])]
    for idx in perm[1:]:
        if len(sel) >= k:
            break
        if min_sep <= 0 or float(((points[sel] - points[idx]) ** 2).sum(-1).min()) >= ms2:
            sel.append(int(idx))
    if len(sel) < k:
        sel += [int(x) for x in perm if int(x) not in set(sel)][: k - len(sel)]
    return np.asarray(sel[:k], dtype=np.int64)


def _h2d(x, device):
    """Host->device transfer, pinned + non_blocking on CUDA so the copy overlaps GPU compute."""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(x))
    if str(device).startswith("cuda"):
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def _cube_unit(V, device):
    return patch_offsets_tensor(cube_spec(1.0, V), thick_axis=0, device=device)   # [V,V,V,3] unit cube


def _gather_cubes(scan, centers, sizes, unit, device):
    """centers [K,3] world-mm, sizes [K] (mm) -> cube patches [K,V,V,V] from ONE scan (one grid_sample)."""
    off = unit[None] * sizes[:, None, None, None, None]                            # [K,V,V,V,3]
    phys = off + centers[:, None, None, None, :]
    vox = (phys - scan.affine_trans) @ scan.affine_inv.T
    return sample_patches(scan.volume, vox)                                        # [K,V,V,V]


def sample_v5_batch(bundles, *, batch_size, n_src, n_anchor, n_tgt, prism_choices=(32., 64.),
                    prism_patch=None, voxels=8, rng, device, tumor_frac=0.0):
    """Per item: pick a target modality D; the source bag = `n_src` patches of the OTHER 3 modalities
    (random position+modality) + `n_anchor` D anchors; the targets = `n_tgt` held D patches to ORDER.
    Cube patches (voxels^3), prism-conditional size (32mm->4mm, 64mm->8mm). Positions from the common
    foreground. Because every target is modality D, the ordering match is honest (no modality shortcut).
    Returns patches_src/held [B,*,V,V,V], coords_src/tgt [B,*,3], series_src [B,nS], mod_tgt [B,P],
    sizes_src/tgt [B,*,3]."""
    import torch
    prism_patch = prism_patch or {32.0: 4.0, 64.0: 8.0}
    P, V, nS = n_tgt, voxels, n_src + n_anchor
    req = {0, 1, 2, 3}
    elig = [b for b in bundles if req.issubset({sc.series_idx for sc in b.values()})]
    if not elig:
        raise RuntimeError(f"v5 needs complete T1/T1c/T2/FLAIR bundles; none of {len(bundles)} qualify")
    scan_list, gid = [], {}
    for bnd in elig:
        for sc in bnd.values():
            if id(sc) not in gid:
                gid[id(sc)] = len(scan_list); scan_list.append(sc)
    unit = _cube_unit(V, device)
    Src = torch.empty(batch_size, nS, V, V, V, device=device); Tgt = torch.empty(batch_size, P, V, V, V, device=device)
    csrc_np = np.zeros((batch_size, nS, 3), np.float32); ctgt_np = np.zeros((batch_size, P, 3), np.float32)
    ser_np = np.zeros((batch_size, nS), np.int64); mod_np = np.zeros((batch_size, P), np.int64)
    zsrc_np = np.zeros((batch_size, nS, 3), np.float32); ztgt_np = np.zeros((batch_size, P, 3), np.float32)
    common_cache = {}; tumor_hits = 0
    G_ctr, G_key, G_buf, G_i, G_k, G_sz = [], [], [], [], [], []
    for i in range(batch_size):                                        # ALL geometry in numpy -> no per-item GPU sync
        bnd = elig[int(rng.integers(len(elig)))]
        sid2 = {sc.series_idx: sc for sc in bnd.values()}
        rep = sid2[sorted(sid2)[0]]
        bid = id(bnd)
        if bid not in common_cache:
            _tmm = rep.tumor_np if rep.tumor_np is not None else (
                rep.tumor_mm.detach().cpu().numpy() if rep.tumor_mm is not None else None)
            common_cache[bid] = (_v5_common_fg_np(sid2), _tmm)
        cfg, tmm = common_cache[bid]
        prism = float(np.asarray(prism_choices)[rng.integers(len(prism_choices))]); half = prism / 2.0
        ps = float(prism_patch[prism])
        use_tumor = tumor_frac > 0 and tmm is not None and rng.random() < tumor_frac
        for _t in range(8):
            if use_tumor:                                              # anchor = tumor voxel + offset so it lands in the prism
                a_a = tmm[int(rng.integers(len(tmm)))] + (rng.random(3) * 2 - 1) * half
            else:
                a_a = cfg[int(rng.integers(len(cfg)))]
            local = _v5_local_np(cfg, a_a.astype(np.float32), half)
            if len(local) >= n_tgt:
                break
        tumor_hits += int(use_tumor)
        K = len(local)
        D = int(rng.integers(4)); ctx = np.array([m for m in range(4) if m != D], dtype=np.int64)
        src_pos = local[rng.choice(K, size=n_src, replace=(K < n_src))]
        src_mod = ctx[rng.integers(3, size=n_src)]                                  # random A/B/C per source patch
        anc_pos = local[rng.choice(K, size=n_anchor, replace=(K < n_anchor))]
        tgt_pos = local[_v5_pick_np(local, n_tgt, ps, rng)]
        src_all = np.concatenate([src_pos, anc_pos], 0).astype(np.float32)          # [nS,3] world
        mods = np.concatenate([src_mod, np.full(n_anchor, D, np.int64)])
        csrc_np[i] = src_all - a_a[None]; ser_np[i] = mods
        ctgt_np[i] = tgt_pos - a_a[None]; mod_np[i] = D
        zsrc_np[i] = ps; ztgt_np[i] = ps                                            # cube: per-axis extent = ps
        bmap = np.zeros(8, np.int64)
        for sid, sc in sid2.items():
            bmap[sid] = gid[id(sc)]
        for ctr, mm, buf in ((src_all, mods, 0), (tgt_pos.astype(np.float32), np.full(n_tgt, D, np.int64), 1)):
            G_ctr.append(ctr); G_key.append(bmap[mm]); G_buf.append(np.full(len(mm), buf, np.int64))
            G_i.append(np.full(len(mm), i, np.int64)); G_k.append(np.arange(len(mm), dtype=np.int64))
            G_sz.append(np.full(len(mm), ps, np.float32))

    csrc = _h2d(csrc_np, device); ctgt = _h2d(ctgt_np, device)              # pinned + non_blocking single transfers
    ser_src = _h2d(ser_np, device); mod_tgt = _h2d(mod_np, device)
    zsrc = _h2d(zsrc_np, device); ztgt = _h2d(ztgt_np, device)
    # sort the gather requests by scan so each scan's cubes are one contiguous segment (one grid_sample, no per-scan H2D)
    allkey = np.concatenate(G_key); order = np.argsort(allkey, kind="stable")
    allkey_s = allkey[order]
    ctr_g = _h2d(np.concatenate(G_ctr)[order], device); sz_g = _h2d(np.concatenate(G_sz)[order], device)
    alli_s = np.concatenate(G_i)[order]; allk_s = np.concatenate(G_k)[order]; allbuf_s = np.concatenate(G_buf)[order]
    N = allkey_s.shape[0]
    Patches = torch.empty(N, V, V, V, device=device)
    uniq, starts, counts = np.unique(allkey_s, return_index=True, return_counts=True)
    for u, s, c in zip(uniq.tolist(), starts.tolist(), counts.tolist()):      # ONE grid_sample per resident scan
        Patches[s:s + c] = _gather_cubes(scan_list[u], ctr_g[s:s + c], sz_g[s:s + c], unit, device)  # contiguous, no sync
    for bb, buf in ((0, Src), (1, Tgt)):                                      # exactly two integer-index scatters (no sync)
        mb = np.nonzero(allbuf_s == bb)[0]
        if mb.size:
            buf[_h2d(alli_s[mb], device), _h2d(allk_s[mb], device)] = Patches[_h2d(mb.astype(np.int64), device)]
    return dict(patches_src=Src, held=Tgt,                             # [B,*,V,V,V] (grid is V^3; stem adds channel)
                coords_src=csrc, coords_tgt=ctgt, sizes_src=zsrc, sizes_tgt=ztgt,
                series_src=ser_src, mod_tgt=mod_tgt,
                tumor_anchor_frac=tumor_hits / batch_size)


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
