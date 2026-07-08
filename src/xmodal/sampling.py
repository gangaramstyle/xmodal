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


def sample_self_batch(bundles, *, batch_size, token_count, patch_spec, prism_mm, rng, device):
    """Phase-0 (self) batch: one bag of patches per item from a single random scan. Vectorized
    (grouped by scan). `prism_mm` may be non-cubic (variable aspect ratio); `patch_spec` may be
    a 2.5D slab. Returns patches [B,n,v0,v1,v2], coords [B,n,3], series [B]."""
    import torch
    from collections import defaultdict
    prism = tuple(prism_mm) if not np.isscalar(prism_mm) else (float(prism_mm),) * 3
    half = torch.as_tensor(prism, device=device, dtype=torch.float32) / 2.0
    n = token_count
    refs = []
    while len(refs) < batch_size:
        bi = int(rng.integers(len(bundles))); mods = list(bundles[bi].keys())
        refs.append((bi, mods[int(rng.integers(len(mods)))]))
    groups = defaultdict(list)
    for k, r in enumerate(refs):
        groups[r].append(k)
    patch_l = [None] * batch_size; coord_l = [None] * batch_size; ser = [0] * batch_size
    for (bi, m), ks in groups.items():
        sc = bundles[bi][m]; G = len(ks)
        off = patch_offsets_tensor(patch_spec, thick_axis=sc.thick_axis, device=device)
        M = sc.foreground_mm.shape[0]
        anchors = sc.foreground_mm[torch.randint(M, (G,), device=device)]
        centers = anchors[:, None] + (torch.rand(G, n, 3, device=device) * 2 - 1) * half
        coords = centers - anchors[:, None]
        vox = (off[None, None] + centers[:, :, None, None, None, :] - sc.affine_trans) @ sc.affine_inv.T
        p = sample_patches_group(sc.volume, vox)
        for gi, k in enumerate(ks):
            patch_l[k], coord_l[k], ser[k] = p[gi], coords[gi], sc.series_idx
    return dict(patches=torch.stack(patch_l).float(), coords=torch.stack(coord_l).float(),
                series=torch.tensor(ser, device=device, dtype=torch.long))


def sample_cross_batch_vec(bundles, *, batch_size, token_count, patch_spec, prism_mm, rng, device,
                           pairs_per_patient=1):
    """Vectorized `sample_cross_batch`: items sharing a (bundle, source, target) are drawn and
    gathered together — one batched anchor/center draw and ONE `grid_sample` per group per
    modality, instead of per-item kernels. Same output contract as `sample_cross_batch`."""
    import torch
    from collections import defaultdict
    prism = tuple(prism_mm) if not np.isscalar(prism_mm) else (float(prism_mm),) * 3
    half = torch.as_tensor(prism, device=device, dtype=torch.float32) / 2.0
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
    ssrc = [0] * batch_size; stgt = [0] * batch_size
    for (bi, sm, tm), ks in groups.items():
        s = bundles[bi][sm]; t = bundles[bi][tm]; G = len(ks)
        off_s = patch_offsets_tensor(patch_spec, thick_axis=s.thick_axis, device=device)
        off_t = patch_offsets_tensor(patch_spec, thick_axis=t.thick_axis, device=device)
        M = s.foreground_mm.shape[0]
        anchors = s.foreground_mm[torch.randint(M, (G,), device=device)]                 # [G,3]
        centers = anchors[:, None] + (torch.rand(G, n, 3, device=device) * 2 - 1) * half  # [G,n,3]
        coords = centers - anchors[:, None]
        # grouped world-mm -> voxel: [G,n,v0,v1,v2,3]
        vox_s = (off_s[None, None] + centers[:, :, None, None, None, :] - s.affine_trans) @ s.affine_inv.T
        vox_t = (off_t[None, None] + centers[:, :, None, None, None, :] - t.affine_trans) @ t.affine_inv.T
        ps = sample_patches_group(s.volume, vox_s)                                        # [G,n,v0,v1,v2]
        pt = sample_patches_group(t.volume, vox_t)
        for gi, k in enumerate(ks):
            src_p[k], tgt_p[k], coord_l[k] = ps[gi], pt[gi], coords[gi]
            ssrc[k], stgt[k] = s.series_idx, t.series_idx
    return dict(
        source=torch.stack(src_p).float(),
        target=torch.stack(tgt_p).float(),
        coords=torch.stack(coord_l).float(),
        source_series=torch.tensor(ssrc, device=device, dtype=torch.long),
        target_series=torch.tensor(stgt, device=device, dtype=torch.long),
    )
