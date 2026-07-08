# xmodal — clean cross-modal SSL for medical imaging

One distilled, GPU-efficient implementation of the pieces that worked, runnable across
**molab** (scratch/GPU), **Betty**, and **CUBIC**. Code is in importable modules under
`src/xmodal/`; notebooks `import xmodal.*`. See [`docs/CATALOG.md`](docs/CATALOG.md) for the
map of the old strains and the cleanup plan.

## Status

**Done — `xmodal/sampling.py` (GPU-efficient physical-mm 2.5D sampler, CT + MR):**
- Physical-mm coordinates; anisotropic / mixed-orientation scans handled uniformly.
- **2.5D / thick-axis:** `cube_spec` (3D) or `slice_spec` (thin slab whose thin axis = the
  scan's acquisition/through-plane axis, from affine geometry). Same sampler for CT and MR.
- **Vectorized cross-modal batch** (`sample_cross_batch_vec`): items grouped by scan, one
  `grid_sample` per group per modality (the efficiency-forward path).
- **Validated on real BraTS (HF) on an RTX PRO 6000 Blackwell in molab:**
  geometry (patch-center vs direct voxel = 0.06, bilinear), co-registration (t1/t1c identical
  voxel coords), 2.5D slab shape (16,16,1), throughput **673k patches/s** (vectorized, 1.63×
  the per-item path).

**Done — `xmodal/model.py` (phase-0 mm-RoPE ViT encoder, self-supervised MAE):**
- ViT-S (27.8M), per-spec patch stems + pixel heads, series/view CLS + register + mask tokens.
- **Validated in molab** in one mixed training run: 2.5D slabs `(16,16,1)` + 4mm/8mm cubes
  (variable patch sizes) + random non-cubic prism aspect ratios `(24–48mm)³`, masked-MAE loss
  **0.57 → 0.23** over 80 steps, 41 steps/s on the Blackwell. mm-RoPE positions tokens by
  physical-mm coords so any prism shape / patch size / 2.5D orientation just works.

**Next (the cross-modal "both" path):**
1. `models/` — mm-RoPE ViT encoder + cross-attention decoder + patch stem (lift from
   `brats2026/models/{rope,vit,stem}`).
2. `models/matching.py` — CLIP position→patch matching head (lift from `brats2026`).
3. `train/` — one trainer: phase-0 self (view/series-CLS + patch-MAE) + cross `both`
   (MAE + matching, `forward_cross`) + latent (`forward_cross_latent`).
4. `eval/` — ET-from-T1 specificity probe + held-out ladder + `ablate_source`.
5. `data/` — BraTS-HF loader (done inline in the molab notebook) + CT/NLST loader.

**Dropped (documented dead ends):** Sinkhorn / band matching (`band_ce`/`band_ot`/`dustbin`).

## Use

```bash
pip install -e .        # or add src/ to sys.path
```
```python
from xmodal import sampling as S
scan = S.to_device_scan(vol_np, affine, modality="t1", device="cuda")   # -> CachedScan
batch = S.sample_cross_batch_vec(bundles, batch_size=64, token_count=256,
                                 patch_spec=S.cube_spec(4.0, 16), prism_mm=(36.,)*3,
                                 rng=rng, device="cuda", pairs_per_patient=8)
```

molab notebook: `https://sb-...molab.run/` — clones this repo to `/marimo/xmodal`, `git pull`
in the setup cell picks up new commits.
