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

**Phase-0 convergence (held-out patients) validated in molab:** 1500 steps, patient-disjoint
holdout — val MAE **0.59 → 0.097**, tracking train (0.082) → learns generalizable structure,
not memorization. 35 steps/s.

**Done — cross-modal objectives (`model.forward_cross`, `model.forward_cross_latent`) + `xmodal/matching.py`:**
- Cross-attention decoder; `forward_cross(objective='mae'|'match'|'both')` — source@recon +
  target@anchor context → decode target at recon positions → pixel-MAE and/or CLIP
  position→patch matching (blind `ColorHead` + symmetric-InfoNCE `slot_match_loss`).
- `forward_cross_latent` — JEPA-style: predict frozen-teacher target latents (1 - cosine).
- **Validated in molab** (35.3M params): cross-MAE **0.65 → 0.12**, match_acc rises to ~3×
  chance and climbing, latent path runs.

**Done — `xmodal/train.py` (real phase-0 trainer, efficiency infra):**
- **Objective: pixel-MAE** (real; series-CLS + view-CLS to be **ported faithfully**, not approximated).
- Infra: **torch.compile** (encode path) + **bf16 autocast** + **fused AdamW** + grad-clip 1.0 +
  cosine-warmup LR + checkpointing + held-out val.
- **Validated in molab:** converges (val MAE 0.607 → 0.076 on held-out patients); throughput A/B
  on the Blackwell: fp32 37.8 → bf16 43.9 → **bf16+compile 61.9 steps/s (1.64×), 6.7 → 5.2 GB**.

**Loading validated** (real BraTS, HF): all modalities co-registered (identical affines), orientation
LAS (affine/world-mm driven so correct), geometry patch-center=direct-voxel 0.06. NOTE the brightest
t1c voxels are physiologic (label 0), not tumor — the project's premise, not a loading bug.

**Not yet pulled from the best runs (port faithfully — see audit):**
1. **series-CLS** — `rank_hinge_xmod_loss` (`brats2026/losses/contrastive.py`), weight 1.0, the
   largest phase-0 term; needs scan-mate + cross-patient same-modality positives.
2. **view-CLS** — 8-way BCE (3 spatial-ordering + 2 window + 3 rotation signs); needs
   **paired-prism sampling** with rel_targets (`draw_paired_centers_prism_gpu`, window/rotation jitter).
3. **Rotating GPU cache** + prefetch (scale beyond what fits resident); per-modality normalization
   (MR z-score / CT HU windows) + window-jitter aug; brain-mask foreground; NaN/4D sanitize.

**Then:** `eval/` (ET-from-T1 specificity probe + held-out ladder + ablate_source), CT/NLST loader,
BraTS-MET for physiologic-enhancement specificity, longer runs on Betty/CUBIC.

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
