# v5: cross-modal ordering (simplified) + ablation + tumor focus

Built to answer, cheaply, why the pretext stopped correlating with tumor-F1 (the ~0.75 ceiling across
v1–v4). Branches off **v2** (the trusted encoder), strips the A-vs-B relative-ordering / view-CLS /
curriculum, and switches to **3D cube patches**.

## Task

Per bag: pick a target modality **D**. The encoder sees the **other 3 modalities** (random positions,
~30/30/30) **+ a few D anchors** (~6). The decoder must **recreate the ordering** of `n_tgt` (32) held
**modality-D** patches — position matching (chance 1/32) — plus optional **pixel MAE** of the held D.

Because every target is modality D, the ordering match has **no modality shortcut** (honest for free) —
we get v3's honesty in a v1-style setup with no conditional-loss machinery.

- **Cube patches** (`voxels³`, default 8³) — 3D, not 2.5D slabs. Grid parameterized via
  `EncoderConfig.patch_grid`.
- **Prism-conditional physical size:** 32 mm prism → 4 mm patch, 64 mm prism → 8 mm patch (both sampled
  at the same 8³ voxel grid; physical scale rides in the size embedding).
- Positions from the common foreground (union of the 4 modalities), so patches land on anatomy.

## The three questions it answers

1. **Ordering vs cross-MAE ablation** (hyp #4, never done): `--match-weight` / `--mae-weight` →
   ordering-only / MAE-only / both. Is the ordering loss doing anything over pure MAE?
2. **Tumor focus** (hyp #1, the top suspect): `--tumor-frac` biases the prism so **at least some
   segmented tumor is inside it** (anchor = a tumor voxel + offset within the prism half-extent), so the
   bag naturally contains pathology patches instead of ~95% normal tissue. Tumor cloud (`CachedScan.
   tumor_mm`) is loaded from seg (METS/GoAT-gt; graceful fallback to foreground where no seg).
   `tumor_anchor_frac` is logged.
3. **Eval OOD** (hyp #3): TBD — a readout variant that feeds mixed-modality context matching training.

## Metrics / gate

`match_acc` (ordering, chance 1/n_tgt) + a **coord-shuffle control** (permuting target coords must drop
order accuracy to chance — no positional leak). Downstream: the same held-out patch-F1 eval as v1–v4.

## Files

`sampling.sample_v5_batch` (+ cube gather, foreground helpers); `model.forward_v5`,
`EncoderConfig.patch_grid`; `train.train_v5`; `data` tumor-cloud plumbing; `scripts/run_v5.py`;
`scripts/cubic/job_v5.sbatch` (env `MATCH_W`/`MAE_W`/`TUMOR_FRAC`/`SEED`, pins SHA).

## Planned runs (seed 0)

`both` (match 1 / mae 0.25), `ordering-only` (mae 0), `mae-only` (match 0) — each ± `--tumor-frac 0.8`.
The 2×3 tells us whether ordering helps, and whether tumor-focus lifts the ceiling.
