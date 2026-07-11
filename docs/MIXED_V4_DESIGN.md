# v4: direct modality completion

v3 is **spatial completion** ("what belongs at this location, given surrounding anatomy" — all 4
modalities held per position). The primary scientific goal is **modality completion**: *given T1, T2,
FLAIR here, what does T1c look like here?* v4 implements that directly. (v3 keeps running as a
spatial-completion / anomaly-prior baseline.)

You can't get modality completion by relaxing v3's exclusion: because v3 holds all 4 modalities at each
position, exposing one to the encoder would leak another held target. The **target structure** must
change.

## Task

- **32 co-located positions** in a **64 mm** prism, all **8 mm** patches.
- At each position, exactly **one** modality is the hidden **target**; the other **three are visible**.
- Target modality **balanced**: 8 × {T1, T1c, T2, FLAIR}. → **96 visible + 32 hidden** per bag.
- For a T1c target at position p: T1, T2, FLAIR at p are visible; T1c at p is hidden. The 3 co-located
  modalities give the local pathology evidence; the other 31 positions give broader anatomical context.

## Exclusion is automatic

Positions are sampled from foreground with a **min-separation ≥ patch size (8 mm)** (greedy Poisson-disk,
`_pick_targets`). Source patches sit at the *same* 32 positions, so a same-modality source and a target
are always ≥8 mm apart → 8 mm slab footprints never overlap. No position-wide exclusion, no register /
source-count logic. (Smoke: worst-item target min-dist 8.1 mm.)

## Loss

**position | target-modality** only. Targets are ordered **modality-major** (`m*8 + p`); within each
target modality, retrieve the right position among that modality's 8 (chance **1/8**), both directions
(`modality_completion_loss`). Modality can't shortcut it — every candidate in the comparison is already
the requested modality. Plus **pixel MAE** on the (view-domain) target.

```
L = 1.0 · L_position|target-modality  +  0.25 · L_pixel
```

Dropped vs v3: modality|position, global 48-way, source-dominance curriculum, aligned/cross/balanced
panels, hard-negatives, view-B + relative spatial/window losses, variable registers, source-count logic,
position-wide exclusion (and the older phased/latent/series paths are unused).

## Kept

BraTS loading, foreground-aware mm-sampling, mm-RoPE, shared stem, modality embeddings, clean
raw/z-score/CDF scan-relative target (scan-context A), EMA target encoder, continuous matching, pixel
recon, the fixed register bank (the 4 encoder registers). Architecture unchanged — only the sampler,
target layout, matching loss, and training loop are new (minimal-risk: change the task, not the model).

## Fixed for the first experiment

prism 64 mm · patch 8 mm · 32 positions · 4 modalities · 8 targets/modality · scan-context A · EMA.
Deliberately **not** combined with multi-prism/-size, missing-modality simulation, spatial-only holes,
or intensity-augmentation curricula — those come after the minimal experiment works.

## Files

`matching.modality_completion_loss`; `sampling.sample_modality_completion_batch`;
`model.forward_modality_completion`; `train.train_modality`; `scripts/run_modality.py`;
`scripts/cubic/job_mixed_v4.sbatch` (pins the launch SHA).

## Metrics / gate

`acc_pos` (position|target-modality, chance 1/8) is the headline — the honest cross-modal completion
accuracy. Val also logs a **coord-shuffle control** (acc_pos must fall to chance), collapse diagnostics
(effective rank, within-modality off-diagonal cosine), and EMA drift. Gate before 70k: shapes 96/32,
balanced 8/8/8/8, target min-dist ≥ 8 mm, acc_pos≈1/8 at init, shuffle-control at chance, no collapse.
