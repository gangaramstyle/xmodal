# v3: structured 4-way targets + scan-conditioned target encoder

Evolves the mixed-modality objective (`MIXED_MODAL_DESIGN.md`) with the two **low-risk** pieces from the
review, deliberately **without** prototype/cluster prediction (too finicky; separate bet). Both pieces
are behind flags so we can ablate structured-vs-mixed targets and scan-context on/off. The matching
loss (symmetric InfoNCE, `ema_color` arm) is **unchanged**.

Addresses two legitimate critiques of hard instance matching:
- **modality×size shortcut** — the query is handed target modality+size, and both are inferable from
  pixels, so a representation of just "T1/T1c/T2/FLAIR × 4/8/16 mm" beats chance without anatomy.
- **no scan calibration** — the blind `ColorHead` sees a voxel is 0.72 but not where 0.72 sits in the
  scan's tissue histogram (CSF/GM/WM/tumor modes, contrast compression, scanner idiosyncrasy).

---

## 1. Structured 4-way target set  (`--structured-targets`)

Replace the independently-drawn mixed target patches with a **grid**: draw `P` positions (default 12) in
the prism, one fixed size (default 8 mm), and gather **every position in all 4 modalities** → `held = P×4`
targets. Requires complete 4-modality bundles (BraTS standard; incomplete bundles skipped in this mode).

- Modality can't identify position (every position exists in all 4); size can't (one size) — the two
  shortcuts are structurally removed.
- Same-position/different-modality targets are now **always present** (this is `hardneg` taken to full
  4-way), so cross-modal consistency is directly in the loss and directly measurable.
- Exclusion (§4 of the mixed doc) still applies per (position, modality) vs same-series source.

**Curriculum reframe.** With all modalities as targets there is no "target-dominant series," so self↔cross
moves to the **source** side: the source bag's dominant-modality share ramps from **balanced early**
(`src_share_lo`, ~0.3 → every target modality has same-modality context → easy self) to **peaked late**
(`src_share_hi`, ~0.9 → only the dominant modality's targets have context; the other three are genuine
cross-modal prediction). Stochastic per item, floored. Replaces the v2 alignment coin in structured mode.

## 1b. Conditional matching loss — the structure must live in the LOSS  (review Blocker 2)

The 12×4 layout in the sampler is not enough: a **global 48-way InfoNCE** lets modality be eliminated
for free (a T1c query rejects the 36 non-T1c candidates and ties among 12 T1c → loss log48→log12,
top-1 2.1%→8.3%, with *zero* anatomy learned). So the objective is **conditional**
(`structured_match_loss`):

- **position | modality**: fix the modality, retrieve the right POSITION among P (chance **1/P**) —
  weight **1.0**. This is the anatomical-correspondence signal.
- **modality | position**: fix the position, retrieve the right MODALITY among M=4 (chance **1/4**) —
  weight **0.25**. Contrast held fixed on anatomy.
- global 48-way: **0** (may return later at 0.05–0.1 if it helps).

Both slot→target and target→slot directions; EMA uses the two-direction BYOL form (slots ↦ stop-grad
EMA target; online target ↦ stop-grad slots). This is the CAPI flavor (stable target encoder defines
what a masked patch contains) with a **continuous** target — no prototypes, Sinkhorn, or codebook.

## 2. Scan-conditioned target — Version A: scan-RELATIVE channels  (`--scan-context`)

Semantics fixed to **normalization** (review): the target should mean "this patch is bright *relative
to this scan's tissue distribution*". We implement that as deterministic **scan-relative input
channels**, not a learned AdaLN style vector (AdaLN could as easily inject scanner style as remove it).

**Per-scan stats** (position-free, computed once at load on the normalized foreground): 9 percentiles
`[1,5,10,25,50,75,90,95,99]` + mean + std + foreground-fraction + 16-bin histogram = **28 dims** →
`CachedScan.stats`.

**Channels** (`_scan_channels`, deterministic from stats): every patch becomes **3 channels** —
`[raw, robust z-score = (x−median)/IQR, histogram-CDF(x)]`. The stem and the target `ColorHead` take
`in_ch=3`. Both source and target get the same transform (symmetry), so the encoder reasons in scan-
relative space and the target is scan-*invariant*.

**Consequences.** (a) The decoder query needs **no** scan-context (the target is scan-invariant). (b)
There is **no learned scan network on the target side**, so the whole target encoder is EMA'd cleanly —
the review's "EMA is only partially EMA" problem simply doesn't arise. (c) No cross-attention to a scan
thumbnail (position-leak trap) — channels are per-voxel and global-stat-derived.

## 2b. Clean semantic target vs view-specific pixel target (final data-flow fix)

Window jitter is a **student-side** augmentation only. The held patch plays two roles that must not be
the same tensor:
- **Semantic (EMA) target** = the **clean** held patch (`held_semantic`). Jittering it would inject
  random label noise into the BYOL target; a deterministic `content_blur` sets bandwidth, nothing more.
- **Pixel-reconstruction target** = the held patch under **view-A's** window (`held_pixel_target`), so
  MAE is well-posed: predict the held patch in the same intensity domain as the view-A context.

And the scan-relative channels: the **raw** channel is the presented (possibly view-augmented) intensity;
**z-score and CDF come from the clean `reference`**, so calibration is invariant to the augmentation
(the old code computed all three from the jittered tensor → corrupted z, saturated CDF). Data flow:

```
clean scans → visible A: window-A → raw-A channel ;  clean → z/CDF channels   (student, robust)
            → visible B: window-B → raw-B channel ;  clean → z/CDF channels
            → held clean → EMA SEMANTIC target (raw=ref=clean)
                        → window-A → PIXEL recon target
```

Batch fields: `patches_a_raw`/`patches_a_reference`, `patches_b_raw`/`patches_b_reference`,
`held_semantic`, `held_pixel_target`. At eval there is no augmentation → `reference = raw = patches`.

## 3. What's unchanged / removed

Kept: per-patch series (Sites A/B), view-CLS, EMA target, fixed val panels. **Removed** vs the first v3
cut: the global 48-way loss as primary (→ conditional, §1b), the AdaLN `ScanConditionedPatchTeacher` and
`scan_ctx` MLP (→ scan-relative channels), the Dirichlet breadth-mix (→ exact counts, §1), the circular
in-plane exclusion (→ inverted slab geometry, §4). **No** prototypes, Sinkhorn, or cluster prediction.

## 4. Exclusion — inverted + slab geometry  (review pt 3)

Targets placed **first** (only P), then source patches whose **axis-aligned slab** footprint overlaps a
target are redrawn (`_reject_source_overlap`). Slab overlap = both in-plane axes within `(s_src+s_tgt)/2`
AND the thin through-plane axis within ~1 mm — a 2.5D slab is thin, so thick-separated patches don't
overlap even when in-plane-close (the old circular in-plane test massively over-counted). Because only P
small footprints must be avoided, this reaches **overlap_rate = 0** in practice (logged every step;
`val/*/overlap`). Launch gate requires it at 0.

## 5. Build map (as implemented)

1. `sampling.py`: `scan_stats` (28-d) on `CachedScan.stats` (via `data._cpu_payload` / `to_device_scan`);
   structured branch (targets-first, exact counts `_exact_source_series`, slab exclusion, per-patch
   `*_stats`), `overlap_rate` returned.
2. `matching.py`: `structured_match_loss` (position|modality + modality|position); `ColorHead(in_ch)`.
3. `model.py`: `_scan_channels` (raw/z/CDF), `in_ch`, `_structured_match`, EMA-target metrics,
   `ema_drift`, control flags (`drop_source`/`shuffle_coords`), temperature clamp ≥1.
4. `train.py`: structured metrics + fixed panels + controls + drift + overlap logging; git provenance.
5. `run_mixed.py` `--structured --scan-context --assert-v3`; `scripts/cubic/job_mixed_v3.sbatch`.
6. `eval_battery`/`eval_patch_f1`: read `scan_context` from ckpt cfg, cache+pass per-scan `sstats`.

## 6. Launch gate (small run before 70k)  — all must hold

- saved cfg shows `structured/scan_context/ema_color` all true (+ branch/commit).
- every target bag exactly 12×4; **source–target overlap_rate = 0**.
- balanced panel exactly 32/32/32/32; `position|modality` chance = 1/12, `modality|position` = 1/4.
- **controls**: `drop_source` and `shuffle_coords` acc_pos stay at ~1/12 (no context/position shortcut).
- effective rank / within-modality diversity not collapsing; `logit_scale` in [1,100]; EMA drift bounded.
- an eval checkpoint reloads with `scan_context` from cfg and passes stats.

## 7. Ablation plan (once v2 gives a first F1 read)

`{mixed, structured} × {no-scan-ctx, scan-ctx}` at a fixed seed + `ema_color`, isolating each piece's F1
delta; then pick the winner and add seeds.
