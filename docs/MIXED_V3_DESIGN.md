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

## 2. Scan-conditioned target encoder — Version A, symmetric  (`--scan-context`)

**Per-scan stats** (deterministic, position-free, computed once at load on the normalized foreground):
9 percentiles `[1,5,10,25,50,75,90,95,99]` + mean + std + foreground-fraction + 16-bin histogram = **28
dims** → `CachedScan.stats`. This is exactly the histogram calibration the critique asks for and, being
global, cannot reveal any patch's location.

**Context vector** `s = G(stats)` — a small MLP `28→W→W`. `s` is **per-patch** (each patch's stats come
from its own modality's scan).

**Target teacher** `ScanConditionedPatchTeacher` replaces `ColorHead`: same blind conv over patch pixels,
then **AdaLN** modulation `γ(s)⊙LN(f)+β(s)` — a global calibration knob that cannot localize the patch
(no cross-attention to spatial scan tokens; that path is a position-leak trap and is avoided). `s` is
shared across all patches of a scan, so it calibrates appearance but cannot distinguish the P positions.

**Symmetry (the review's correction).** The main encoder **also** receives `s` — added to each token
alongside size+series (`_add_cond`). Otherwise the encoder must predict a scan-calibrated target from
information it structurally can't see → irreducible error. Both sides scan-calibrated → the gap closes.

## 3. What's unchanged

Matching InfoNCE + `ema_color` arm; per-patch series conditioning (Sites A/B); view-CLS; exclusion;
fixed val panels + `match_breakdown`. **No** prototypes, Sinkhorn, or cluster prediction.

## 4. Build map

1. `data.py`/`sampling.py`: `_scan_stats(voln, keep)` (28-d); plumb through `_cpu_payload`/`to_device_scan`
   → `CachedScan.stats`.
2. `sampling.py`: structured-target branch in `sample_mixed_paired_batch` (`P` positions × 4 modalities,
   one size, source-breadth curriculum) + emit per-patch `*_stats` tensors.
3. `matching.py`: `ScanConditionedPatchTeacher` (conv + AdaLN(γ,β from s)).
4. `model.py`: `scan_ctx` MLP (`G`); `_add_cond` adds `G(stats)`; `forward_mixed` uses the teacher +
   passes held stats; `teacher_readout` accepts stats (eval symmetry).
5. `train.py`/`run_mixed.py`: `--structured-targets`, `--target-positions`, `--target-size`,
   `--scan-context`, `src_share_lo/hi`; wire configs.
6. `eval_battery`/`eval_patch_f1`: pass per-scan stats when `--scan-context`.

## 5. Ablation plan (2×2, once v2 gives a first F1 read)

`{mixed, structured} × {no-scan-ctx, scan-ctx}` at a fixed seed + `ema_color`, so each piece's F1 delta
is isolated. Then pick the winner and add seeds.

## 6. Open risks

- Structured mode assumes complete 4-modality bundles; incomplete ones are skipped (log the count).
- Scan-stats are computed on the already-normalized volume, so they capture histogram *shape* not raw
  units — intended (shape carries the tissue modes; raw units are the nuisance we already removed).
- `held = P×4 = 48` matches v2's `held_count`, so memory/throughput are ~unchanged.
