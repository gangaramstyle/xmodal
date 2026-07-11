# Mixed-modality conditioned SSL — design spec

Supersedes the hard self→cross→latent phasing (`PHASED_DESIGN.md`) with a single continuous loop.
Series becomes a **conditioning signal** (like patch-size), bags become **variable mixed-modality**,
and self↔cross becomes a **stochastic alignment curriculum** rather than a phase boundary.

Goal is unchanged: better tumor-characterization patch-F1 (representation quality). This spec is the
plan to review *before* any code lands.

---

## 0. One-paragraph summary

Every patch — encoder token and decoder query alike — self-describes its modality via a learned
series embedding. The encoder bag is a **variable mixture** over the co-registered series of one
patient (e.g. ~90% T1 / rest spread over FLAIR/T1c/T2, proportions sampled). The decoder is asked to
reconstruct **held-out** positions, each tagged with the series we *want* back. A curriculum slowly
**misaligns** the source-dominant series from the target-dominant series: early, you mostly predict
the same modality that dominates context (easy, near mono-modal); late, you predict a modality that
is *scarce* in context (hard, cross-modal). No phases. Latent dropped for this experiment.

---

## 1. Series conditioning — two embedding tables

Today series identity is *dynamic*: a single CLS token (`model.py:153 self.series_token`) is encoded
and its **output** `x[:,0]` (`series_repr`) is the per-prism descriptor, trained by `rank_hinge_xmod`
and fused into the decoder (`fuse_series :364`, cross query `:390`). **We remove all of that.**

Replace with two **static learned tables** (separate — "have" vs "want" are different claims, and
cross-modal translation is the whole job):

```python
# EncoderConfig already has n_series = 8
self.series_in_embed = nn.Embedding(cfg.n_series, cfg.width)   # Site A: "this token IS series S"
self.series_q_embed  = nn.Embedding(cfg.n_series, cfg.width)   # Site B: "produce series S"
```

**Site A — encoder token conditioning** (exact analog of `_add_size`, `model.py:187`
`tok + self._size_emb(sizes)`). Each input patch token also gets `+ series_in_embed[sid]`:

```python
def _add_cond(self, tok, sizes, series_ids):        # replaces _add_size at the call sites
    return tok + self._size_emb(sizes).to(tok.dtype) + self.series_in_embed(series_ids).to(tok.dtype)
```

`series_ids` is now **per-patch** `[B,n]` (not per-prism), because a bag mixes series.

**Site B — decoder query conditioning** (`model.py:263` self, `:390` cross). The query carries the
*target* series it is asking for:

```python
query = (self.query_seed[None, None, :]
         + self._size_emb(held_sizes)
         + self.series_q_embed(target_series_ids)).contiguous()   # [B,m,W]
```

**Removed:** `series_token`, `series_repr`, `rank_hinge_xmod_loss`, `fuse_series`, and the
frozen-teacher series-CLS pass in `forward_cross`. View-CLS (`view_token`, spatial/window BCE) stays —
it is orthogonal to this change.

---

## 2. Variable mixed-modality bags — new **paired** sampler

Co-registration is the enabling property already in the codebase: a bundle is `{series: CachedScan}`
all in the same world-mm frame, so any position can be gathered from any series. The current
`sample_cross_batch_vec` gathers *all* source patches from one series and *all* target patches from
another. The new sampler assigns series **per patch** from a mixture.

We keep **view-CLS**, so the sampler stays **paired** (generalizes `sample_paired_batch`): two bags
`a`, `b` from the same patient with anchors ~log-uniform apart, carrying `rel_targets[B,5]` (3 spatial
orderings + 2 window signs) for the view-CLS head. Bag `a` is the **main** bag (masked encode →
reconstruct held positions = MAE + matching). Bag `b` exists **only** for its `view_token` output that
feeds the relative-spatial/window loss vs `a` — no held task on `b`. Series mixing rides on both.

New `sample_mixed_paired_batch(bundles, *, batch_size, token_count, held_count, source_mix,
target_mix, patch_sizes, prism_choices, orient, rng, device, pair_dist_min, pair_dist_max, win_*)`:

- Per item: pick a bundle; draw two prisms (`a`, `b`) as in `sample_paired_batch` (anchor band +
  `rel_targets`).
- Bag `a`: `token_count` source centers + `held_count` held centers (disjoint positions).
  Bag `b`: `token_count` source centers only.
- **Per-patch series**: source ids `[n]` ~ `Categorical(source_mix)` for both `a` and `b`; held ids
  `[m]` ~ `Categorical(target_mix)` for `a`.
- **Gather grouped by series** (efficiency): for each unique series id present, one `grid_sample`
  against that series' volume for all positions assigned to it, then scatter back into bag order.
  ≤ ~n_series grid_samples per bundle-group instead of 1 — still vectorized, no per-patch loop.
- Held targets are **not** in bag `a`'s encoder set (see §4).

Returns: `patches_a[B,n,V,V,1]`, `coords_a[B,n,3]`, `sizes_a[B,n,3]`, `source_series_a[B,n]`;
`patches_b`, `coords_b`, `sizes_b`, `source_series_b[B,n]`; `held_patches[B,m,V,V,1]`,
`held_coords[B,m,3]`, `held_sizes[B,m,3]`, `target_series[B,m]`; `rel_targets[B,5]`.

`source_mix`/`target_mix` are `[B, n_series]` (per-item) categorical rows supplied by the curriculum
(§3). Because gathering is co-registered, a held position can equal a source position but be drawn
from a *different* series — that is the cross-modal reconstruction at a shared location.

---

## 3. Alignment curriculum — the only "phase" knob

Two per-item distributions, both with one **dominant** series (~90%, sampled, not fixed) and the rest
spread over the remaining series. The curriculum controls whether the two dominants **coincide**.

```python
def sample_mixes(rng, n_series, present, step, total, *, dom_lo=0.7, dom_hi=0.95, floor=0.1):
    # present: series ids available in this bundle
    s_dom = rng.choice(present)
    # alignment ramps DOWN: early -> target dominant == source dominant (aligned/easy)
    align_p = max(floor, 1.0 - step / (0.8 * total))          # 1.0 -> floor over ~80% of training
    aligned = rng.random() < align_p
    t_dom = s_dom if aligned else rng.choice([s for s in present if s != s_dom])
    def mix(dom):
        w = rng.dirichlet(np.ones(len(present)))               # stochastic minor proportions
        p = np.full(n_series, 0.0)
        share = rng.uniform(dom_lo, dom_hi)                    # stochastic dominant share
        for s, wi in zip(present, w): p[s] = (1 - share) * wi
        p[dom] += share
        return p / p.sum()
    return mix(s_dom), mix(t_dom)
```

- **Early**: `align_p≈1` → target-dominant = source-dominant → reconstruct mostly the modality that
  dominates context (near mono-modal, easy).
- **Late**: `align_p→floor` → target-dominant drifts to a series that is *scarce* in the bag →
  genuine cross-modal translation.
- **Floors everywhere**: dominant share < 1 and `align_p ≥ floor > 0`, so (a) every series gets
  gradient from step 0 (no minor-series cold start), and (b) even late there is some easy same-modal
  signal. All proportions and the aligned/misaligned coin are **stochastic per item**.

Tunables: `dom_lo/hi` (how peaked the bag is), `floor`, ramp fraction (`0.8*total`).

---

## 4. Masking invariant (the thing that makes or breaks it)

"Same series early" is about the **distribution**, never about showing the answer. Held/target
patches are **always disjoint from the encoder bag** (the existing held-out mechanism). With series on
both token and query, an overlapping (position, series) target would let the decoder key-lookup the
visible token and copy — InfoNCE → 0, encoder gets ~no gradient. So: source positions and held
positions are drawn disjoint, and a held position that *coincides* spatially with a source position
must differ in series (cross-modal), never be the identical visible patch.

---

## 5. Loss & negatives

Matching InfoNCE (`_match_loss`, `slot_match_loss`) is **unchanged**. `color_head` stays
**series-agnostic** — it embeds blurred raw pixels, and pixel appearance already carries modality
(T1 ≠ FLAIR intensities), so the target is implicitly series-typed without conditioning the head.

Negatives = other held patches in the prism. With mixed target series this now includes
**same-position / different-series** pairs — free hard negatives that force `series_q_embed` to route
the query to modality-specific appearance. Pixel-MAE head kept as aux/viz. Latent path dropped for
this experiment.

`--ema-color` retained as the ablation arm: online vs BYOL-style EMA `color_head` target
(`update_color_ema`, `_match_loss(..., ema=True)`), single shared head across all series.

---

## 6. Training loop

One loop, no phase boundaries:

```
for step in range(total):
    src_mix, tgt_mix = sample_mixes(step, total)                  # per-item [B,n_series]
    batch = sample_mixed_paired_batch(..., src_mix, tgt_mix)
    out   = model.forward_mixed(batch, ema_color=cfg.ema_color)   # encode a -> decode held -> MAE+match; view-CLS from b
    loss  = (mae_weight*out.mae + match_weight*out.match
             + rel_spatial_weight*out.rel_spatial + rel_window_weight*out.rel_window)
    loss.backward(); opt.step()
    if cfg.ema_color: model.update_color_ema(cfg.ema_color_m)
```

`forward_mixed` generalizes `forward_self`+`forward_phase0` (minus series-CLS): encode masked mixed
bag `a` with per-patch `source_series` at Site A; build held-position queries with `target_series` at
Site B; predict held pixels (MAE) + match slots↔colors. Encode bag `b` (`_encode_view`) for its
`view_token`; the view-CLS head takes (`a.view_repr`, `b.view_repr`) → rel-spatial (3) + rel-window
(2) BCE against `rel_targets`. **No** series-CLS, `rank_hinge`, teacher pass, or `fuse`.

---

## 7. Config / CLI (fresh, not a mutation)

`scripts/run_mixed.py` + `scripts/cubic/job_mixed.sbatch`. New TrainConfig fields:
`total_steps`, `held_count`, `align_ramp_frac=0.8`, `align_floor=0.1`, `dom_lo=0.7`, `dom_hi=0.95`,
`ema_color`, `ema_color_m=0.996`, `seed`. Two arms only: `--ema-color` on/off, paired seeds
(0/1/2) with `torch.manual_seed` for a clean A/B.

---

## 8. Build checklist

1. `model.py`: add `series_in_embed`/`series_q_embed`; `_add_cond`; `forward_mixed` (keeps view-CLS
   path); strip series-CLS / `series_repr` / `rank_hinge` / `fuse_series` / teacher-series pass.
2. `sampling.py`: `sample_mixed_paired_batch` (per-patch series, group-by-series gather, paired views)
   + `sample_mixes`.
3. `train.py`: mixed loop, curriculum wiring, drop phase/latent branches, keep view-CLS + EMA step.
4. `losses.py`: `rank_hinge_xmod_loss` now unused (leave or delete); view-CLS BCE stays.
5. `scripts/run_mixed.py`, `scripts/cubic/job_mixed.sbatch`.
6. Eval: reuse `eval_battery.py` unchanged (readout is on the frozen encoder; conditioning only
   changes training).

## 9. Open risks

- **Copy shortcut** if held/source overlap in (position, series) — enforced disjoint in §4.
- **Group-by-series gather** adds ≤ n_series grid_samples per group; profile vs single — expected fine.
- **View-CLS retained**: bag `b` mixes series too; the rel-spatial/window targets depend only on the
  two anchors, so mixing is orthogonal and view-CLS is unaffected.
