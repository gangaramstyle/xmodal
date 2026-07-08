# Cross-Modal SSL — Code Catalog & Cleanup Plan

A map of every code/data strain for the cross-modal self-supervised MRI project, why
each exists, what's canonical vs. superseded vs. data, and a plan to consolidate the
pieces that worked into this repo (`~/xmodal`).

> **Scope:** this catalogs the **local (Mac) copies** at `/Users/vineethgangaram/`.
> The actual **datasets, checkpoints, and run outputs live on the Betty cluster**
> (`/vast/projects/witschey/corlab-foundational-mode/people/gangaram/...`, currently
> down) and are **not** in any of these repos by design. Nothing is on CUBIC yet.

---

## 1. Lineage (how the ideas flowed)

```
siglip2/                 brats-xmodal/templates/         brats2026/                 tcia_crossmodal/brats2026/
(experimentation)   →    (first "clean" reimpl)     →   (canonical production)  →  (this thread's fork)
 - SigLIP2 notebook       - phase0_clean recipe          - brats26 package          - phase 2/3/4 cross-modal
 - xmodal notebook        - src/xmodal package           - cls_naflex trainer       - forward_cross(_latent)
 - BRATS2026_DESIGN.md  ──────────────── design doc carried forward ────────────►    - freeze/holdout ladder
 - crossmodal_summary   ──────────────── results memory carried forward ─────────►   - ET-from-T1 probe
```

- **`siglip2`** = genesis: pedagogical SigLIP2 (natural images) + the original brain-MRI
  cross-modal notebook + the master design doc. Superseded as code, invaluable as record.
- **`brats-xmodal/templates`** = first from-scratch reimplementation (`phase0_clean`).
  Predecessor to `brats2026`; likely archive once its unique kernels are confirmed ported.
- **`brats2026`** = the **canonical production repo** (installable `brats26` package, tests, docs).
- **`tcia_crossmodal/brats2026`** = a plain rsync'd **fork** where all of *this thread's*
  cross-modal work (phases 2–4, frozen/holdout ladder, latent prediction) was built. Not
  git-tracked; needs upstreaming.

---

## 2. Repo-by-repo inventory

### Canonical code

| Repo | Size | Git | Role |
|---|---|---|---|
| **`brats2026`** | 47M | yes (branch `pmbb-pretrain`) | **Canonical** production codebase. |
| **`tcia_crossmodal/brats2026`** | 1.6M src | no (rsync copy) | **Active fork** — this thread's cross-modal code. Upstream it. |

**`brats2026` package (`src/brats26/`):**
- `models/` — `matching.py` (slot-match "predict position of color" head + `slot_match_loss`), `series.py` (MRI series taxonomy), `vit.py` (RoPE ViT + cross-attn predictor), `stem.py` (3D patch stem + series embed), `rope.py` (mm-RoPE), `factory.py`/`config.py`.
- `train/` — **`cls_naflex.py` (the main trainer, ~2850 lines): NaFlex view-CLS + series-CLS + patch-embed pretraining, MIM objectives, rotating GPU cache**; `ct_naflex.py` (NLST chest CT), `phase0_smoke.py`, `t1_t1c_plane_mim.py`.
- `eval/` — `cls_probes.py` (CLS probes), `band_slice_sheets.py`, `naflex_shape_grid.py`.
- `data/` — `exam_store.py` (exam-store contract), `rotating_cache.py` (jittered GPU cache).
- `losses/` — `ot.py` (Sinkhorn + dustbin OT), `contrastive.py` (rank-hinge / series-CLS).
- `sampling/` — `physical.py` (mm-space prism + patch-bag sampler).
- `ct/`, `pmbb/` — CT (NLST) and PMBB series-pretraining strains.
- `experiments/manifest.py`, `ops/safety.py`.
- **Docs:** `BRATS2026_DESIGN.md` (master blueprint, K1–K9 kernels, H1–H11, S0–S5 DAG), `crossmodal_ssl_summary.md` (prediction→matching pivot), `AGENTS.md`, `BETTY_HANDOFF.md`, `TOOLS_AND_PROCESS.md`, `docs/` (BETTY_PATHS, MATCHING_REFRAME_BRIEF, PHASE0_PHASE1_TRIALS_SPEC).
- **`scripts/betty/`** — parametrized sbatch launchers + `sync_to_betty.sh`, hygiene check.

**`tcia_crossmodal/brats2026` — what's diverged (the upstreaming target):**
- **`train/cls_naflex.py`** (2849 → 3215 lines): added `match`/`both` objectives + `ColorHead`/`match_slot_proj`/`match_logit_scale`; `forward_cross` (phase-2 T1→T1c recon); `forward_cross_latent` (phase-4 JEPA latent target); `build_patient_modalities`/`make_bundle_loader`/`sample_cross_all_batch` (`cross_all` any-pair data path); args `--mim-mode cross_all`, `--pairs-per-patient`, `--phase2-cross-recon`, `--anchor-frac`, `--freeze-encoder`, `--latent-target`, `--match-weight`, `--compile`, `cosine_warmup_lr`.
- **New `eval/` modules:** `probe_et_t1.py` (**ET-from-T1 specificity probe — the key metric**), `ablate_source.py` (decoder-shortcut test), `viz_cross.py` (phase-2 recon GIF), `viz_latent.py` (phase-4 latent-match GIF), `tumor_probe.py`, `viz_matching.py`, `viz_reconstruct.py`.
- **New sbatch:** `job_phase2/3/4.sbatch`, `job_phase3_frozen/frozrand.sbatch`, `job_p2_frozen/_rand.sbatch`, `job_gpu_eval.sbatch`, `job_et_probe.sbatch`, `job_tcia_*`.
- **Fix:** `eval/cls_probes.py` strips `_orig_mod.` so `torch.compile` checkpoints load.
- **Root tests** (move into `tests/` when upstreaming): `tests_cls_naflex_objectives.py`, `tests_forward_cross.py`.
- `matching.py` + `phase0_smoke.py` are **byte-identical** to canonical — already upstreamed.

### Experimentation / predecessors (mine, then archive)

| Repo | Size | Role |
|---|---|---|
| **`siglip2`** | 11G (99% = `yale/` data) | Genesis notebooks + design docs. **Preserve the 3 `.md` docs + `xmodal_notebook_backup.py` + `mm_run.json`; archive the SigLIP2 natural-image notebook; move `yale/` to a data store; delete secrets (`.wandb-key`, `.molab-*`).** |
| **`brats-xmodal/templates`** | 2.2M | First clean reimpl (`phase0_clean`, `src/xmodal`). Predecessor to `brats2026`; confirm kernels ported, then archive. |

**`siglip2` specifics:**
- `siglip2_notebook_backup.py` — from-scratch SigLIP/SigLIP2 + dense-SSL (CAPI/DINOv3/FRANCA) on natural images. Pedagogical; ideas live in the design doc. Archive code.
- `xmodal_notebook_backup.py` — **the ORIGINAL brain-MRI cross-modal prototype**: mm-RoPE ViT, dustbin-OT matcher, `train_dedicated`/`train_combined`/`train_latent` (self vs cross vs data2vec-latent). Mine before deleting — the matching + latent ideas started here.
- `mm_run.json` — recorded self-vs-cross R²/probe curves (the evidence cross > self).
- `FINAL_REPORT.md` (SigLIP2 results), `crossmodal_ssl_summary.md` (MRI results), `BRATS2026_DESIGN.md` (bridge to production).

### Data & output (not code — relocate/keep out of code repos)

| Path | Role |
|---|---|
| **`pmbb/`** (131M) | "pmbb-pretrain" = **PMBB key-image DATA** (~506 PNG organ crops + CSV manifests). No code. Candidate CT/body pretraining corpus. |
| **`siglip2/yale/`** (11G) | TCIA **Yale-Brain-Mets-Longitudinal** raw MRI (400 patients, longitudinal). Move to data store. |
| **`brats2026_visual_inspection/`** (1.3M) | QC **image output** (band slices, T1/T1c planes) from the brats2026 pipeline. No code. |

### Tooling / stubs (ignore or delete)

| Path | Role |
|---|---|
| `marimo-xmodal/` | **Empty** (0 bytes). Intended dashboard, never built. |
| `marimo-mcp/` | Generic marimo+MCP authoring scaffold. Not project-specific. |
| `marimo_fastapi_test/` | marimo-in-FastAPI PoC (DICOM viewer). Throwaway. |
| `tcia/` | 4K stub (working dir). |

---

## 3. What worked (this thread's leak-free findings)

These are the results the clean implementation should be built around:

1. **Cross-modal recon + matching ("both") is the objective.** Source patches (95%) + a few
   target anchors (5%) → reconstruct the target modality (pixel MAE **and** slot-matching).
2. **Held-out ET-vs-physiologic specificity ≈ 0.97 on unseen mets**, stable across training —
   the T1 representation distinguishes tumor-enhancement from vascular enhancement. This is
   the real, leak-free result (the `probe_et_t1.py` probe).
3. **Leak-free ladder verdict:** pretraining is *necessary and sufficient-frozen*; encoder
   finetuning adds nothing on held-out specificity (full ≈ frozen-pretrained ≫ frozen-random).
4. **Phase-4 latent cross-prediction works** (~0.86 cosine to the frozen teacher's target
   latents) — a JEPA-style alternative to pixel targets.
5. **Anti-patterns (documented dead ends):** prediction-*residual* anomaly is anti-specific
   (≈0.38); training `match_acc` on frozen encoders is leaky (memorization) — always evaluate
   on **held-out** patients; the "tumor-as-anomaly" signal was a sampling artifact.

---

## 4. Cleanup / abstraction plan

**Target:** this repo (`~/xmodal`) = one minimal, clean, tested implementation of the pieces
that worked, plus this catalog. Proposed layout:

```
xmodal/
  docs/CATALOG.md            # this file
  src/xmodal/
    data/        exam_store + rotating GPU cache          (from brats26.data)
    sampling/    mm-space prism / patch-bag sampler        (from brats26.sampling.physical)
    models/      mm-RoPE ViT encoder + cross-attn decoder  (from brats26.models.vit/rope/stem)
                 series/view CLS heads; slot-match head    (from brats26.models.matching, series)
    train/       one trainer: phase0 (self) + cross (2/3) + latent (4)   (distilled cls_naflex.py)
    eval/        et_from_t1 probe; frozen/holdout ladder; ablate_source  (the leak-free evals)
    viz/         cross-recon + latent-match GIFs
  scripts/       one parametrized sbatch launcher + sync
  tests/         objectives, forward_cross, shapes
```

**Steps:**
1. Upstream the fork's cross-modal additions (§2) into a clean trainer — but **strip** the
   objectives we've retired (see open questions below).
2. Port the data/sampler/model kernels from `brats26` (canonical), not the fork (identical).
3. Keep the leak-free eval methodology as first-class (`et_from_t1`, holdout, ablate_source).
4. Archive `siglip2` (docs first), `brats-xmodal` (after kernel-port check); relocate `yale/`,
   `pmbb/`, `brats2026_visual_inspection/` to a data store.

### Finalized build decisions (locked)

**Base:** distill from **`ct_naflex.py`** (1016 lines, cleaner) even for the MRI work — it
already has the efficient vectorized sampler and multi-axis handling we want.

**KEEP (proven, must carry over):**
- **Efficient vectorized GPU sampling** — the whole efficiency-forward path (`sample_batch_ct_vec`
  + `draw_*_prism_gpu` + `phys_to_vox_bag_gpu` + `sample_patches_group`). GPU runtime is the #1 cost.
- **CLIP-style position→patch matching** — `models/matching.py` (`ColorHead`, `slot_match_loss`)
  + the `match`/`both` objective branches.
- **3D mm-RoPE** — `models/rope.py`, `vit.py`.
- **Native patch loading, 2.5D-style patches** — `thick_axis` + `patch_offsets_tensor` per-scan.
- **Multi-axis loading** (axial/coronal/sagittal via `plane_id` from affine geometry) +
  **multi-axis series-CLS** (`--series-multi`).
- **Objectives:** phase-0 self (view-CLS + series-CLS + patch-MAE); cross-modal **`both`**
  (MAE + matching, `forward_cross`); **latent** cross-prediction (`forward_cross_latent`).
- **Leak-free eval methodology:** `probe_et_t1` (ET-vs-physio specificity), holdout ladder,
  `ablate_source`.

**DROP (do not carry over):**
- **Sinkhorn path + band matching** entirely — `losses/ot.py`, and the `band_ce` / `band_ot` /
  `band_dustbin_ot` objectives + `band_heads`. Documented dead ends; archive in `siglip2` docs only.

**How:** distill fresh, copy the proven code verbatim where it's earned its place, and iterate
in a **molab notebook** (interactive) before committing to the package.

**Lift map (source → xmodal):**
| xmodal piece | source |
|---|---|
| vectorized sampler, plane/thick/axis handling, series-multi | `brats26/train/ct_naflex.py` |
| mm-RoPE ViT encoder + cross-attn decoder, stem | `brats26/models/{rope,vit,stem}.py` |
| CLIP position→patch matching head | `brats26/models/matching.py` |
| exam-store + rotating GPU cache | `brats26/data/{exam_store,rotating_cache}.py` |
| `forward_cross`, `forward_cross_latent`, `cross_all` path, `--freeze-encoder`, `--latent-target` | `tcia_crossmodal/brats2026/train/cls_naflex.py` |
| ET-from-T1 probe, holdout ladder, ablate_source, viz_cross/viz_latent | `tcia_crossmodal/brats2026/eval/*` |
