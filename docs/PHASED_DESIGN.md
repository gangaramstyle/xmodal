# Phased cross-modal SSL — design (agreed)

One continuous A40 run: **self → cross(pixel+match) → latent**, wandb throughout, warm-start
transitions inside the run. Codebase = `xmodal` (clean); data = BraTS-2026 (METS/PED/GoAT).

## Encoder (shared across all phases)
- Input = **2.5D patches + mm-RoPE** (positions relative to the prism center). **No series_embed
  on tokens** — the series identity must be *learned* into the series-CLS token, not injected.
- Applied **per-series / block-diagonal** (≡ series in the batch dim — series never cross-attend
  in the encoder). Each series yields patch encodings + a **series-CLS** + a **view-CLS**.
- Two prisms per patient, **positions X and Y** (co-registered → shared mm frame).
- Encoder objectives keep training in every phase:
  - **view-CLS** — *within a series*, predict relative geometry between its X and Y encodings
    (3 spatial + 2 window signs, BCE; rotation dropped).
  - **series-CLS** — *across patients*, positive = same series (A's T1 ↔ B's T1),
    `rank_hinge_xmod_loss`.

## Self phase
Single-series masked reconstruction (MAE) + view-CLS + series-CLS. Produces the frozen teacher.

## Cross phase (pixel + matching)
Two encoders per step:
- **Frozen phase-self teacher** — extra **no-grad** forward on the *same prisms*, only to read out
  the **per-prism series-CLS** (stable conditioner; a bit noisier than a prototype average, accepted).
- **Online encoder** — warm-started, **keeps training** (encoder objectives + decoder loss flow back).

Decoder:
- **Context** = online patch encodings **fused with the frozen series-CLS** via a small MLP
  (`concat → MLP`), so each token carries "content + which series." Source-heavy (~80%) + a few
  **target-series anchor** patches (the model knows which is which via the fused series-CLS).
- **Queries** = position + **frozen target series-CLS** → "predict the target series here."
- **Losses:**
  - **Pixel** — reconstruct held-out target patches.
  - **Matching** — slots = position + target series-CLS; colors = raw target patches (**blind**:
    no coords, no series-CLS). CLIP-style. **Anchors ∩ masked = ∅** (disjoint — no answer lookup).

## Latent phase
Same wiring; target = the frozen teacher's target-series **latents** instead of pixels (cosine).

## Holdout & contamination (`xmodal/holdout.py`)
- Deterministic md5 patient split — identical on CUBIC / molab / anywhere.
- Explicit manifest of challenge val/test IDs → **excluded from pretraining** (contamination guard,
  asserted before every run).

## Eval (molab, one challenge/subset per notebook)
Frozen encoder → small **seg decoder fine-tune** on labels → predict → **lesion-wise DSC + NSD + F1**
(`xmodal/metrics.py`, 27 mm³ floor, ET/TC/WT) on held-out. METS first (specificity thesis), then PED/GoAT.

## Infra
`--propagate=NONE` (avoid the login CPU-time ulimit), bf16 + torch.compile + fused AdamW, rotating
GPU cache, wandb. Data = local BraTS-26 nii dirs + corrected labels (`data.py` loader, wired to holdout).
