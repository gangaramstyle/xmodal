# xmodal v5 — Representation & Segmentation-Quality Findings

_Snapshot of the eval-suite work: readout ladder, seg-query F1, oracle dense-seg (Dice/NSD),
conv smoothing, and the large-pool readout. Numbers are on the 51 held-out METS unless noted._

## Executive summary
We built an in-distribution eval ladder measuring both **representation quality** (frozen-feature
probes) and **segmentation quality** (oracle dense-seg → Dice/NSD vs BraTS-METS SOTA).
Classification-style readouts look strong (macro-F1 up to **0.924**), but true voxel-level
segmentation is **mid — ET Dice ~0.48 vs SOTA 0.752 even with oracle detection**. The gap is
dominated by resolution/representation, not spatial coherence or a metric artifact.

## What we built
- **Readout ladder** (prism-scoped, in-distribution): linear → MLP → seg-query decoder readout;
  linear vs full decoder unfreeze. Confirmed the old "lost task correlation" was a coordinate-range
  OOD artifact of the whole-brain readout, not the model.
- **Seg-query eval** — segmentation as a 5th modality (decoder + seg-token + class head), 3-way macro-F1.
- **Dense-seg eval** — oracle-detected tumor prism → densely classify every 2mm voxel → DSC + NSD
  for ET/TC/WT. Isolates segmentation *quality* from *detection*.
- **Two-stage conv smoothing head** — stage-1 decoder+seg-token+linear on scattered voxels; stage-2
  3D-conv classifier on the frozen dense embeddings (adds the spatial prior the decoder lacks).
- **Large-pool readout** — fine-tune the two-stage readout on 600 `mets_train` patients (test IDs
  excluded by basename), test on the 51 held-out.
- **Ablations in flight** — multi-size ViT-base, seg-task route-A (CE) / route-B (SupCon), half-data
  (patient-frac 0.5). Blur-2 killed as a confirmed wash.

## Results

**Seg-query readout (full decoder unfreeze), 3-way macro-F1:**

| arm | macro-F1 | ET F1 |
|---|---|---|
| both_tumor | **0.924** | 0.863 |
| both | 0.913 | 0.852 |
| mae_tumor | 0.894 | 0.832 |

(frozen decoder: 0.77 / 0.74 / 0.70 → decoder fine-tuning is worth ~+0.15 macro-F1.)

**Dense-seg (oracle), best = ViT-small conv-smooth, GroupKFold within held-out:**

| region | stage-1 linear | stage-2 conv-smooth | SOTA (Junho_zzang) |
|---|---|---|---|
| ET | 0.440 | **0.478** | 0.752 |
| TC | 0.488 | **0.531** | 0.772 |
| WT | 0.591 | **0.630** | 0.722 |
| ET NSD | 0.707 | **0.753** | 0.817 |

## Findings (interpretations — several are tentative pending the multi-scale results)

1. **Longer training still improves the representation.** The 4mm-MLP frozen probe climbs
   0.835→0.867 (10k→60k steps), ET-driven, no saturation. **Not yet tested with the decoder-based
   readouts** (seg-query / dense-seg were only run at 70k) — an open gap.

2. **Tumor-specific vs whole-brain training: no large difference yet** (both_tumor 0.924 vs both
   0.913). Small consistent edge → worth continuing tumor-based training, but it is not a big lever.

3. **MAE shapes the encoder; ordering shapes the decoder; both matter.** MAE-only gives the best
   fine-scale *frozen*-feature MLP (mae_tumor 0.871, a late bloomer), but with the decoder unfrozen,
   ordering+MAE (0.924) beats MAE-only (0.894). _Reasonable interpretation for now; may be falsified
   by the multi-scale runs._

4. **F1 promising, dense predictions less so.** macro-F1 0.924 says the features *know* the class at
   a point; oracle Dice ~0.48 (vs 0.75) says voxel-level segmentation is mid — and this is the
   *optimistic* number (detection handed to us).

**Mechanistic:**
- The decoder predicts each voxel **independently** (pure cross-attention, no query↔query attention)
  → dense inference is count-invariant *and* has no spatial prior.
- The **conv smoother** adds that prior: +0.03–0.08 DSC — but **NSD rose the same amount**, so it
  improved boundaries, not just speckle, and did **not** close the SOTA gap.
- **NSD gap (0.72 vs 0.82) ≪ Dice gap (0.48 vs 0.75)** → boundaries are roughly right; we bleed
  volume overlap on thin/small structures → **resolution-limited**.

**Metric honesty:** macro-F1 (0.924) and Dice (0.46) are *not the same measurement*. Identical
formula for a single binary problem, but the F1 was pooled + macro-averaged (incl. easy background)
over *sampled* points, while Dice is per-prism over *every* voxel. Never compare a macro-F1 to a
SOTA Dice.

## The four open levers (what we still need to resolve)

| Lever | Experiment | Status (jobs) |
|---|---|---|
| (a) More pretrain patients | half-data (patient-frac 0.5) as down-signal; up-signal TBD | running (16316142) |
| (b) Finer resolution | multi-size ViT-base (2/3/4mm), then dense-seg at 2mm/1mm in-distribution | training (16316032); eval pending |
| (c) More fine-tune patients | large-pool readout: 600 mets_train patients, test held-out | running |
| (d) Bake readout into pretraining | seg-task route-A (CE) / route-B (SupCon), ~10% of steps | training (16316057/58) |
| (none) plateau | compare all vs the both_tumor 70k baseline | — |

**No headline arm chosen yet** — we are still exploring the different representations.

## Caveats / variance
- ViT-base vs ViT-small dense-seg flipped between runs (base 0.463 → 0.374, one seed). Treated as
  **variance**, not a capacity finding; a long downstream fine-tune is expected to flip it back.

## Operational
- **CUBIC access:** `ssh -i ~/.ssh/id_ed25519 -p 2020 gangarav@192.168.1.155` (`.0.20` times out;
  hostnames need the VPN/DNS). Jobs run independent of the tunnel — they fetch code from GitHub and
  the held-out set from wandb, so cutting the tunnel only stops monitoring.
- **Bugs fixed along the way:** (1) sampling source points globally then filtering to a prism →
  nearly always empty; restrict-then-sample. (2) conflating query *size embedding* with grid
  *stride* (feeding a 4mm-trained model a 2mm size is OOD). (3) flat `BraTS-*` glob missed nested
  `mets_train/MICCAI-.../BraTS-MET-*` → discover patient dirs by recursive `*-seg.nii.gz`.
- A session **date artifact** (+10 days) once faked a "jobs thrashing for days" scare — verify
  elapsed time via git/log timestamps, not the reported date.
