"""CLIP-style position->patch matching ("predict position of color").

SLOTS  = held-out patch *positions* (the decoder queries, RoPE-positioned, context-aware).
COLORS = held-out patch *contents*, embedded BLIND by `ColorHead` — its own conv, NO
         coordinates, NO RoPE, NO series/stem (so the color path carries no positional leak).
Alignment = cosine similarity, symmetric InfoNCE with identity targets (color j <-> slot j),
matched WITHIN each prism bag.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorHead(nn.Module):
    """Blind value encoder: patch contents -> embedding, no positional signal. `s` (scan-context) is
    accepted for interface-compat with ScanConditionedPatchTeacher and IGNORED."""

    def __init__(self, width, voxels):
        super().__init__()
        self.embed = nn.Conv3d(1, width, tuple(voxels), stride=tuple(voxels))
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, width))

    def forward(self, patches, s=None):
        b, q = patches.shape[:2]
        x = patches.reshape(b * q, 1, *patches.shape[2:])
        z = self.embed(x).reshape(b * q, -1)
        return self.mlp(z).reshape(b, q, -1)


class ScanConditionedPatchTeacher(nn.Module):
    """Blind value encoder + scan-relative calibration (v3 §2). Same conv over patch pixels as ColorHead
    (no coordinates/RoPE/stem -> no positional leak), then AdaLN modulation by a per-patch scan-context
    vector `s` = G(scan stats). The scan histogram calibrates appearance (where the intensity sits among
    the tissue modes) but, being global to the scan, cannot distinguish patches by location. Injected via
    AdaLN, never cross-attention, so the target patch cannot localize itself in a scan map."""

    def __init__(self, width, voxels):
        super().__init__()
        self.embed = nn.Conv3d(1, width, tuple(voxels), stride=tuple(voxels))
        self.norm = nn.LayerNorm(width, elementwise_affine=False)     # AdaLN: affine comes from s
        self.to_scale_shift = nn.Linear(width, 2 * width)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        nn.init.zeros_(self.to_scale_shift.weight)                   # start at identity (gamma=beta=0) ->
        nn.init.zeros_(self.to_scale_shift.bias)                     # ~plain blind encoder, calibration grows

    def forward(self, patches, s=None):
        b, q = patches.shape[:2]
        z = self.embed(patches.reshape(b * q, 1, *patches.shape[2:])).reshape(b, q, -1)   # [B,Q,W]
        h = self.norm(z)
        if s is not None:
            gamma, beta = self.to_scale_shift(s).chunk(2, dim=-1)
            h = h * (1 + gamma) + beta                              # AdaLN(z; s)
        return self.mlp(h)


def blur_contents(patches, kernel):
    if kernel is None or kernel <= 1:
        return patches
    b, q = patches.shape[:2]
    x = patches.reshape(b * q, 1, *patches.shape[2:])
    ks = tuple(min(kernel, d) for d in x.shape[2:])   # cap per-dim so 2.5D thin axis (size 1) -> kernel 1
    pad = tuple(k // 2 for k in ks)
    x = F.avg_pool3d(x, kernel_size=ks, stride=1, padding=pad)
    return x.reshape(b, q, *patches.shape[2:])


def slot_match_loss(slots, colors, logit_scale, soft_tau=None, soft_feat=None):
    """slots, colors: [B,Q,D] L2-normalized, order-matched. Symmetric InfoNCE.
    soft_tau=None -> hard identity targets. soft_tau>0 -> SIMILARITY-SOFTENED targets: the target for
    slot q is softmax(similarity / soft_tau) over the slots, so genuinely-interchangeable patches SHARE
    the positive mass (confusing them is barely penalized) while far confusions cost full price.
    `soft_feat` [B,Q,F] chooses WHERE that similarity comes from:
      None  -> use `colors` (the trainable color_head) = MODEL-based -> circular / collapse-prone.
      given -> a FIXED, non-trainable descriptor (e.g. raw blurred pixels) = non-circular, collapse-safe
               (the target is external, so collapsing colors only RAISES the loss)."""
    scale = logit_scale.exp().clamp(max=100.0)
    logits = scale * torch.einsum("bqd,bkd->bqk", slots, colors)   # [B,Q,Q]
    b, q, _ = logits.shape
    target = torch.arange(q, device=logits.device).expand(b, q)
    if soft_tau is None:
        loss = 0.5 * (F.cross_entropy(logits.reshape(b * q, q), target.reshape(-1))
                      + F.cross_entropy(logits.transpose(1, 2).reshape(b * q, q), target.reshape(-1)))
    else:
        with torch.no_grad():
            if soft_feat is not None:                            # FIXED raw pixels: RBF (Euclidean), intensity-aware.
                d2 = torch.cdist(soft_feat, soft_feat) ** 2      # [B,Q,Q]; cosine would over-pool flat patches
                med = d2.flatten(1).median(1).values.clamp_min(1e-6)
                sim = -d2 / med[:, None, None]                   # neg normalized sq-dist; self=0=max
            else:                                                # trainable color_head: cosine
                f = F.normalize(colors, dim=-1); sim = torch.einsum("bqd,bkd->bqk", f, f)
            soft = F.softmax(sim / soft_tau, dim=-1)
        lp = F.log_softmax(logits, dim=-1); lpt = F.log_softmax(logits.transpose(1, 2), dim=-1)
        loss = 0.5 * (-(soft * lp).sum(-1).mean() - (soft * lpt).sum(-1).mean())
    with torch.no_grad():
        acc = (logits.argmax(dim=1) == target).float().mean()
    return loss, {"match_acc": float(acc), "match_chance": 1.0 / max(q, 1)}


def default_log_logit_scale(temperature=0.07):
    return math.log(1.0 / max(temperature, 1e-6))
