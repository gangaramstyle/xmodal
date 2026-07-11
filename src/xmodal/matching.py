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
    """Blind value encoder: patch contents -> embedding, no positional signal (no coords/RoPE/stem).
    `in_ch` > 1 carries the v3 scan-relative channels (raw + z-score + CDF) so the target is a scan-
    CALIBRATED appearance (interpretation A: normalization, not style injection) — still position-free."""

    def __init__(self, width, voxels, in_ch=1):
        super().__init__()
        self.embed = nn.Conv3d(in_ch, width, tuple(voxels), stride=tuple(voxels))
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, width))

    def forward(self, patches):
        # patches [B,Q,C,V,V,1] (multi-channel) or [B,Q,V,V,1] (single)
        b, q = patches.shape[:2]
        x = patches.reshape(b * q, *patches.shape[2:]) if patches.dim() == 6 else patches.reshape(b * q, 1, *patches.shape[2:])
        z = self.embed(x).reshape(b * q, -1)
        return self.mlp(z).reshape(b, q, -1)


def _ce_last(logits, target):
    """Cross-entropy where `logits` [..., K] retrieve index `target` [...] over the last axis."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def structured_match_loss(slots, colors, n_pos, logit_scale, *, w_pos=1.0, w_mod=0.25):
    """v3 CONDITIONAL matching (review Blocker 2). slots/colors [B, P*M, D] L2-normalized, ordered
    position-major (index = p*M + m; M=4 modalities). Two conditional objectives so the 12x4 structure
    actually trains the model — a global P*M softmax lets modality be eliminated for free (log48->log12
    with zero anatomy learned):
      position|modality: fix modality, retrieve the right POSITION among P (chance 1/P).
      modality|position: fix position, retrieve the right MODALITY among M (chance 1/M).
    Both slot->target and target->slot directions. Returns (loss, metrics)."""
    B, Q, D = slots.shape
    P = n_pos; M = Q // P
    s = logit_scale.exp().clamp(min=1.0, max=100.0)
    zs = slots.reshape(B, P, M, D); hs = colors.reshape(B, P, M, D)
    zp, hp = zs.permute(0, 2, 1, 3), hs.permute(0, 2, 1, 3)              # [B,M,P,D]
    Lp = s * torch.einsum("bmpd,bmqd->bmpq", zp, hp)                     # [B,M,P(query),P(cand)]
    tp = torch.arange(P, device=slots.device).expand(B, M, P)
    loss_pos = 0.5 * (_ce_last(Lp, tp) + _ce_last(Lp.transpose(-1, -2), tp))
    Lm = s * torch.einsum("bpmd,bpnd->bpmn", zs, hs)                     # [B,P,M(query),M(cand)]
    tm = torch.arange(M, device=slots.device).expand(B, P, M)
    loss_mod = 0.5 * (_ce_last(Lm, tm) + _ce_last(Lm.transpose(-1, -2), tm))
    loss = w_pos * loss_pos + w_mod * loss_mod
    with torch.no_grad():
        acc_pos = (Lp.argmax(-1) == tp).float().mean()                  # slot -> target (which position)
        acc_mod = (Lm.argmax(-1) == tm).float().mean()
        acc_pos_t2s = (Lp.transpose(-1, -2).argmax(-1) == tp).float().mean()   # target -> slot
        acc_mod_t2s = (Lm.transpose(-1, -2).argmax(-1) == tm).float().mean()
    return loss, {"loss_pos": float(loss_pos.detach()), "loss_mod": float(loss_mod.detach()),
                  "acc_pos": float(acc_pos), "acc_mod": float(acc_mod),
                  "acc_pos_t2s": float(acc_pos_t2s), "acc_mod_t2s": float(acc_mod_t2s),
                  "chance_pos": 1.0 / P, "chance_mod": 1.0 / M}


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


def modality_completion_loss(slots, colors, n_mod, logit_scale):
    """v4 modality completion (docs/MIXED_V4_DESIGN.md). slots/colors [B, n_mod*n_per, D] L2-normalized,
    ordered MODALITY-major (idx = m*n_per + p). For each target modality, retrieve the right POSITION
    among the n_per targets of THAT modality (chance 1/n_per). Modality can't help — every candidate in
    the comparison is already the requested modality. Both slot->target and target->slot. (loss, met)."""
    B, Q, D = slots.shape
    M = n_mod; P = Q // M
    s = logit_scale.exp().clamp(min=1.0, max=100.0)
    zs = slots.reshape(B, M, P, D); hs = colors.reshape(B, M, P, D)
    L = s * torch.einsum("bmpd,bmqd->bmpq", zs, hs)                      # [B,M,P(query),P(cand)]
    tp = torch.arange(P, device=slots.device).expand(B, M, P)
    loss = 0.5 * (_ce_last(L, tp) + _ce_last(L.transpose(-1, -2), tp))
    with torch.no_grad():
        acc = (L.argmax(-1) == tp).float().mean()
        acc_t2s = (L.transpose(-1, -2).argmax(-1) == tp).float().mean()
    return loss, {"acc_pos": float(acc), "acc_pos_t2s": float(acc_t2s), "chance_pos": 1.0 / P}


def default_log_logit_scale(temperature=0.07):
    return math.log(1.0 / max(temperature, 1e-6))
