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
    """Blind value encoder: patch contents -> embedding, no positional signal."""

    def __init__(self, width, voxels):
        super().__init__()
        self.embed = nn.Conv3d(1, width, tuple(voxels), stride=tuple(voxels))
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, width))

    def forward(self, patches):
        b, q = patches.shape[:2]
        x = patches.reshape(b * q, 1, *patches.shape[2:])
        z = self.embed(x).reshape(b * q, -1)
        return self.mlp(z).reshape(b, q, -1)


def blur_contents(patches, kernel):
    if kernel is None or kernel <= 1:
        return patches
    b, q = patches.shape[:2]
    x = patches.reshape(b * q, 1, *patches.shape[2:])
    x = F.avg_pool3d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    return x.reshape(b, q, *patches.shape[2:])


def slot_match_loss(slots, colors, logit_scale):
    """slots, colors: [B,Q,D] L2-normalized, order-matched. Symmetric InfoNCE, identity targets."""
    scale = logit_scale.exp().clamp(max=100.0)
    logits = scale * torch.einsum("bqd,bkd->bqk", slots, colors)   # [B,Q,Q]
    b, q, _ = logits.shape
    target = torch.arange(q, device=logits.device).expand(b, q)
    loss = 0.5 * (F.cross_entropy(logits.reshape(b * q, q), target.reshape(-1))
                  + F.cross_entropy(logits.transpose(1, 2).reshape(b * q, q), target.reshape(-1)))
    with torch.no_grad():
        acc = (logits.argmax(dim=1) == target).float().mean()
    return loss, {"match_acc": float(acc), "match_chance": 1.0 / max(q, 1)}


def default_log_logit_scale(temperature=0.07):
    return math.log(1.0 / max(temperature, 1e-6))
