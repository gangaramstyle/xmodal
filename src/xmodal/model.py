"""Phase-0 mm-RoPE ViT encoder (self-supervised) — distilled from brats2026 models.

Handles, by construction:
- **2.5D patches**: a patch spec's voxel grid can be a thin slab (v, v, 1); the stem is a
  Conv3d with the spec's kernel, so slab or cube both tokenize to one width-dim token.
- **Variable prism aspect ratios**: purely a sampler concern — the encoder positions tokens
  by their physical-mm coords via mm-RoPE, so any prism box (cubic or not) just works.
- **Variable patch sizes**: one stem conv + one pixel head per patch spec (keyed), so a model
  can embed 4 mm cubes, 8 mm cubes, and 2.5D slabs with size-specific weights.

Phase-0 objective here is masked patch reconstruction (MAE): mask a fraction of patch tokens,
encode with the register/series/view CLS tokens, reconstruct the masked patches' pixels. The
series-CLS / view-CLS tokens are returned for the downstream heads (added later).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    width: int = 384
    depth: int = 12
    heads: int = 6
    mlp_ratio: int = 4
    n_series: int = 8
    n_registers: int = 4
    rope_lambda_min_mm: float = 2.0
    rope_lambda_max_mm: float = 1024.0


# --- mm-RoPE over physical-mm coords --------------------------------------------------
def build_rope(coords, head_dim, lambda_min_mm=2.0, lambda_max_mm=1024.0):
    axes = coords.shape[-1]
    half = head_dim // 2
    if half < axes:
        raise ValueError(f"head_dim={head_dim} too small for {axes} axes")
    per_axis = max(1, half // axes)
    if per_axis == 1:
        lambdas = torch.tensor([lambda_min_mm], device=coords.device, dtype=torch.float32)
    else:
        steps = torch.arange(per_axis, device=coords.device, dtype=torch.float32) / (per_axis - 1)
        lambdas = lambda_min_mm * (lambda_max_mm / lambda_min_mm) ** steps
    freqs = (2.0 * math.pi) / lambdas
    angles = torch.cat([coords[..., a:a + 1] * freqs for a in range(axes)], dim=-1)
    if angles.shape[-1] < half:
        angles = F.pad(angles, (0, half - angles.shape[-1]))
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    dim = x.shape[-1]
    first, second = x[..., : dim // 2], x[..., dim // 2:]
    c, s = (cos[None, None], sin[None, None]) if cos.dim() == 2 else (cos[:, None], sin[:, None])
    return torch.cat([first * c - second * s, first * s + second * c], dim=-1)


class RoPEAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x, cos, sin):
        B, N, W = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, W))


class Block(nn.Module):
    def __init__(self, c: EncoderConfig):
        super().__init__()
        self.n1 = nn.LayerNorm(c.width)
        self.attn = RoPEAttention(c.width, c.heads)
        self.n2 = nn.LayerNorm(c.width)
        self.mlp = nn.Sequential(nn.Linear(c.width, c.mlp_ratio * c.width), nn.GELU(),
                                 nn.Linear(c.mlp_ratio * c.width, c.width))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class Phase0Encoder(nn.Module):
    """mm-RoPE ViT encoder with per-spec patch stems + MAE heads."""

    def __init__(self, cfg: EncoderConfig, patch_specs):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.width // cfg.heads
        # per-spec patch embedding (Conv3d kernel = spec.voxels -> one token) + pixel head
        self.stem = nn.ModuleDict({
            s.key: nn.Conv3d(1, cfg.width, tuple(s.voxels), stride=tuple(s.voxels)) for s in patch_specs
        })
        self.pixel_head = nn.ModuleDict({
            s.key: nn.Linear(cfg.width, int(np.prod(s.voxels))) for s in patch_specs
        })
        self.series_embed = nn.Embedding(cfg.n_series, cfg.width)
        self.series_token = nn.Parameter(torch.randn(cfg.width) * 0.02)
        self.view_token = nn.Parameter(torch.randn(cfg.width) * 0.02)
        self.registers = nn.Parameter(torch.randn(cfg.n_registers, cfg.width) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(cfg.width) * 0.02)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.width)

    def embed(self, patches, spec_key, series_idx):
        """patches [B,n,v0,v1,v2] -> tokens [B,n,W]."""
        B, n = patches.shape[:2]
        fmap = self.stem[spec_key](patches.reshape(B * n, 1, *patches.shape[2:]))
        tok = fmap.reshape(B, n, self.cfg.width)
        return tok + self.series_embed(series_idx)[:, None, :]

    def encode(self, tokens, coords):
        """tokens [B,T,W] with matching coords [B,T,3] (CLS/regs use coord 0) -> [B,T,W]."""
        cos, sin = build_rope(coords, self.head_dim, self.cfg.rope_lambda_min_mm, self.cfg.rope_lambda_max_mm)
        x = tokens
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.norm(x)

    def forward_mae(self, patches, coords, spec, series_idx, *, mask_ratio=0.5):
        """Masked patch reconstruction. patches [B,n,v0,v1,v2], coords [B,n,3]."""
        B, n = patches.shape[:2]
        dev = patches.device
        tok = self.embed(patches, spec.key, series_idx)                       # [B,n,W]
        mask = torch.rand(B, n, device=dev) < mask_ratio                      # [B,n] masked positions
        tok = torch.where(mask[..., None], self.mask_token, tok)
        cls = torch.stack([self.series_token, self.view_token])[None].expand(B, -1, -1)
        regs = self.registers[None].expand(B, -1, -1)
        nreg = 2 + regs.shape[1]
        x = torch.cat([cls, regs, tok], dim=1)
        cc = torch.cat([torch.zeros(B, nreg, 3, device=dev), coords], dim=1)
        x = self.encode(x, cc)
        patch_out = x[:, nreg:]                                               # [B,n,W]
        recon = self.pixel_head[spec.key](patch_out)                         # [B,n,prod(voxels)]
        target = patches.reshape(B, n, -1)
        loss = F.l1_loss(recon[mask], target[mask]) if mask.any() else recon.new_zeros(())
        return dict(loss=loss, series_cls=x[:, 0], view_cls=x[:, 1], n_masked=int(mask.sum()))
