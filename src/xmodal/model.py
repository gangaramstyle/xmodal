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

from xmodal.matching import ColorHead, default_log_logit_scale, slot_match_loss


@dataclass
class EncoderConfig:
    width: int = 384
    depth: int = 12
    heads: int = 6
    mlp_ratio: int = 4
    n_series: int = 8
    n_registers: int = 4
    decoder_depth: int = 4
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


class CrossAttentionBlock(nn.Module):
    """Decoder block: queries cross-attend into an encoded context, RoPE on both sides."""

    def __init__(self, c: EncoderConfig):
        super().__init__()
        self.heads, self.head_dim = c.heads, c.width // c.heads
        self.nq = nn.LayerNorm(c.width)
        self.nkv = nn.LayerNorm(c.width)
        self.q = nn.Linear(c.width, c.width)
        self.kv = nn.Linear(c.width, 2 * c.width)
        self.proj = nn.Linear(c.width, c.width)
        self.n2 = nn.LayerNorm(c.width)
        self.mlp = nn.Sequential(nn.Linear(c.width, c.mlp_ratio * c.width), nn.GELU(),
                                 nn.Linear(c.mlp_ratio * c.width, c.width))
        self._rope = (c.rope_lambda_min_mm, c.rope_lambda_max_mm)

    def forward(self, q_tok, ctx_tok, q_coords, ctx_coords):
        B, Nq, W = q_tok.shape
        Nc = ctx_tok.shape[1]
        q = self.q(self.nq(q_tok)).reshape(B, Nq, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(self.nkv(ctx_tok)).reshape(B, Nc, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        qc, qs = build_rope(q_coords, self.head_dim, *self._rope)
        kc, ks = build_rope(ctx_coords, self.head_dim, *self._rope)
        q, k = apply_rope(q, qc, qs), apply_rope(k, kc, ks)
        o = F.scaled_dot_product_attention(q, k, v)
        q_tok = q_tok + self.proj(o.transpose(1, 2).reshape(B, Nq, W))
        return q_tok + self.mlp(self.n2(q_tok))


class Phase0Encoder(nn.Module):
    """mm-RoPE ViT encoder + cross-attention decoder. Objectives: phase-0 self MAE
    (`forward_mae`), cross-modal recon+matching (`forward_cross`), latent cross-prediction
    (`forward_cross_latent`)."""

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
        # cross-modal decoder fuses each patch encoding with its (frozen-teacher) series-CLS so the
        # decoder can tell sources apart (no series_embed lookup — series identity is the CLS output).
        self.fuse = nn.Sequential(nn.Linear(2 * cfg.width, cfg.width), nn.GELU(),
                                  nn.Linear(cfg.width, cfg.width))
        self.series_token = nn.Parameter(torch.randn(cfg.width) * 0.02)
        self.view_token = nn.Parameter(torch.randn(cfg.width) * 0.02)
        self.registers = nn.Parameter(torch.randn(cfg.n_registers, cfg.width) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(cfg.width) * 0.02)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.width)
        # cross-modal decoder + heads
        self.decoder = nn.ModuleList(CrossAttentionBlock(cfg) for _ in range(cfg.decoder_depth))
        self.query_seed = nn.Parameter(torch.zeros(cfg.width))
        self.dec_pixel_head = nn.ModuleDict({
            s.key: nn.Linear(cfg.width, int(np.prod(s.voxels))) for s in patch_specs})
        self.match_slot_proj = nn.Linear(cfg.width, cfg.width)
        self.match_logit_scale = nn.Parameter(torch.tensor(default_log_logit_scale(0.07)))
        self.color_head = nn.ModuleDict({s.key: ColorHead(cfg.width, s.voxels) for s in patch_specs})
        self.latent_head = nn.Linear(cfg.width, cfg.width)
        # view-CLS head: 3 spatial-ordering + 2 window signs (rotation dropped) = 5-way BCE
        self.rel_view_head = nn.Sequential(nn.Linear(2 * cfg.width, cfg.width), nn.GELU(),
                                           nn.Linear(cfg.width, 5))

    def embed(self, patches, spec_key, series_idx=None):
        """patches [B,n,v0,v1,v2] -> tokens [B,n,W]. NO series_embed added — series identity is
        *learned* into the series-CLS token from content, not injected into every patch token
        (else series-CLS is trivialized). Conditioning happens later via the series-CLS output.
        `series_idx` kept for call-site compatibility but ignored (see PHASED_DESIGN.md)."""
        B, n = patches.shape[:2]
        fmap = self.stem[spec_key](patches.reshape(B * n, 1, *patches.shape[2:]))
        return fmap.reshape(B, n, self.cfg.width)

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

    # ---- phase-0 self: decoder MAE + matching + series-CLS + view-CLS -----------------
    def _gather_patches(self, patches, idx):
        """Gather patches [B,n,v0,v1,v2] at idx [B,k] -> [B,k,v0,v1,v2]."""
        rest = patches.shape[2:]
        return patches.gather(1, idx.view(idx.shape[0], idx.shape[1], *([1] * len(rest))).expand(-1, -1, *rest))

    def _readout(self, patches, coords, spec, *, mask_ratio):
        """Encode a (masked) prism -> (series_repr[B,W], view_repr[B,W]) for the 2nd view's CLS heads."""
        B, n = patches.shape[:2]; dev = patches.device
        tok = self.embed(patches, spec.key)
        mask = torch.rand(B, n, device=dev) < mask_ratio
        tok = torch.where(mask[..., None], self.mask_token, tok)
        toks, ccs = self._context([tok], [coords], dev, B)
        x = self.encode(toks, ccs)
        return x[:, 0], x[:, 1]

    def forward_self(self, patches, coords, spec, *, mask_ratio=0.5):
        """Self decoder task: hold out `mask_ratio` of patches; encode with them masked; the DECODER
        predicts the held-out pixels (MAE) AND matches held-out position <-> blind content (matching).
        Context = encoder output at VISIBLE positions only (held carry mask_token -> no content leak);
        held (colors) are disjoint from the context. Returns mae, match, match_acc + CLS readouts."""
        B, n = patches.shape[:2]; dev = patches.device; W = self.cfg.width
        nreg = 2 + self.registers.shape[0]
        perm = torch.argsort(torch.rand(B, n, device=dev), dim=1)
        n_held = max(1, min(n - 1, int(round(n * mask_ratio))))
        held, vis = perm[:, :n_held], perm[:, n_held:]
        tok = self.embed(patches, spec.key)
        tok = tok.scatter(1, held[..., None].expand(-1, -1, W), self.mask_token.expand(B, n_held, W))
        x = self.encode(*self._context([tok], [coords], dev, B))
        patch_enc = x[:, nreg:]
        vis_ctx = self._gather(patch_enc, vis, W); vis_coords = self._gather(coords, vis, 3)
        ctx, cc = self._context([vis_ctx], [vis_coords], dev, B)               # decoder context = visible only
        held_coords = self._gather(coords, held, 3)
        query = self.query_seed[None, None, :].expand(B, n_held, W).contiguous()
        query = self._decode(query, ctx, cc, held_coords)
        pv = int(np.prod(spec.voxels))
        held_patches = self._gather_patches(patches, held)                     # [B,n_held,v,v,v]
        mae = F.l1_loss(self.dec_pixel_head[spec.key](query), held_patches.reshape(B, n_held, pv))
        slots = F.normalize(self.match_slot_proj(query), dim=-1)
        colors = F.normalize(self.color_head[spec.key](held_patches), dim=-1)
        m_loss, m_met = slot_match_loss(slots, colors, self.match_logit_scale)
        return dict(mae=mae, match=m_loss, match_acc=m_met["match_acc"],
                    series_repr=x[:, 0], view_repr=x[:, 1])

    def forward_phase0(self, batch, spec, *, mask_ratio=0.5, mae_weight=0.25, match_weight=1.0,
                       series_weight=1.0, rel_spatial_weight=0.25, rel_window_weight=0.25, n_xmod=1):
        """Phase-0 self: decoder MAE + matching (view a) + series-CLS (rank_hinge_xmod) + view-CLS
        (5-way BCE) across views a,b."""
        from xmodal.losses import rank_hinge_xmod_loss
        out = self.forward_self(batch["patches_a"], batch["coords_a"], spec, mask_ratio=mask_ratio)
        sa, va = out["series_repr"], out["view_repr"]
        sb, vb = self._readout(batch["patches_b"], batch["coords_b"], spec, mask_ratio=mask_ratio)
        series_loss, series_viol = rank_hinge_xmod_loss(sa.float(), sb.float(), batch["series"], batch["patient"], n_xmod=n_xmod)
        rel_logits = self.rel_view_head(torch.cat([va, vb], dim=1))
        rel_bce = F.binary_cross_entropy_with_logits(rel_logits, batch["rel_targets"], reduction="none")
        spatial = rel_bce[:, :3].mean(); window = rel_bce[:, 3:5].mean()
        total = (mae_weight * out["mae"] + match_weight * out["match"] + series_weight * series_loss
                 + rel_spatial_weight * spatial + rel_window_weight * window)
        with torch.no_grad():
            rel_acc = ((rel_logits > 0).float() == batch["rel_targets"]).float().mean()
        return dict(loss=total, mae=out["mae"].detach(), match=out["match"].detach(),
                    match_acc=out["match_acc"], series=series_loss.detach(),
                    rel_spatial=spatial.detach(), rel_window=window.detach(),
                    rel_acc=float(rel_acc), series_viol=float(series_viol))

    # ---- cross-modal ---------------------------------------------------------------
    def _gather(self, x, idx, d):
        return x.gather(1, idx[..., None].expand(-1, -1, d))

    def _split(self, B, n, anchor_frac, device):
        perm = torch.argsort(torch.rand(B, n, device=device), dim=1)
        n_anchor = min(n - 1, max(1, int(round(n * anchor_frac))))
        return perm[:, :n_anchor], perm[:, n_anchor:]

    def _context(self, cls_regs_and, coords_and, device, B):
        """Prepend series/view CLS + registers (coord 0) to a token list and its coords."""
        cls = torch.stack([self.series_token, self.view_token])[None].expand(B, -1, -1)
        regs = self.registers[None].expand(B, -1, -1)
        nreg = 2 + regs.shape[1]
        toks = torch.cat([cls, regs, *cls_regs_and], dim=1)
        ccs = torch.cat([torch.zeros(B, nreg, 3, device=device), *coords_and], dim=1)
        return toks, ccs

    def _encode_prism(self, patches, coords, series, spec):
        """Teacher encode: full prism of one modality -> per-patch latents [B,n,W]."""
        B, n = patches.shape[:2]
        nreg = 2 + self.registers.shape[0]
        toks, ccs = self._context([self.embed(patches, spec.key, series)], [coords], patches.device, B)
        return self.encode(toks, ccs)[:, nreg:]

    def teacher_readout(self, patches, coords, spec):
        """One (frozen-teacher) encoder pass -> (series_cls [B,W], patch_latents [B,n,W]). The
        series_cls is the stable per-prism series descriptor that conditions the decoder; the
        patch_latents are the target for the latent phase."""
        B = patches.shape[0]
        nreg = 2 + self.registers.shape[0]
        toks, ccs = self._context([self.embed(patches, spec.key)], [coords], patches.device, B)
        x = self.encode(toks, ccs)
        return x[:, 0], x[:, nreg:]

    def fuse_series(self, patch_enc, series_cls):
        """Fuse each patch encoding with its (broadcast) series-CLS -> 'content + which series'."""
        B, n, W = patch_enc.shape
        return self.fuse(torch.cat([patch_enc, series_cls[:, None, :].expand(B, n, W)], dim=-1))

    def _decode(self, query, ctx, cc, rec_coords):
        for blk in self.decoder:
            query = blk(query, ctx, rec_coords, cc)
        return query

    def forward_cross(self, source_patches, target_patches, coords, spec, src_series_cls, tgt_series_cls,
                      *, anchor_frac=0.05, objective="both", match_weight=1.0):
        """Cross-modal. Online-encode source + target per-series; decoder context = fused(source, its
        series-CLS) [all positions] + fused(target anchors, its series-CLS) [few, disjoint from the
        masked recon]; queries = position + target series-CLS. Predict held-out target patches (pixel
        MAE) and match position<->blind-content. `src/tgt_series_cls` [B,W] come from the FROZEN
        teacher (stable conditioner)."""
        B, n = source_patches.shape[:2]; dev = source_patches.device; W = self.cfg.width
        anchor, recon = self._split(B, n, anchor_frac, dev); n_recon = recon.shape[1]
        src_enc = self._encode_prism(source_patches, coords, None, spec)             # online [B,n,W]
        tgt_enc = self._encode_prism(target_patches, coords, None, spec)
        src_fused = self.fuse_series(src_enc, src_series_cls)                        # all source (context)
        anc_fused = self.fuse_series(self._gather(tgt_enc, anchor, W), tgt_series_cls)  # few target anchors
        rec_coords = self._gather(coords, recon, 3); anc_coords = self._gather(coords, anchor, 3)
        ctx, cc = self._context([src_fused, anc_fused], [coords, anc_coords], dev, B)
        query = (self.query_seed[None, None, :] + tgt_series_cls[:, None, :]).expand(B, n_recon, W).contiguous()
        query = self._decode(query, ctx, cc, rec_coords)
        pv = int(np.prod(spec.voxels))
        tgt_recon = self._gather(target_patches.reshape(B, n, -1), recon, pv)        # disjoint from anchors
        zero = query.new_zeros(())
        mae = F.l1_loss(self.dec_pixel_head[spec.key](query), tgt_recon) if objective in ("mae", "both") else zero
        match_acc = 0.0
        if objective in ("match", "both"):
            slots = F.normalize(self.match_slot_proj(query), dim=-1)
            colors = F.normalize(self.color_head[spec.key](tgt_recon.reshape(B, n_recon, *spec.voxels)), dim=-1)
            m_loss, m_met = slot_match_loss(slots, colors, self.match_logit_scale)
            match_acc = m_met["match_acc"]
            loss = mae + match_weight * m_loss if objective == "both" else m_loss
        else:
            loss = mae
        return dict(loss=loss, mae=mae.detach(), match_acc=match_acc, n_recon=n_recon, n_anchor=int(anchor.shape[1]))

    def forward_cross_latent(self, source_patches, target_patches, coords, spec, src_series_cls,
                             tgt_series_cls, tgt_latents, *, anchor_frac=0.05):
        """Latent phase: same wiring as forward_cross, but the target is the FROZEN teacher's
        per-patch target latents (`tgt_latents` [B,n,W]) instead of pixels. Loss = 1 - cosine."""
        B, n = source_patches.shape[:2]; dev = source_patches.device; W = self.cfg.width
        anchor, recon = self._split(B, n, anchor_frac, dev); n_recon = recon.shape[1]
        src_enc = self._encode_prism(source_patches, coords, None, spec)
        tgt_enc = self._encode_prism(target_patches, coords, None, spec)
        src_fused = self.fuse_series(src_enc, src_series_cls)
        anc_fused = self.fuse_series(self._gather(tgt_enc, anchor, W), tgt_series_cls)
        rec_coords = self._gather(coords, recon, 3); anc_coords = self._gather(coords, anchor, 3)
        ctx, cc = self._context([src_fused, anc_fused], [coords, anc_coords], dev, B)
        query = (self.query_seed[None, None, :] + tgt_series_cls[:, None, :]).expand(B, n_recon, W).contiguous()
        query = self._decode(query, ctx, cc, rec_coords)
        tgt = self._gather(tgt_latents, recon, W)
        cos = (F.normalize(self.latent_head(query), dim=-1) * F.normalize(tgt, dim=-1)).sum(-1)
        return dict(loss=(1.0 - cos).mean(), cos=float(cos.mean().detach()), n_recon=n_recon, n_anchor=int(anchor.shape[1]))
