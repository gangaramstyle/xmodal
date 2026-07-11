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

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from xmodal.matching import (ColorHead, blur_contents, default_log_logit_scale, modality_completion_loss,
                             slot_match_loss, structured_match_loss)
from xmodal.sampling import SCAN_STATS_DIM


@dataclass
class EncoderConfig:
    width: int = 384
    depth: int = 12
    heads: int = 6
    mlp_ratio: int = 4
    n_series: int = 8
    n_registers: int = 4
    decoder_depth: int = 4
    patch_voxels: int = 16            # 2.5D slab sample grid (V,V,1); physical size is per-patch (size embed)
    rope_lambda_min_mm: float = 2.0
    rope_lambda_max_mm: float = 1024.0
    scan_context: bool = False        # v3: scan-relative input channels on stem + target teacher
    max_source: int = 128             # v3: source-slot bank size for ACTIVE variable registers (missing patches)


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

    def __init__(self, cfg: EncoderConfig, patch_specs=None):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.width // cfg.heads
        # SINGLE shared stem/heads: every patch is a 2.5D V×V×1 grid regardless of physical size, so
        # one Conv3d stem + one pixel head + one color head handle all sizes. Physical scale rides
        # along as a per-patch SIZE EMBEDDING (MLP of size in mm), added to the token + decoder query.
        V = cfg.patch_voxels
        self.grid = (V, V, 1)
        self.pv = V * V
        # v3 scan-context A (normalization): patch input carries 3 deterministic scan-RELATIVE channels
        # (raw, robust z-score, histogram-CDF) computed from the scan's stats. Interpretation A -> the
        # target is scan-invariant, so NO learned scan network (no EMA-completeness gap) and the query
        # needs no scan-context. in_ch=1 when scan_context is off.
        self.in_ch = 3 if cfg.scan_context else 1
        self.stem = nn.Conv3d(self.in_ch, cfg.width, self.grid, stride=self.grid)
        # per-axis physical extent (mm) [.,.,3] -> W. For a 2.5D slab two axes = the size, the thin
        # (through-plane) axis ~= 0, so this encodes BOTH scale AND orientation (which plane the slab is).
        self.size_embed = nn.Sequential(nn.Linear(3, cfg.width), nn.GELU(), nn.Linear(cfg.width, cfg.width))
        # per-patch SERIES conditioning (mixed-modality design, docs/MIXED_MODAL_DESIGN.md): two
        # SEPARATE tables — "this token IS series S" (encoder, Site A) vs "produce series S" (decoder
        # query, Site B). Static (unlike the dynamic series-CLS), so a bag can mix modalities per patch.
        self.series_in_embed = nn.Embedding(cfg.n_series, cfg.width)
        self.series_q_embed = nn.Embedding(cfg.n_series, cfg.width)
        nn.init.normal_(self.series_in_embed.weight, std=0.02)
        nn.init.normal_(self.series_q_embed.weight, std=0.02)
        self.pixel_head = nn.Linear(cfg.width, self.pv)
        # cross-modal decoder fuses each patch encoding with its (frozen-teacher) series-CLS so the
        # decoder can tell sources apart (no series_embed lookup — series identity is the CLS output).
        self.fuse = nn.Sequential(nn.Linear(2 * cfg.width, cfg.width), nn.GELU(),
                                  nn.Linear(cfg.width, cfg.width))
        self.series_token = nn.Parameter(torch.randn(cfg.width) * 0.02)
        self.view_token = nn.Parameter(torch.randn(cfg.width) * 0.02)
        self.registers = nn.Parameter(torch.randn(cfg.n_registers, cfg.width) * 0.02)
        # ACTIVE variable registers (v3): a bank of DISTINCT learned embeddings, one per source slot, used
        # for missing source patches. Distinct (not identical copies) so they add attention diversity, not a
        # single repeated token. No fake modality/size/scan embedding; coord 0 (non-spatial, set by sampler).
        self.source_registers = nn.Parameter(torch.randn(cfg.max_source, cfg.width) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(cfg.width) * 0.02)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.width)
        # cross-modal decoder + heads
        self.decoder = nn.ModuleList(CrossAttentionBlock(cfg) for _ in range(cfg.decoder_depth))
        self.query_seed = nn.Parameter(torch.zeros(cfg.width))
        self.dec_pixel_head = nn.Linear(cfg.width, self.pv)
        self.match_slot_proj = nn.Linear(cfg.width, cfg.width)
        self.match_logit_scale = nn.Parameter(torch.tensor(default_log_logit_scale(0.07)))
        # target (value) encoder: blind ColorHead over the (scan-relative) patch channels. Scan context
        # is deterministic input channels (scan-context A) -> the ENTIRE target side is EMA'd cleanly.
        self.color_head = ColorHead(cfg.width, self.grid, in_ch=self.in_ch)
        # EMA (target) copy of the SINGLE shared target encoder (BYOL/DINO-style stable matching target).
        # Init identical; updated by EMA (never by grad). Used only when ema_color is enabled.
        self.color_head_ema = copy.deepcopy(self.color_head)
        for p in self.color_head_ema.parameters():
            p.requires_grad_(False)
        self.latent_head = nn.Linear(cfg.width, cfg.width)
        # view-CLS head: 3 spatial-ordering + 2 window signs (rotation dropped) = 5-way BCE
        self.rel_view_head = nn.Sequential(nn.Linear(2 * cfg.width, cfg.width), nn.GELU(),
                                           nn.Linear(cfg.width, 5))

    def _size_emb(self, sizes):
        """Per-patch per-axis physical extent (mm) [B,k,3] -> additive size+orientation embedding [B,k,W]."""
        return self.size_embed(sizes)

    def _scan_channels(self, raw, reference, stats):
        """v3 scan-context A -> 3-channel scan-RELATIVE input [B,Q,3,V,V,1]. The RAW channel is the
        PRESENTED intensity (may be window-augmented, for student robustness); z-score and CDF are
        computed from the CANONICAL `reference` (clean) intensity against the scan's stats, so the
        calibration channels are invariant to the augmentation (review). Position-free."""
        B, Q = raw.shape[:2]
        med = stats[..., 4].view(B, Q, 1, 1, 1)                           # p50
        iqr = (stats[..., 5] - stats[..., 3]).clamp_min(1e-3).view(B, Q, 1, 1, 1)   # p75-p25 (numerical floor)
        z = (reference - med) / iqr
        cumh = stats[..., 12:28].cumsum(-1)                               # [B,Q,16] CDF over the 16-bin histogram
        idx = (reference.reshape(B, Q, -1) * 16).long().clamp(0, 15)      # [B,Q,V*V]
        cdf = torch.gather(cumh, -1, idx).reshape_as(reference)
        return torch.stack([raw, z, cdf], dim=2)                          # [B,Q,3,V,V,1]

    def _stem_in(self, raw, reference, stats):
        """Prepare stem/color input: 3 scan-relative channels when scan_context (raw=presented, z/CDF from
        `reference`), else the raw patch. `reference` defaults to `raw` (clean = no augmentation)."""
        if self.in_ch == 3 and stats is not None:
            return self._scan_channels(raw, raw if reference is None else reference, stats)
        return raw

    def embed(self, patches, sizes, series_ids=None, reference=None, stats=None, valid=None):
        """patches [B,n,V,V,1] (PRESENTED raw) + sizes [B,n,3] -> tokens [B,n,W]. `series_ids` additively
        injects modality (Site A). `reference`+`stats` build the scan-relative channels (scan-context A).
        `valid` [B,n] bool: slots that are False are MISSING source patches -> replaced wholesale by a
        distinct active register (no content/size/series/scan signal); the sampler sets their coord to 0."""
        B, n = patches.shape[:2]
        inp = self._stem_in(patches, reference, stats).reshape(B * n, self.in_ch, *self.grid)
        tok = self.stem(inp).reshape(B, n, self.cfg.width)
        tok = tok + self._size_emb(sizes).to(tok.dtype)
        if series_ids is not None:
            tok = tok + self.series_in_embed(series_ids).to(tok.dtype)
        if valid is not None:
            tok = torch.where(valid[..., None], tok, self.source_registers[:n][None].to(tok.dtype))
        return tok

    def encode(self, tokens, coords):
        """tokens [B,T,W] with matching coords [B,T,3] (CLS/regs use coord 0) -> [B,T,W]."""
        cos, sin = build_rope(coords, self.head_dim, self.cfg.rope_lambda_min_mm, self.cfg.rope_lambda_max_mm)
        x = tokens
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.norm(x)

    def forward_mae(self, patches, coords, sizes, *, mask_ratio=0.5):
        """Masked patch reconstruction (encoder pixel head). patches [B,n,V,V,1], coords/sizes [B,n]."""
        B, n = patches.shape[:2]
        dev = patches.device
        tok = self.embed(patches, sizes)                                     # [B,n,W]
        mask = torch.rand(B, n, device=dev) < mask_ratio                      # [B,n] masked positions
        tok = torch.where(mask[..., None], self.mask_token, tok)
        cls = torch.stack([self.series_token, self.view_token])[None].expand(B, -1, -1)
        regs = self.registers[None].expand(B, -1, -1)
        nreg = 2 + regs.shape[1]
        x = torch.cat([cls, regs, tok], dim=1)
        cc = torch.cat([torch.zeros(B, nreg, 3, device=dev), coords], dim=1)
        x = self.encode(x, cc)
        patch_out = x[:, nreg:]                                               # [B,n,W]
        recon = self.pixel_head(patch_out)                                   # [B,n,pv]
        target = patches.reshape(B, n, -1)
        loss = F.l1_loss(recon[mask], target[mask]) if mask.any() else recon.new_zeros(())
        return dict(loss=loss, series_cls=x[:, 0], view_cls=x[:, 1], n_masked=int(mask.sum()))

    # ---- phase-0 self: decoder MAE + matching + series-CLS + view-CLS -----------------
    def _gather_patches(self, patches, idx):
        """Gather patches [B,n,v0,v1,v2] at idx [B,k] -> [B,k,v0,v1,v2]."""
        rest = patches.shape[2:]
        return patches.gather(1, idx.view(idx.shape[0], idx.shape[1], *([1] * len(rest))).expand(-1, -1, *rest))

    def _readout(self, patches, coords, sizes, *, mask_ratio):
        """Encode a (masked) prism -> (series_repr[B,W], view_repr[B,W]) for the 2nd view's CLS heads."""
        B, n = patches.shape[:2]; dev = patches.device
        tok = self.embed(patches, sizes)
        mask = torch.rand(B, n, device=dev) < mask_ratio
        tok = torch.where(mask[..., None], self.mask_token, tok)
        toks, ccs = self._context([tok], [coords], dev, B)
        x = self.encode(toks, ccs)
        return x[:, 0], x[:, 1]

    def forward_self(self, patches, coords, sizes, *, mask_ratio=0.5, content_blur=0, held_size=None,
                     soft_match_tau=None, soft_match_sim="model", ema_color=False):
        """MAE-style self task: hold out `mask_ratio` of patches; the encoder sees ONLY the visible
        patches (held-out ones NEVER enter the encoder); the DECODER queries the held-out positions
        to reconstruct pixels (MAE) and match position <-> blind content. Held contents are optionally
        blurred (`content_blur`) so matching can't cheat on exact pixels. The query carries the held
        patch's SIZE (target scale). If `held_size` is set (a size in mm or an iterable of sizes), held-out
        patches are restricted to those in-plane size(s) so matching/recon targets are clean scales
        (e.g. {8,16}, never 4mm); smaller/other sizes stay visible as context. Returns mae, match, match_acc + CLS readouts."""
        B, n = patches.shape[:2]; dev = patches.device; W = self.cfg.width
        nreg = 2 + self.registers.shape[0]
        n_held = max(1, min(n - 1, int(round(n * mask_ratio))))
        if held_size is not None:
            # Hold out ONLY patches whose in-plane extent is an allowed target size. Bias the sort so
            # those come first, then clamp n_held to the per-item target count so held is pure target-scale.
            sz = sizes.amax(-1)                                              # [B,n] in-plane extent (mm)
            allowed = [float(held_size)] if isinstance(held_size, (int, float)) else [float(s) for s in held_size]
            is_t = (sz[..., None] == torch.tensor(allowed, device=dev, dtype=sz.dtype)).any(-1)
            n_held = max(1, min(n_held, int(is_t.sum(1).min().item())))
            perm = torch.argsort(torch.rand(B, n, device=dev) + (~is_t).float(), dim=1)  # targets in [0,1)
        else:
            perm = torch.argsort(torch.rand(B, n, device=dev), dim=1)
        held, vis = perm[:, :n_held], perm[:, n_held:]
        # MAE-style: embed + encode ONLY the visible patches; held positions are queried by the decoder.
        vis_patches = self._gather_patches(patches, vis); vis_coords = self._gather(coords, vis, 3)
        vis_sizes = sizes.gather(1, vis[..., None].expand(-1, -1, 3))
        x = self.encode(*self._context([self.embed(vis_patches, vis_sizes)], [vis_coords], dev, B))
        vis_enc = x[:, nreg:]
        ctx, cc = self._context([vis_enc], [vis_coords], dev, B)               # decoder context = visible encodings
        held_coords = self._gather(coords, held, 3)
        held_sizes = sizes.gather(1, held[..., None].expand(-1, -1, 3))        # [B,n_held,3]
        query = (self.query_seed[None, None, :] + self._size_emb(held_sizes)).contiguous()   # position + target scale
        query = self._decode(query, ctx, cc, held_coords)
        held_patches = self._gather_patches(patches, held)                     # [B,n_held,V,V,1]
        recon = self.dec_pixel_head(query)
        mae = F.l1_loss(recon, held_patches.reshape(B, n_held, self.pv))
        slots = F.normalize(self.match_slot_proj(query), dim=-1)
        m_loss, m_met, colors = self._match_loss(slots, held_patches, content_blur,
                                                 soft_tau=soft_match_tau, soft_sim=soft_match_sim, ema=ema_color)
        return dict(mae=mae, match=m_loss, match_acc=m_met["match_acc"],
                    series_repr=x[:, 0], view_repr=x[:, 1],
                    recon=recon.detach(), held_patches=held_patches.detach(),   # viz: predicted vs truth pixels
                    slots=slots.detach(), colors=colors)                        # viz: match assignment

    def forward_phase0(self, batch, spec=None, *, mask_ratio=0.5, content_blur=0, held_size=None, mae_weight=0.25,
                       match_weight=1.0, series_weight=1.0, rel_spatial_weight=0.25, rel_window_weight=0.25, n_xmod=1,
                       soft_match_tau=None, soft_match_sim="model", ema_color=False):
        """Phase-0 self: decoder MAE + matching (view a) + series-CLS (rank_hinge_xmod) + view-CLS
        (5-way BCE) across views a,b. `spec` is unused (size is per-patch) — kept for call compat."""
        from xmodal.losses import rank_hinge_xmod_loss
        out = self.forward_self(batch["patches_a"], batch["coords_a"], batch["sizes_a"],
                                mask_ratio=mask_ratio, content_blur=content_blur, held_size=held_size,
                                soft_match_tau=soft_match_tau, soft_match_sim=soft_match_sim, ema_color=ema_color)
        sa, va = out["series_repr"], out["view_repr"]
        sb, vb = self._readout(batch["patches_b"], batch["coords_b"], batch["sizes_b"], mask_ratio=mask_ratio)
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

    # ---- mixed-modality conditioned SSL (docs/MIXED_MODAL_DESIGN.md) ------------------
    def _view_repr(self, patches, coords, sizes, series_ids, reference=None, stats=None, valid=None):
        """Encode a bag (per-patch source series, Site A) -> its view-CLS output [B,W] (for view-CLS)."""
        B = patches.shape[0]
        toks, ccs = self._context([self.embed(patches, sizes, series_ids, reference, stats, valid)], [coords], patches.device, B)
        return self.encode(toks, ccs)[:, 1]

    def forward_mixed(self, batch, *, content_blur=0, ema_color=False, mae_weight=0.25, match_weight=1.0,
                      rel_spatial_weight=0.25, rel_window_weight=0.25, structured=False, n_pos=12,
                      drop_source=False, shuffle_coords=False):
        """One continuous objective (no phases). Bag `a` (Site A + scan-relative channels) is encoded as
        decoder context; disjoint held positions carry per-patch TARGET series at the query (Site B) and
        are matched (+ pixel MAE). `structured` -> the P×4 CONDITIONAL loss (position|modality +
        modality|position) instead of the global 48-way. `drop_source`/`shuffle_coords` are validation
        CONTROLS (source removed / target coords permuted) — position accuracy should fall to chance."""
        dev = batch["patches_a_raw"].device
        nreg = 2 + self.registers.shape[0]
        pa, ref_a = batch["patches_a_raw"], batch.get("patches_a_reference")         # presented (aug) + clean ref
        ca, za, ssa = batch["coords_a"], batch["sizes_a"], batch["source_series_a"]
        sta = batch.get("stats_a")                                                  # v3 scan-context (None if off)
        B, n = pa.shape[:2]
        x = self.encode(*self._context([self.embed(pa, za, ssa, ref_a, sta, batch.get("source_valid_a"))], [ca], dev, B))
        src_enc = x[:, nreg:]; va = x[:, 1]
        if drop_source:                                                             # CONTROL: no anatomical context
            src_enc = torch.zeros_like(src_enc)
        ctx, cc = self._context([src_enc], [ca], dev, B)                            # decoder context = bag a
        hsem, hpix = batch["held_semantic"], batch["held_pixel_target"]             # clean target vs view-A pixel target
        hc, hz, tser = batch["held_coords"], batch["held_sizes"], batch["target_series"]
        m = hsem.shape[1]
        if shuffle_coords:                                                          # CONTROL: destroy position signal
            hc = hc[:, torch.randperm(m, device=dev)]
        query = (self.query_seed[None, None, :] + self._size_emb(hz)
                 + self.series_q_embed(tser)).contiguous()                          # Site B: position+size+want-series
        query = self._decode(query, ctx, cc, hc)
        recon = self.dec_pixel_head(query)
        mae = F.l1_loss(recon, hpix.reshape(B, m, self.pv))                         # pixel: view-A domain
        slots = F.normalize(self.match_slot_proj(query), dim=-1)
        hstats = batch.get("held_stats")
        if structured:                                                             # match: CLEAN semantic target
            m_loss, m_met, colors = self._structured_match(slots, hsem, content_blur, hstats, n_pos, ema=ema_color)
            match_acc = m_met["acc_pos"]                                            # position|modality is the headline
        else:
            m_loss, m_met, colors = self._match_loss(slots, hsem, content_blur, ema=ema_color, stats=hstats)
            match_acc = m_met["match_acc"]
        vb = self._view_repr(batch["patches_b_raw"], batch["coords_b"], batch["sizes_b"],
                             batch["source_series_b"], batch.get("patches_b_reference"), batch.get("stats_b"),
                             batch.get("source_valid_b"))
        rel_logits = self.rel_view_head(torch.cat([va, vb], dim=1))
        rel_bce = F.binary_cross_entropy_with_logits(rel_logits, batch["rel_targets"], reduction="none")
        spatial = rel_bce[:, :3].mean(); window = rel_bce[:, 3:5].mean()
        total = (mae_weight * mae + match_weight * m_loss
                 + rel_spatial_weight * spatial + rel_window_weight * window)
        with torch.no_grad():
            rel_acc = float(((rel_logits > 0).float() == batch["rel_targets"]).float().mean())
        return dict(loss=total, mae=mae.detach(), match=m_loss.detach(), match_acc=match_acc,
                    acc_pos=m_met.get("acc_pos"), acc_mod=m_met.get("acc_mod"),
                    loss_pos=m_met.get("loss_pos"), loss_mod=m_met.get("loss_mod"),
                    acc_pos_t2s=m_met.get("acc_pos_t2s"), acc_mod_t2s=m_met.get("acc_mod_t2s"),
                    rel_spatial=spatial.detach(), rel_window=window.detach(), rel_acc=rel_acc,
                    recon=recon.detach(), held_semantic=hsem.detach(),
                    slots=slots.detach(), colors=colors, target_series=tser)

    # ---- v4 modality completion (docs/MIXED_V4_DESIGN.md) -----------------------------
    def forward_modality_completion(self, batch, *, content_blur=0, ema_color=False, mae_weight=0.25, match_weight=1.0):
        """Modality completion: encode 3*P visible co-located patches (the 3 non-target modalities at P
        positions), decode the P hidden targets (query = position + size + requested modality), match
        position|target-modality (chance 1/(P/4)) + pixel MAE. No view-CLS, curriculum, or registers."""
        dev = batch["patches_src_raw"].device
        nreg = 2 + self.registers.shape[0]
        ps, refs = batch["patches_src_raw"], batch.get("patches_src_ref")
        cs, zs, sers, sts = batch["coords_src"], batch["sizes_src"], batch["series_src"], batch.get("stats_src")
        B = ps.shape[0]
        x = self.encode(*self._context([self.embed(ps, zs, sers, refs, sts)], [cs], dev, B))
        ctx, cc = self._context([x[:, nreg:]], [cs], dev, B)                       # decoder context = visible bag
        hsem, hpix = batch["held_semantic"], batch["held_pixel_target"]
        ct, zt, modt, hstats = batch["coords_tgt"], batch["sizes_tgt"], batch["mod_tgt"], batch.get("stats_tgt")
        m = hsem.shape[1]
        query = (self.query_seed[None, None, :] + self._size_emb(zt) + self.series_q_embed(modt)).contiguous()
        query = self._decode(query, ctx, cc, ct)
        recon = self.dec_pixel_head(query)
        mae = F.l1_loss(recon, hpix.reshape(B, m, self.pv))
        slots = F.normalize(self.match_slot_proj(query), dim=-1)
        if ema_color:                                                             # BYOL two-direction
            colors_t = self.color_embed(hsem, content_blur, hstats, ema=True).detach()
            colors_o = self.color_embed(hsem, content_blur, hstats, ema=False)
            l1, met = modality_completion_loss(slots, colors_t, 4, self.match_logit_scale)
            l2, _ = modality_completion_loss(colors_o, slots.detach(), 4, self.match_logit_scale)
            m_loss = 0.5 * (l1 + l2); colors = colors_t
        else:
            colors = self.color_embed(hsem, content_blur, hstats, ema=False)
            m_loss, met = modality_completion_loss(slots, colors, 4, self.match_logit_scale)
        total = mae_weight * mae + match_weight * m_loss
        return dict(loss=total, mae=mae.detach(), match=m_loss.detach(), acc_pos=met["acc_pos"],
                    acc_pos_t2s=met["acc_pos_t2s"], chance_pos=met["chance_pos"],
                    slots=slots.detach(), colors=colors, mod_tgt=modt)

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

    def _encode_prism(self, patches, coords, sizes):
        """Online/teacher encode: full prism of one modality -> per-patch latents [B,n,W]."""
        B, n = patches.shape[:2]
        nreg = 2 + self.registers.shape[0]
        toks, ccs = self._context([self.embed(patches, sizes)], [coords], patches.device, B)
        return self.encode(toks, ccs)[:, nreg:]

    def teacher_readout(self, patches, coords, sizes, series_ids=None, stats=None):
        """One (frozen-teacher) encoder pass -> (series_cls [B,W], patch_latents [B,n,W]). Pass
        `series_ids` [B,n] (and `stats` [B,n,SCAN_STATS_DIM] for scan-context models) so the readout
        conditions the encoder the SAME way training did (Site A + scan-context); leave None for legacy."""
        B = patches.shape[0]
        nreg = 2 + self.registers.shape[0]
        toks, ccs = self._context([self.embed(patches, sizes, series_ids, patches, stats)], [coords], patches.device, B)
        x = self.encode(toks, ccs)
        return x[:, 0], x[:, nreg:]

    @torch.no_grad()
    def update_color_ema(self, m):
        """EMA-update the (single, shared) target color_head from the online one. Call after opt.step()."""
        for pe, po in zip(self.color_head_ema.parameters(), self.color_head.parameters()):
            pe.mul_(m).add_(po.detach(), alpha=1.0 - m)

    @torch.no_grad()
    def ema_drift(self, batch, content_blur=0, n_pos=None):
        """EMA vs online target diagnostic (review): all CROSS-space (self-similarity would be trivially 1).
        cos = paired online·EMA; cross_agree = online query retrieves the same EMA target (over all Q);
        for structured, xspace_acc_pos = online→EMA position|modality retrieval, and pos_margin = positive
        cosine minus the hardest same-modality/different-position negative (unsolvable by modality)."""
        hsem, st = batch["held_semantic"], batch.get("held_stats")
        on = self.color_embed(hsem, content_blur, st, ema=False)
        em = self.color_embed(hsem, content_blur, st, ema=True)
        B, Q, D = on.shape
        cos = (on * em).sum(-1).mean()
        S = torch.einsum("bqd,bkd->bqk", on, em)                            # online query vs EMA candidates
        tgt = torch.arange(Q, device=on.device).expand(B, Q)
        out = dict(cos=float(cos), cross_agree=float((S.argmax(-1) == tgt).float().mean()))
        if n_pos and Q % n_pos == 0:
            P, Mn = n_pos, Q // n_pos
            o = on.reshape(B, P, Mn, D).permute(0, 2, 1, 3); e = em.reshape(B, P, Mn, D).permute(0, 2, 1, 3)  # [B,M,P,D]
            Lp = torch.einsum("bmpd,bmqd->bmpq", o, e)                      # online pos vs EMA pos, same modality
            pos = torch.arange(P, device=on.device)
            posc = Lp[..., pos, pos]                                        # [B,M,P] positive
            neg = Lp.masked_fill(torch.eye(P, dtype=torch.bool, device=on.device)[None, None], -1e9).max(-1).values
            out["xspace_acc_pos"] = float((Lp.argmax(-1) == pos).float().mean())
            out["pos_margin"] = float((posc - neg).mean())
        return out

    @staticmethod
    def _eff_rank(x):
        """Participation-ratio effective rank of the centered rows of x [N,D]: (Σσ²)² / Σσ⁴."""
        x = (x - x.mean(0, keepdim=True)).float()
        s2 = torch.linalg.svdvals(x) ** 2
        return float((s2.sum() ** 2) / (s2.square().sum() + 1e-9))

    @torch.no_grad()
    def repr_diag(self, embeds, series):
        """Collapse instrumentation (review): global effective rank + WITHIN-modality effective rank and
        mean off-diagonal cosine (4 modality centroids can look healthy globally while each modality has
        collapsed). `embeds` [B,Q,D] L2-normalized, `series` [B,Q]."""
        B, Q, D = embeds.shape
        flat = embeds.reshape(-1, D); ser = series.reshape(-1)
        ers, coss = [], []
        for s in ser.unique():
            e = flat[ser == s]
            if e.shape[0] >= 8:
                ers.append(self._eff_rank(e))
                g = e @ e.T; nk = g.shape[0]
                coss.append(float((g.sum() - g.diagonal().sum()) / (nk * (nk - 1))))
        return dict(eff_rank=self._eff_rank(flat), prenorm_std=float(embeds.std()),
                    eff_rank_mod=float(sum(ers) / len(ers)) if ers else 0.0,
                    offdiag_cos_mod=float(sum(coss) / len(coss)) if coss else 0.0)

    @torch.no_grad()
    def assert_structured(self, batch, n_pos, tol=1e-4):
        """Fail-fast on the position-major 12×4 layout the structured loss reshapes assume (review)."""
        tser = batch["target_series"]; B, Q = tser.shape
        assert Q == n_pos * 4, f"held {Q} != n_pos*4 ({n_pos*4})"
        exp = torch.arange(4, device=tser.device)
        assert (tser.reshape(B, n_pos, 4) == exp).all(), "target_series not position-major [0,1,2,3]"
        hc = batch["held_coords"].reshape(B, n_pos, 4, 3)
        assert float(hc.std(dim=2).max()) < tol, "4-way targets don't share a center"
        hz = batch["held_sizes"].reshape(B, n_pos, 4, 3)
        assert float(hz.std(dim=2).max()) < tol, "4-way targets don't share a size"

    def color_embed(self, content, content_blur, stats, ema=False):
        """CLEAN target patch (held_semantic) -> L2-normalized appearance embedding [B,Q,W]. content is
        canonical (no window jitter); a deterministic blur sets bandwidth. raw=reference=content, so z/CDF
        are canonical. ema=True uses the stop-grad EMA head (whole target side EMA'd under scan-context A)."""
        co = blur_contents(content, content_blur)
        inp = self._stem_in(co, co, stats)
        head = self.color_head_ema if ema else self.color_head
        return F.normalize(head(inp), dim=-1)

    def _match_loss(self, slots, content, content_blur, *, soft_tau=None, soft_sim="model", ema=False, stats=None):
        """Global slot<->content matching (v2 mixed path — NOT the structured objective). `content` =
        held patches [B,Q,V,V,1]; `stats` -> scan-relative channels. ema=True: BYOL two-direction. Returns
        (loss, met, target_embed) where target_embed is the space the METRICS should use (EMA when ema)."""
        colors = self.color_embed(content, content_blur, stats, ema=False)
        if ema:
            colors_t = self.color_embed(content, content_blur, stats, ema=True).detach()
            s = self.match_logit_scale.exp().clamp(min=1.0, max=100.0); B, Q, _ = slots.shape
            tgt = torch.arange(Q, device=slots.device).expand(B, Q)
            la = s * torch.einsum("bqd,bkd->bqk", slots, colors_t)          # slots -> stable EMA target (trains slots)
            lb = s * torch.einsum("bqd,bkd->bqk", colors, slots.detach())   # online colors -> slots (trains color_head)
            loss = 0.5 * (F.cross_entropy(la.reshape(B * Q, Q), tgt.reshape(-1))
                          + F.cross_entropy(lb.reshape(B * Q, Q), tgt.reshape(-1)))
            with torch.no_grad():
                acc = float((la.argmax(1) == tgt).float().mean())
            return loss, {"match_acc": acc}, colors_t                       # metrics use the EMA target space
        co = blur_contents(content, content_blur)
        soft_feat = co.reshape(slots.shape[0], slots.shape[1], -1) if (soft_tau is not None and soft_sim == "pixel") else None
        loss, met = slot_match_loss(slots, colors, self.match_logit_scale, soft_tau=soft_tau, soft_feat=soft_feat)
        return loss, met, colors.detach()

    def _structured_match(self, slots, content, content_blur, stats, n_pos, ema=False):
        """v3 conditional matching over the P×4 grid (position|modality + modality|position). EMA gives
        the two-direction BYOL form (slots -> stop-grad EMA target; online target -> stop-grad slots).
        Returns (loss, met, target_embed) with target_embed = the space metrics/breakdown should use."""
        colors = self.color_embed(content, content_blur, stats, ema=False)
        if ema:
            colors_t = self.color_embed(content, content_blur, stats, ema=True).detach()
            l1, met = structured_match_loss(slots, colors_t, n_pos, self.match_logit_scale)         # trains slots
            l2, _ = structured_match_loss(colors, slots.detach(), n_pos, self.match_logit_scale)    # trains online head
            return 0.5 * (l1 + l2), met, colors_t
        loss, met = structured_match_loss(slots, colors, n_pos, self.match_logit_scale)
        return loss, met, colors.detach()

    @torch.no_grad()
    def match_breakdown(self, slots, colors, series):
        """Diagnostic (review pt 4): does match accuracy reflect anatomical correspondence or just
        modality classification? `slots`,`colors` [B,Q,W] normalized, `series` [B,Q] target-series ids.
        - acc_all: argmax over ALL Q candidates (easy — cross-modality candidates are eliminable by
          appearance before anatomy).
        - acc_same: argmax restricted to SAME-series candidates (+ the true one) — the hard anatomical
          discrimination. acc_all >> acc_same means the score is inflated by modality elimination.
        - n_same: avg same-series candidate-pool size (context for reading acc_all)."""
        s = self.match_logit_scale.exp().clamp(max=100.0)
        B, Q, _ = slots.shape
        logits = s * torch.einsum("bqd,bkd->bqk", slots, colors)         # [B,Q,Q]
        tgt = torch.arange(Q, device=slots.device)[None]
        acc_all = (logits.argmax(-1) == tgt).float().mean()
        same = series[:, :, None] == series[:, None, :]                  # [B,Q,Q] candidate k same series as query q
        eye = torch.eye(Q, dtype=torch.bool, device=slots.device)[None]
        masked = logits.masked_fill(~(same | eye), float("-inf"))        # keep same-series + always the true target
        acc_same = (masked.argmax(-1) == tgt).float().mean()
        return dict(acc_all=float(acc_all), acc_same=float(acc_same), n_same=float(same.float().sum(-1).mean()))

    def fuse_series(self, patch_enc, series_cls):
        """Fuse each patch encoding with its (broadcast) series-CLS -> 'content + which series'."""
        B, n, W = patch_enc.shape
        return self.fuse(torch.cat([patch_enc, series_cls[:, None, :].expand(B, n, W)], dim=-1))

    def _decode(self, query, ctx, cc, rec_coords):
        for blk in self.decoder:
            query = blk(query, ctx, rec_coords, cc)
        return query

    def forward_cross(self, source_patches, target_patches, coords, sizes, src_series_cls, tgt_series_cls,
                      *, anchor_frac=0.05, objective="both", match_weight=1.0, content_blur=0, ema_color=False):
        """Cross-modal. Online-encode source + target; decoder context = fused(source, its series-CLS)
        [all positions] + fused(target anchors, its series-CLS) [few, disjoint from the masked recon];
        queries = position + target series-CLS + target SIZE. Predict held-out target patches (pixel
        MAE) and match position<->blind-content. `src/tgt_series_cls` [B,W] come from the FROZEN
        teacher (stable conditioner); `sizes` [B,n] are the shared source/target per-patch sizes."""
        B, n = source_patches.shape[:2]; dev = source_patches.device; W = self.cfg.width
        anchor, recon = self._split(B, n, anchor_frac, dev); n_recon = recon.shape[1]
        src_enc = self._encode_prism(source_patches, coords, sizes)                  # online [B,n,W]
        tgt_enc = self._encode_prism(target_patches, coords, sizes)
        src_fused = self.fuse_series(src_enc, src_series_cls)                        # all source (context)
        anc_fused = self.fuse_series(self._gather(tgt_enc, anchor, W), tgt_series_cls)  # few target anchors
        rec_coords = self._gather(coords, recon, 3); anc_coords = self._gather(coords, anchor, 3)
        ctx, cc = self._context([src_fused, anc_fused], [coords, anc_coords], dev, B)
        rec_sizes = sizes.gather(1, recon[..., None].expand(-1, -1, 3))                                          # [B,n_recon] target scale
        query = (self.query_seed[None, None, :] + tgt_series_cls[:, None, :] + self._size_emb(rec_sizes)).contiguous()
        query = self._decode(query, ctx, cc, rec_coords)
        tgt_recon = self._gather(target_patches.reshape(B, n, -1), recon, self.pv)   # disjoint from anchors
        zero = query.new_zeros(())
        pred = self.dec_pixel_head(query)
        mae = F.l1_loss(pred, tgt_recon) if objective in ("mae", "both") else zero
        match_acc = 0.0
        if objective in ("match", "both"):
            slots = F.normalize(self.match_slot_proj(query), dim=-1)
            m_loss, m_met, _ = self._match_loss(slots, tgt_recon.reshape(B, n_recon, *self.grid), content_blur, ema=ema_color)
            match_acc = m_met["match_acc"]
            loss = mae + match_weight * m_loss if objective == "both" else m_loss
        else:
            loss = mae
        src_recon = self._gather(source_patches.reshape(B, n, -1), recon, self.pv)   # viz: source @ predicted positions
        return dict(loss=loss, mae=mae.detach(), match_acc=match_acc, n_recon=n_recon, n_anchor=int(anchor.shape[1]),
                    pred=pred.detach(), tgt_recon=tgt_recon.detach(), src_recon=src_recon.detach())

    @torch.no_grad()
    def cross_eval(self, source_patches, target_patches, coords, sizes, src_series_cls,
                   tgt_series_cls, tgt_latents, *, anchor_frac=0.05):
        """Eval-only: ONE anchor/recon split, ONE decode, BOTH heads (pixel + latent share the
        identical context/query, so this is perfectly aligned). Returns per-patch [B,n_recon]
        `latent_mismatch` = 1 - cos(pred_latent, frozen-teacher target latent) and `pixel_err` =
        MSE(pred_pixels, true target pixels), plus the `recon` indices so callers can attach seg
        labels. Use latent_mismatch for a latent ckpt, pixel_err for the pixel-cross baseline."""
        B, n = source_patches.shape[:2]; dev = source_patches.device; W = self.cfg.width
        anchor, recon = self._split(B, n, anchor_frac, dev); n_recon = recon.shape[1]
        src_enc = self._encode_prism(source_patches, coords, sizes)
        tgt_enc = self._encode_prism(target_patches, coords, sizes)
        src_fused = self.fuse_series(src_enc, src_series_cls)
        anc_fused = self.fuse_series(self._gather(tgt_enc, anchor, W), tgt_series_cls)
        rec_coords = self._gather(coords, recon, 3); anc_coords = self._gather(coords, anchor, 3)
        ctx, cc = self._context([src_fused, anc_fused], [coords, anc_coords], dev, B)
        rec_sizes = sizes.gather(1, recon[..., None].expand(-1, -1, 3))
        query = (self.query_seed[None, None, :] + tgt_series_cls[:, None, :] + self._size_emb(rec_sizes)).contiguous()
        query = self._decode(query, ctx, cc, rec_coords)
        pred_lat = self.latent_head(query); tgt_lat = self._gather(tgt_latents, recon, W)
        latent_mismatch = 1.0 - (F.normalize(pred_lat, dim=-1) * F.normalize(tgt_lat, dim=-1)).sum(-1)
        pred_px = self.dec_pixel_head(query)
        tgt_px = self._gather(target_patches.reshape(B, n, -1), recon, self.pv)
        pixel_err = ((pred_px - tgt_px) ** 2).mean(-1)
        return dict(recon=recon, latent_mismatch=latent_mismatch.detach(), pixel_err=pixel_err.detach(), n_recon=n_recon)

    def forward_cross_latent(self, source_patches, target_patches, coords, sizes, src_series_cls,
                             tgt_series_cls, tgt_latents, *, anchor_frac=0.05):
        """Latent phase: same wiring as forward_cross, but the target is the FROZEN teacher's
        per-patch target latents (`tgt_latents` [B,n,W]) instead of pixels. Loss = 1 - cosine."""
        B, n = source_patches.shape[:2]; dev = source_patches.device; W = self.cfg.width
        anchor, recon = self._split(B, n, anchor_frac, dev); n_recon = recon.shape[1]
        src_enc = self._encode_prism(source_patches, coords, sizes)
        tgt_enc = self._encode_prism(target_patches, coords, sizes)
        src_fused = self.fuse_series(src_enc, src_series_cls)
        anc_fused = self.fuse_series(self._gather(tgt_enc, anchor, W), tgt_series_cls)
        rec_coords = self._gather(coords, recon, 3); anc_coords = self._gather(coords, anchor, 3)
        ctx, cc = self._context([src_fused, anc_fused], [coords, anc_coords], dev, B)
        rec_sizes = sizes.gather(1, recon[..., None].expand(-1, -1, 3))
        query = (self.query_seed[None, None, :] + tgt_series_cls[:, None, :] + self._size_emb(rec_sizes)).contiguous()
        query = self._decode(query, ctx, cc, rec_coords)
        tgt = self._gather(tgt_latents, recon, W)
        cos = (F.normalize(self.latent_head(query), dim=-1) * F.normalize(tgt, dim=-1)).sum(-1)
        return dict(loss=(1.0 - cos).mean(), cos=float(cos.mean().detach()), n_recon=n_recon, n_anchor=int(anchor.shape[1]))
