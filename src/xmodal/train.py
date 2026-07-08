"""Phase-0 trainer: full faithful objective + real efficiency infra.

Objective (paired views, weighted): pixel-MAE (0.25) + series-CLS (rank_hinge_xmod_loss, 1.0,
same-sequence-different-patient positives) + view-CLS (5-way BCE: 3 spatial + 2 window,
0.25 + 0.25; rotation dropped). See model.forward_phase0 + sampling.sample_paired_batch.

Infra: torch.compile (encode path), bf16 autocast, fused AdamW, grad-clip 1.0, cosine-warmup LR,
checkpointing, held-out val (MAE + view-CLS rel_acc).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from xmodal import sampling as S


@torch.no_grad()
def modality_1nn_acc(emb, labels):
    """Leave-one-out 1-NN modality accuracy from the series embedding (diagnostic)."""
    emb = F.normalize(emb, dim=-1)
    sim = emb @ emb.T
    sim.fill_diagonal_(-1e9)
    nn = sim.argmax(1)
    return float((labels[nn] == labels).float().mean())


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 24
    token_count: int = 128
    mask_ratio: float = 0.35
    mae_weight: float = 0.25
    series_weight: float = 1.0
    rel_spatial_weight: float = 0.25
    rel_window_weight: float = 0.25
    n_xmod: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    prism_lo: float = 24.0
    prism_hi: float = 48.0
    amp_bf16: bool = True
    compile: bool = True
    val_every: int = 250
    val_iters: int = 20
    ckpt_every: int = 1000
    ckpt_dir: str | None = None
    seed: int = 0


def _cosine_warmup(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * p))


def train_phase0(model, train_bundles, val_bundles, specs, cfg: TrainConfig, *, device="cuda", log=print):
    """Run phase-0 training. `specs` is a dict name->PatchSpec (mixed across steps)."""
    torch.manual_seed(cfg.seed)
    spec_list = list(specs.values())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.95), fused=(device == "cuda"))
    if cfg.compile:
        try:
            model.encode = torch.compile(model.encode, dynamic=True)
        except Exception as e:  # compile is best-effort
            log(f"[compile] skipped: {e}")
    rng = np.random.default_rng(cfg.seed)
    warmup = int(cfg.warmup_frac * cfg.steps)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.amp_bf16 and device == "cuda")

    def draw(bundles):
        spec = spec_list[rng.integers(len(spec_list))]
        prism = tuple(float(x) for x in rng.uniform(cfg.prism_lo, cfg.prism_hi, size=3))
        b = S.sample_paired_batch(bundles, batch_size=cfg.batch_size, token_count=cfg.token_count,
                                  patch_spec=spec, prism_mm=prism, rng=rng, device=device)
        return b, spec

    def phase0(b, spec):
        return model.forward_phase0(b, spec, mask_ratio=cfg.mask_ratio, mae_weight=cfg.mae_weight,
                                    series_weight=cfg.series_weight, rel_spatial_weight=cfg.rel_spatial_weight,
                                    rel_window_weight=cfg.rel_window_weight, n_xmod=cfg.n_xmod)

    @torch.no_grad()
    def validate():
        model.eval(); maes = []; accs = []
        for _ in range(cfg.val_iters):
            b, spec = draw(val_bundles)
            with torch.autocast(**amp):
                out = phase0(b, spec)
            maes.append(float(out["mae"])); accs.append(out["rel_acc"])
        model.train(); return np.mean(maes), np.mean(accs)

    hist = []
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    vm, va = validate()
    log(f"{'step':>6} {'total':>8} {'mae':>7} {'series':>7} {'relacc':>7} {'val_mae':>8} {'val_rel':>8} {'lr':>8}")
    log(f"{0:>6} {'-':>8} {'-':>7} {'-':>7} {'-':>7} {vm:>8.4f} {va:>8.3f} {'-':>8}")
    for step in range(1, cfg.steps + 1):
        lr = _cosine_warmup(step, cfg.steps, cfg.lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        b, spec = draw(train_bundles)
        with torch.autocast(**amp):
            out = phase0(b, spec)
            total = out["loss"]
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        hist.append(dict(step=step, total=float(total), mae=float(out["mae"]),
                         series=float(out["series"]), rel_acc=out["rel_acc"]))
        if step % cfg.val_every == 0:
            vm, va = validate()
            r = hist[-1]
            log(f"{step:>6} {r['total']:>8.4f} {r['mae']:>7.4f} {r['series']:>7.4f} {r['rel_acc']:>7.3f} {vm:>8.4f} {va:>8.3f} {lr:>8.2e}")
        if cfg.ckpt_dir and step % cfg.ckpt_every == 0:
            import os
            os.makedirs(cfg.ckpt_dir, exist_ok=True)
            torch.save({"model": model.state_dict(), "step": step, "cfg": cfg.__dict__},
                       f"{cfg.ckpt_dir}/step_{step:06d}.pt")
    dt = time.time() - t0
    log(f"done {cfg.steps} steps in {dt:.0f}s ({cfg.steps/dt:.1f} steps/s)")
    return hist
