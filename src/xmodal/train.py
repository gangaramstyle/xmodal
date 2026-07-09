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
    wandb: str | None = None          # W&B project name (None = off)
    wandb_run: str | None = None      # W&B run name
    log_every: int = 50
    # cross / latent phase
    cross_weight: float = 1.0         # weight on the cross-modal decoder loss (added to encoder objectives)
    match_weight: float = 1.0         # matching vs pixel weight inside forward_cross
    anchor_frac: float = 0.05
    pairs_per_patient: int = 6
    # mixed-size 2.5D sampling (categorical): per-patch physical size + per-item prism extent (mm)
    patch_sizes: tuple = (4.0, 8.0, 16.0)
    prism_choices: tuple = (32.0, 64.0, 128.0)
    size_per_bag: bool = False        # ablation: one size per bag (no within-bag scale mixing)
    artifact_every: int = 5000        # log a wandb checkpoint artifact every N steps (0 = off)


def _cosine_warmup(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * p))


def train_phase0(model, train_bundles, val_bundles, specs, cfg: TrainConfig, *, device="cuda", log=print):
    """Run phase-0 training. `specs` is a dict name->PatchSpec (mixed across steps).
    `train_bundles` may be a list of bundles OR a JitteredRotatingCache (anything with
    .resident()/.step()) for datasets that exceed VRAM."""
    torch.manual_seed(cfg.seed)
    _is_cache = hasattr(train_bundles, "resident")
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
    wb = None
    if cfg.wandb:
        try:
            import wandb as wb
            wb.init(project=cfg.wandb, name=cfg.wandb_run, config=cfg.__dict__)
        except Exception as e:
            log(f"[wandb] disabled: {e}"); wb = None

    def draw(bundles):
        return S.sample_paired_batch(bundles, batch_size=cfg.batch_size, token_count=cfg.token_count,
                                     patch_sizes=cfg.patch_sizes, prism_choices=cfg.prism_choices,
                                     rng=rng, device=device)

    def phase0(b):
        return model.forward_phase0(b, mask_ratio=cfg.mask_ratio, mae_weight=cfg.mae_weight,
                                    series_weight=cfg.series_weight, rel_spatial_weight=cfg.rel_spatial_weight,
                                    rel_window_weight=cfg.rel_window_weight, n_xmod=cfg.n_xmod)

    @torch.no_grad()
    def validate():
        model.eval(); maes = []; accs = []
        for _ in range(cfg.val_iters):
            b = draw(val_bundles)
            with torch.autocast(**amp):
                out = phase0(b)
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
        b = draw(train_bundles.resident() if _is_cache else train_bundles)
        with torch.autocast(**amp):
            out = phase0(b)
            total = out["loss"]
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if _is_cache:
            train_bundles.step()
        hist.append(dict(step=step, total=float(total), mae=float(out["mae"]),
                         series=float(out["series"]), rel_acc=out["rel_acc"]))
        if wb and step % cfg.log_every == 0:
            r = hist[-1]
            wb.log({"train/total": r["total"], "train/mae": r["mae"], "train/series": r["series"],
                    "train/rel_spatial": float(out["rel_spatial"]), "train/rel_window": float(out["rel_window"]),
                    "train/rel_acc": r["rel_acc"], "train/series_viol": out["series_viol"], "lr": lr}, step=step)
        if step % cfg.val_every == 0:
            vm, va = validate()
            r = hist[-1]
            log(f"{step:>6} {r['total']:>8.4f} {r['mae']:>7.4f} {r['series']:>7.4f} {r['rel_acc']:>7.3f} {vm:>8.4f} {va:>8.3f} {lr:>8.2e}")
            if wb:
                wb.log({"val/mae": vm, "val/rel_acc": va}, step=step)
        if cfg.ckpt_dir and step % cfg.ckpt_every == 0:
            import os
            os.makedirs(cfg.ckpt_dir, exist_ok=True)
            torch.save({"model": model.state_dict(), "step": step, "cfg": cfg.__dict__},
                       f"{cfg.ckpt_dir}/step_{step:06d}.pt")
    dt = time.time() - t0
    log(f"done {cfg.steps} steps in {dt:.0f}s ({cfg.steps/dt:.1f} steps/s)")
    if wb:
        wb.finish()
    return hist


def train_phased(model, train_source, val_bundles, specs, cfg: TrainConfig, *, phases,
                 device="cuda", log=print):
    """One continuous run: self -> cross -> latent. `phases` = [(name, steps), ...] with name in
    {'self','cross','latent'}. A frozen teacher is snapshotted at the first non-self phase and gives
    the decoder its stable per-prism series-CLS conditioning (+ latent targets). Encoder objectives
    (self-MAE + view-CLS + series-CLS) train in EVERY phase; the cross/latent decoder loss is added
    on top in those phases. wandb throughout, phase-tagged. `train_source` = bundles list or cache."""
    import copy
    import os
    torch.manual_seed(cfg.seed)
    spec_list = list(specs.values())
    is_cache = hasattr(train_source, "resident")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.95), fused=(device == "cuda"))
    teacher = copy.deepcopy(model).eval()                 # created BEFORE compile (clean); weights loaded at transition
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_ready = False
    if cfg.compile:
        try:
            model.encode = torch.compile(model.encode, dynamic=True)
        except Exception as e:
            log(f"[compile] skipped: {e}")
    wb = None
    if cfg.wandb:
        try:
            import wandb as wb
            wb.init(project=cfg.wandb, name=cfg.wandb_run, config={**cfg.__dict__, "phases": phases})
        except Exception as e:
            log(f"[wandb] disabled: {e}"); wb = None
    rng = np.random.default_rng(cfg.seed)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.amp_bf16 and device == "cuda")
    total = sum(s for _, s in phases)
    warmup = int(cfg.warmup_frac * total)

    def pool():
        return train_source.resident() if is_cache else train_source

    def paired(bnd):
        # mixed-size 2.5D: per-patch sizes {4,8,16} mm + per-item prism {32,64,128} mm (sampler defaults)
        return S.sample_paired_batch(bnd, batch_size=cfg.batch_size, token_count=cfg.token_count,
                                     patch_sizes=cfg.patch_sizes, prism_choices=cfg.prism_choices,
                                     size_per_bag=cfg.size_per_bag, rng=rng, device=device)

    def crossb(bnd):
        return S.sample_cross_batch_vec(bnd, batch_size=cfg.batch_size, token_count=cfg.token_count,
                                        patch_sizes=cfg.patch_sizes, prism_choices=cfg.prism_choices,
                                        size_per_bag=cfg.size_per_bag, rng=rng, device=device,
                                        pairs_per_patient=cfg.pairs_per_patient)

    def teach(b, want_latent=False):
        with torch.no_grad(), torch.autocast(**amp):
            s_scls, _ = teacher.teacher_readout(b["source"], b["coords"], b["sizes"])
            t_scls, t_lat = teacher.teacher_readout(b["target"], b["coords"], b["sizes"])
        return s_scls.float(), t_scls.float(), (t_lat.float() if want_latent else None)

    def phase0(b):
        return model.forward_phase0(b, mask_ratio=cfg.mask_ratio, mae_weight=cfg.mae_weight,
                                    match_weight=cfg.match_weight, series_weight=cfg.series_weight,
                                    rel_spatial_weight=cfg.rel_spatial_weight,
                                    rel_window_weight=cfg.rel_window_weight, n_xmod=cfg.n_xmod)

    @torch.no_grad()
    def validate():
        model.eval(); maes = []; accs = []
        for _ in range(cfg.val_iters):
            b = paired(val_bundles)
            with torch.autocast(**amp):
                o = phase0(b)
            maes.append(float(o["mae"])); accs.append(o["rel_acc"])
        model.train(); return float(np.mean(maes)), float(np.mean(accs))

    log(f"phased run: {phases} | total {total} steps"); t0 = time.time()
    gstep = 0
    for pname, psteps in phases:
        if pname != "self":
            # re-snapshot the teacher at EVERY phase boundary: frozen-self conditions cross,
            # frozen-cross conditions latent (each phase continues off the previous one's weights).
            teacher.load_state_dict(model.state_dict()); teacher.eval()
            log(f"[{pname}] re-snapshotted frozen teacher from step {gstep}")
        for _ in range(psteps):
            gstep += 1
            lr = _cosine_warmup(gstep, total, cfg.lr, warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast(**amp):
                b0 = paired(pool())
                out0 = phase0(b0)
                loss = out0["loss"]
                m = dict(mae=float(out0["mae"]), match=float(out0["match"]), series=float(out0["series"]),
                         rel_acc=out0["rel_acc"], self_match_acc=out0["match_acc"], series_viol=out0["series_viol"])
                if pname == "cross":
                    bc = crossb(pool()); s_scls, t_scls, _ = teach(bc)
                    outc = model.forward_cross(bc["source"], bc["target"], bc["coords"], bc["sizes"], s_scls, t_scls,
                                               objective="both", anchor_frac=cfg.anchor_frac, match_weight=cfg.match_weight)
                    loss = loss + cfg.cross_weight * outc["loss"]
                    m.update(cross_mae=float(outc["mae"]), match_acc=outc["match_acc"])
                elif pname == "latent":
                    bc = crossb(pool()); s_scls, t_scls, t_lat = teach(bc, want_latent=True)
                    outc = model.forward_cross_latent(bc["source"], bc["target"], bc["coords"], bc["sizes"],
                                                      s_scls, t_scls, t_lat, anchor_frac=cfg.anchor_frac)
                    loss = loss + cfg.cross_weight * outc["loss"]
                    m.update(latent_cos=outc["cos"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if is_cache:
                train_source.step()
            if wb and gstep % cfg.log_every == 0:
                wb.log({"phase": {"self": 0, "cross": 1, "latent": 2}[pname], "lr": lr,
                        **{f"train/{k}": v for k, v in m.items()}}, step=gstep)
            if gstep % cfg.val_every == 0:
                vm, va = validate()
                log(f"[{pname}] {gstep:>6} loss {float(loss):.4f} | "
                    + " ".join(f"{k} {v:.3f}" for k, v in m.items()) + f" | val_mae {vm:.4f} val_rel {va:.3f}")
                if wb:
                    wb.log({"val/mae": vm, "val/rel_acc": va}, step=gstep)
            if cfg.ckpt_dir and gstep % cfg.ckpt_every == 0:
                os.makedirs(cfg.ckpt_dir, exist_ok=True)
                ckpt_path = f"{cfg.ckpt_dir}/step_{gstep:06d}.pt"
                torch.save({"model": model.state_dict(), "step": gstep, "phase": pname, "cfg": cfg.__dict__}, ckpt_path)
                if wb and cfg.artifact_every and gstep % cfg.artifact_every == 0:
                    try:
                        art = wb.Artifact(f"ckpt-{cfg.wandb_run or 'run'}", type="model",
                                          metadata={"step": gstep, "phase": pname})
                        art.add_file(ckpt_path); wb.log_artifact(art)
                    except Exception as e:
                        log(f"[wandb] artifact log skipped: {e}")
    if wb:
        wb.finish()
    log(f"phased run done: {gstep} steps in {time.time()-t0:.0f}s")
