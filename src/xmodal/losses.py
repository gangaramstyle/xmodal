"""Contrastive / ranking losses — ported verbatim from brats2026/losses/contrastive.py.

Used for series-CLS: pull same-sequence scans together across patients (rank_hinge_xmod_loss),
tolerating near-positive scans better than InfoNCE.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rank_hinge_loss(za, zb, *, rank_frac: float = 0.2, margin: float = 0.1):
    """Rank-k hinge over paired views: the positive pair must outrank the rank_frac
    hardest-negative threshold by `margin`."""
    z = F.normalize(torch.cat([za, zb], dim=0), dim=1)
    batch = za.shape[0]
    total = 2 * batch
    sim = z @ z.T
    idx = torch.arange(total, device=za.device)
    pos = torch.where(idx < batch, idx + batch, idx - batch)
    s_pos = sim.gather(1, pos[:, None]).squeeze(1)
    eye = torch.eye(total, dtype=torch.bool, device=za.device)
    is_pos = F.one_hot(pos, total).bool()
    negatives = sim.masked_fill(eye | is_pos, -1e9)
    k = max(1, int(rank_frac * (total - 2)))
    neg_k = negatives.topk(k, dim=1).values[:, -1]
    loss = F.relu(margin + neg_k - s_pos)
    return loss.mean(), (loss > 0).float().mean()


def rank_hinge_xmod_loss(za, zb, modality, patient_id, *, rank_frac: float = 0.2,
                         margin: float = 0.1, n_xmod: int = 1, generator=None):
    """Rank-k hinge with one or more same-modality, different-patient positives."""
    z = F.normalize(torch.cat([za, zb], dim=0), dim=1)
    batch = za.shape[0]
    total = 2 * batch
    m = torch.cat([modality, modality], dim=0)
    p = torch.cat([patient_id, patient_id], dim=0)
    sim = z @ z.T
    idx = torch.arange(total, device=za.device)
    eye = torch.zeros(total, total, dtype=torch.bool, device=za.device)
    eye[idx, idx] = True
    scan_mate = torch.where(idx < batch, idx + batch, idx - batch)
    positive = torch.zeros(total, total, dtype=torch.bool, device=za.device)
    positive[idx, scan_mate] = True
    candidates = (m[:, None] == m[None, :]) & (p[:, None] != p[None, :]) & ~eye
    candidates[idx, scan_mate] = False
    random_scores = torch.rand(total, total, generator=generator, device=za.device).masked_fill(~candidates, -1.0)
    picks = random_scores.topk(max(1, n_xmod), dim=1).indices
    valid = random_scores.gather(1, picks) > -0.5
    rows = idx[:, None].expand_as(picks)
    positive[rows[valid], picks[valid]] = True
    pick_mates = torch.where(picks < batch, picks + batch, picks - batch)
    positive[rows[valid], pick_mates[valid]] = True
    negatives = sim.masked_fill(eye | positive, -1e9)
    k = max(1, int(rank_frac * (total - 2)))
    neg_k = negatives.topk(k, dim=1).values[:, -1]
    positive_f = positive.float()
    violations = F.relu(margin + neg_k[:, None] - sim) * positive_f
    loss = (violations.sum(dim=1) / positive_f.sum(dim=1).clamp(min=1)).mean()
    return loss, (violations > 0).float().sum() / positive_f.sum().clamp(min=1)


@torch.no_grad()
def batch_modality_1nn_acc(series_repr_a, series_repr_b, series_idx):
    """Leave-one-out 1-NN modality accuracy across both views (diagnostic)."""
    z = F.normalize(torch.cat([series_repr_a.float(), series_repr_b.float()], dim=0), dim=1)
    sim = z @ z.T
    sim.fill_diagonal_(-1e9)
    lab = torch.cat([series_idx, series_idx], dim=0)
    return float((lab[sim.argmax(1)] == lab).float().mean())
