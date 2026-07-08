"""Deterministic, platform-independent patient holdout.

The SAME patient IDs land in the SAME split on CUBIC, molab, or anywhere — because the split is a
pure function of the patient ID (md5 hash), not of local RNG/order. Two mechanisms, used together:

1. **Hash split** — `is_heldout(pid, seed, frac)` buckets each patient by md5; reproducible across
   machines and languages, no shared state.
2. **Explicit manifest** — a committed JSON list of patient IDs that must ALWAYS be excluded from
   pretraining (challenge val/test + any fixed eval set). This is the contamination guard the BraTS
   design doc requires ("strictly hold out all challenge val/test patients from pretraining").

Every trainer + eval on every platform imports these, so eval sets never drift and pretraining never
touches an eval patient.
"""
from __future__ import annotations

import hashlib
import json
import os


def patient_bucket(patient_id: str, seed: int = 0) -> int:
    """Stable bucket in [0, 10000) for a patient — platform/language-independent (md5)."""
    h = hashlib.md5(f"{seed}:{patient_id}".encode()).hexdigest()
    return int(h, 16) % 10000


def is_heldout(patient_id: str, *, seed: int = 0, frac: float = 0.1) -> bool:
    """True if the patient falls in the held-out fraction (deterministic)."""
    return patient_bucket(patient_id, seed) < int(frac * 10000)


def split_patients(patient_ids, *, seed: int = 0, val_frac: float = 0.1, exclude=None):
    """Deterministic (train, val) split. `exclude` (e.g. challenge val/test IDs) is removed from
    BOTH sides so those patients never contaminate pretraining. Returns sorted lists."""
    exclude = set(exclude or ())
    ids = [p for p in patient_ids if p not in exclude]
    val = sorted(p for p in ids if is_heldout(p, seed=seed, frac=val_frac))
    train = sorted(p for p in ids if not is_heldout(p, seed=seed, frac=val_frac))
    return train, val


def load_manifest(path) -> set:
    """Load an explicit holdout manifest (JSON list of patient IDs); empty set if absent."""
    if path and os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_manifest(path, patient_ids) -> None:
    with open(path, "w") as f:
        json.dump(sorted(set(patient_ids)), f, indent=0)


def contamination_check(pretrain_ids, holdout_ids) -> list:
    """Return the (sorted) intersection of pretraining IDs and the holdout — must be empty. Call
    this as an assertion before every pretraining run."""
    bad = sorted(set(pretrain_ids) & set(holdout_ids))
    return bad
