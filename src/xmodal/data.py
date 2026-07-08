"""Data layer: BraTS (HF) loader + rotating GPU cache for datasets that exceed VRAM.

Key design for high GPU util under a rotating cache: **decode on CPU in background prefetch
threads, place on the GPU on the main thread**. nibabel + numpy (decode, normalize, foreground)
release/are-fine off the critical path; only the cheap H2D placement touches CUDA, and it happens
on the training thread so it never contends with the compute stream. Refresh is rare (jittered
lifetimes + capped per-step fraction), so the placement hitch is amortized over many steps.

Ported from brats2026/data/rotating_cache.py, extended with the CPU-loader / GPU-placer split.
"""
from __future__ import annotations

import queue
import random
import time
from dataclasses import dataclass
from threading import Lock, Thread
from typing import Callable, Generic, TypeVar

import numpy as np

from xmodal import sampling as S

T = TypeVar("T")

BRATS_REPO = "Spirit-26/BraTS-2024-Complete"
BRATS_SUFFIX = {"t1": "t1n", "t1c": "t1c", "t2": "t2w", "flair": "t2f"}
BRATS_SERIES = {"t1": 0, "t1c": 1, "t2": 2, "flair": 3}


def discover_brats_patients(repo=BRATS_REPO, split="train", limit=None):
    from huggingface_hub import list_repo_files
    files = list_repo_files(repo, repo_type="dataset")
    pref = f"BraTS-GLI/{split}/"
    pats = sorted({f.split("/")[2] for f in files if f.startswith(pref) and f.count("/") >= 3})
    return pats[:limit] if limit else pats


# ---------------------------------------------------------------------------
# CPU decode (background-safe) -> GPU placement (main thread)
# ---------------------------------------------------------------------------

def load_brats_cpu(pid, *, repo=BRATS_REPO, split="train", modalities=None,
                   fg_thresh=0.02, max_fg=200_000):
    """Decode one patient to a CPU payload (numpy). No CUDA — safe in a prefetch thread."""
    import nibabel as nib
    from huggingface_hub import hf_hub_download
    mods = modalities or list(BRATS_SUFFIX)
    base = f"BraTS-GLI/{split}/{pid}/{pid}-"
    out = {}
    for m in mods:
        img = nib.load(hf_hub_download(repo, base + BRATS_SUFFIX[m] + ".nii.gz", repo_type="dataset"))
        vol = np.nan_to_num(np.ascontiguousarray(img.get_fdata(), dtype=np.float32), copy=False)
        affine = np.asarray(img.affine, np.float32)
        R, t = affine[:3, :3], affine[:3, 3]
        thick, plane, spacing = S._thick_and_plane(R, None)
        hi = float(vol.max()) or 1.0
        vox = np.argwhere(vol > fg_thresh * hi).astype(np.float32)
        if len(vox) == 0:
            vox = np.argwhere(np.ones_like(vol)).astype(np.float32)
        if len(vox) > max_fg:
            vox = vox[np.random.default_rng(0).choice(len(vox), max_fg, replace=False)]
        fg = (vox @ R.T + t).astype(np.float32)
        keep = vol > fg_thresh * hi
        lo, hiP = np.percentile(vol[keep], [0.5, 99.5]) if keep.any() else (0.0, hi)
        voln = np.clip((vol - lo) / max(hiP - lo, 1e-6), 0.0, 1.0).astype(np.float32)
        out[m] = dict(volume=voln, affine_inv=np.linalg.inv(R).astype(np.float32),
                      affine_trans=t.astype(np.float32), foreground_mm=fg, thick=int(thick),
                      plane=int(plane), series=BRATS_SERIES[m], patient=pid, spacing=spacing)
    return out


def place_bundle(cpu_bundle, device):
    """Move a CPU payload onto the GPU as {modality: CachedScan}. Cheap H2D; main-thread only."""
    import torch
    b = {}
    for m, p in cpu_bundle.items():
        b[m] = S.CachedScan(
            volume=torch.as_tensor(p["volume"], device=device),
            affine_inv=torch.as_tensor(p["affine_inv"], device=device),
            affine_trans=torch.as_tensor(p["affine_trans"], device=device),
            foreground_mm=torch.as_tensor(p["foreground_mm"], device=device),
            thick_axis=p["thick"], plane_id=p["plane"], modality=m, series_idx=p["series"],
            patient=p["patient"], spacing=p["spacing"])
    return b


def load_brats_bundle(pid, *, device, **kw):
    """Convenience: decode + place in one call (main thread)."""
    return place_bundle(load_brats_cpu(pid, **kw), device)


# ---------------------------------------------------------------------------
# Local BraTS-style datasets (METS/PED/GoAT unzipped on disk: nii dirs + corrected labels)
# ---------------------------------------------------------------------------

import glob

LOCAL_SUFFIX = {"t1": "t1n", "t1c": "t1c", "t2": "t2w", "flair": "t2f"}
LOCAL_SERIES = {"t1": 0, "t1c": 1, "t2": 2, "flair": 3}


def _cpu_payload(vol_raw, affine, modality, series_idx, patient, *, fg_thresh=0.02, max_fg=200_000):
    """Build the CPU payload (normalized volume + world-mm foreground + geometry) for place_bundle."""
    vol = np.nan_to_num(np.ascontiguousarray(vol_raw, dtype=np.float32), copy=False)
    affine = np.asarray(affine, np.float32)
    R, t = affine[:3, :3], affine[:3, 3]
    thick, plane, spacing = S._thick_and_plane(R, None)
    hi = float(vol.max()) or 1.0
    keep = vol > fg_thresh * hi
    vox = np.argwhere(keep).astype(np.float32)
    if len(vox) == 0:
        vox = np.argwhere(np.ones_like(vol)).astype(np.float32)
    if len(vox) > max_fg:
        vox = vox[np.random.default_rng(0).choice(len(vox), max_fg, replace=False)]
    fg = (vox @ R.T + t).astype(np.float32)
    lo, hiP = np.percentile(vol[keep], [0.5, 99.5]) if keep.any() else (0.0, hi)
    voln = np.clip((vol - lo) / max(hiP - lo, 1e-6), 0.0, 1.0).astype(np.float32)
    return dict(volume=voln, affine_inv=np.linalg.inv(R).astype(np.float32),
                affine_trans=t.astype(np.float32), foreground_mm=fg, thick=int(thick),
                plane=int(plane), series=int(series_idx), patient=patient, spacing=spacing)


def find_brats_patients(root, anchor_suffix="t1n"):
    """Discover BraTS-style patient dirs under `root`: any dir with a `<pid>-<anchor>.nii.gz`.
    Returns {patient_id: patient_dir}."""
    out = {}
    for f in glob.glob(os.path.join(root, "**", f"*-{anchor_suffix}.nii.gz"), recursive=True):
        pid = os.path.basename(f)[: -len(f"-{anchor_suffix}.nii.gz")]
        out[pid] = os.path.dirname(f)
    return out


def _find_seg(pid, patient_dir, labels_dir):
    """Locate a patient's GT seg — prefer `labels_dir` (corrected labels), else the bundled -seg."""
    if labels_dir:
        for pat in (f"{pid}*seg*.nii.gz", f"{pid}*.nii.gz"):
            hits = glob.glob(os.path.join(labels_dir, "**", pat), recursive=True)
            if hits:
                return hits[0]
    p = os.path.join(patient_dir, f"{pid}-seg.nii.gz")
    return p if os.path.exists(p) else None


def load_local_cpu(pid, patient_dir, *, labels_dir=None, suffixes=None, with_seg=False,
                   fg_thresh=0.02, max_fg=200_000):
    """CPU decode a local BraTS-style patient (nii dir). Returns (cpu_bundle, seg_np|None). T2 is
    optional (skipped if missing). Seg (if requested) prefers corrected `labels_dir`."""
    import nibabel as nib
    suffixes = suffixes or LOCAL_SUFFIX
    bundle = {}
    for m, suf in suffixes.items():
        p = os.path.join(patient_dir, f"{pid}-{suf}.nii.gz")
        if not os.path.exists(p):
            continue
        img = nib.load(p)
        bundle[m] = _cpu_payload(img.get_fdata(), img.affine, m, LOCAL_SERIES.get(m, 0), pid,
                                 fg_thresh=fg_thresh, max_fg=max_fg)
    seg = None
    if with_seg:
        segp = _find_seg(pid, patient_dir, labels_dir)
        if segp:
            seg = np.nan_to_num(nib.load(segp).get_fdata(), copy=False).astype(np.int16)
    return bundle, seg


def load_local_bundle(pid, patient_dir, *, device, **kw):
    """Decode + place a local patient in one call. Returns (gpu_bundle, seg_np|None)."""
    cpu_bundle, seg = load_local_cpu(pid, patient_dir, **kw)
    return place_bundle(cpu_bundle, device), seg


# ---------------------------------------------------------------------------
# Rotating GPU cache (CPU-loader / GPU-placer split)
# ---------------------------------------------------------------------------

@dataclass
class CacheSlot(Generic[T]):
    value: T
    source_key: str
    remaining_life: int
    age: int = 0
    refresh_count: int = 0


class JitteredRotatingCache(Generic[T]):
    """Bounded rotating cache. `loader(key)` runs on prefetch threads (CPU-only, background);
    `placer(raw)` runs on the main thread at warm/refresh (GPU placement). Jittered lifetimes +
    capped per-step refresh keep turnover desynchronized and off the critical path."""

    def __init__(self, keys, loader: Callable[[str], object], *, size, placer=None,
                 min_life=64, max_life=256, max_refresh_fraction=0.02, seed=0, warmup_log_every=0):
        if not keys:
            raise ValueError("need at least one key")
        if not (0 < max_refresh_fraction <= 1):
            raise ValueError("max_refresh_fraction must be in (0, 1]")
        self.keys = list(keys)
        self.loader = loader
        self.placer = placer or (lambda x: x)
        self.size = min(size, len(self.keys))
        self.min_life, self.max_life = min_life, max_life
        self.max_refresh_fraction = max_refresh_fraction
        self.rng = random.Random(seed)
        self._next_index = 0
        self._key_lock = Lock()
        self._lock = Lock()
        self._stop = False
        self._prefetch_queue = None
        self._workers = []
        self.slots = []
        for idx in range(self.size):
            self.slots.append(self._load_slot())
            if warmup_log_every and ((idx + 1) % warmup_log_every == 0 or idx + 1 == self.size):
                print(f"[rotating-cache] warmed {idx + 1}/{self.size}", flush=True)

    def _life(self):
        return self.rng.randint(self.min_life, self.max_life)

    def _next_key(self):
        with self._key_lock:
            key = self.keys[self._next_index % len(self.keys)]
            self._next_index += 1
        return key

    def _load_slot(self):
        key = self._next_key()
        return CacheSlot(value=self.placer(self.loader(key)), source_key=key, remaining_life=self._life())

    @property
    def max_refresh_per_step(self):
        return max(1, int(self.size * self.max_refresh_fraction))

    def resident(self):
        return [s.value for s in self.slots]

    def sample(self, rng=None):
        r = rng or self.rng
        return self.slots[r.randrange(len(self.slots))].value

    def step(self):
        """Age one step; refresh up to max_refresh_per_step expired slots. Returns #replaced."""
        with self._lock:
            expired = []
            for idx, slot in enumerate(self.slots):
                slot.age += 1
                slot.remaining_life -= 1
                if slot.remaining_life <= 0:
                    expired.append(idx)
            self.rng.shuffle(expired)
            replaced = 0
            for idx in expired[: self.max_refresh_per_step]:
                old = self.slots[idx].refresh_count
                self.slots[idx] = self._replacement_slot()
                self.slots[idx].refresh_count = old + 1
                replaced += 1
            return replaced

    def _replacement_slot(self):
        if self._prefetch_queue is not None:
            try:
                key, raw = self._prefetch_queue.get_nowait()
                return CacheSlot(value=self.placer(raw), source_key=key, remaining_life=self._life())
            except queue.Empty:
                pass
        return self._load_slot()

    def start_prefetch(self, workers=2, depth=8):
        if workers <= 0:
            return
        self._prefetch_queue = queue.Queue(maxsize=depth)
        self._stop = False

        def run():
            while not self._stop:
                key = self._next_key()
                raw = self.loader(key)                    # CPU-only decode, background thread
                while not self._stop:
                    try:
                        self._prefetch_queue.put((key, raw), timeout=0.1)
                        break
                    except queue.Full:
                        time.sleep(0.01)

        self._workers = [Thread(target=run, daemon=True) for _ in range(workers)]
        for w in self._workers:
            w.start()

    def stop_prefetch(self):
        self._stop = True
        for w in self._workers:
            w.join(timeout=0.3)
        self._workers = []
