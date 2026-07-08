"""Data layer: BraTS (HF) loader + rotating GPU cache for datasets that exceed VRAM.

The rotating cache (ported from brats2026/data/rotating_cache.py) keeps `size` patient bundles
resident on the GPU; each slot has a jittered lifetime and only a small fraction of slots may
refresh per step, so the resident set desynchronizes and turnover never stalls a step. Background
prefetch threads decode the next volumes (nibabel releases the GIL) off the critical path.
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

# ---------------------------------------------------------------------------
# BraTS-on-HF loader (factored out of the notebook)
# ---------------------------------------------------------------------------

BRATS_REPO = "Spirit-26/BraTS-2024-Complete"
BRATS_SUFFIX = {"t1": "t1n", "t1c": "t1c", "t2": "t2w", "flair": "t2f"}
BRATS_SERIES = {"t1": 0, "t1c": 1, "t2": 2, "flair": 3}


def discover_brats_patients(repo=BRATS_REPO, split="train", limit=None):
    from huggingface_hub import list_repo_files
    files = list_repo_files(repo, repo_type="dataset")
    pref = f"BraTS-GLI/{split}/"
    pats = sorted({f.split("/")[2] for f in files if f.startswith(pref) and f.count("/") >= 3})
    return pats[:limit] if limit else pats


def load_brats_bundle(pid, *, device, repo=BRATS_REPO, split="train", modalities=None):
    """Load one BraTS patient -> {modality: CachedScan} on `device` (co-registered)."""
    import nibabel as nib
    from huggingface_hub import hf_hub_download
    mods = modalities or list(BRATS_SUFFIX)
    base = f"BraTS-GLI/{split}/{pid}/{pid}-"
    bundle = {}
    for m in mods:
        img = nib.load(hf_hub_download(repo, base + BRATS_SUFFIX[m] + ".nii.gz", repo_type="dataset"))
        vol = np.nan_to_num(img.get_fdata(), copy=False)                 # sanitize NaN/Inf
        bundle[m] = S.to_device_scan(vol, img.affine, modality=m, series_idx=BRATS_SERIES[m],
                                     patient=pid, device=device)
    return bundle


# ---------------------------------------------------------------------------
# Rotating GPU cache (ported from brats2026/data/rotating_cache.py)
# ---------------------------------------------------------------------------

@dataclass
class CacheSlot(Generic[T]):
    value: T
    source_key: str
    remaining_life: int
    age: int = 0
    refresh_count: int = 0


class JitteredRotatingCache(Generic[T]):
    """Bounded rotating cache with explicit lifetime jitter and capped per-step refresh."""

    def __init__(self, keys, loader: Callable[[str], T], *, size, min_life=16, max_life=128,
                 max_refresh_fraction=0.02, seed=0, warmup_log_every=0):
        if not keys:
            raise ValueError("need at least one key")
        if not (0 < max_refresh_fraction <= 1):
            raise ValueError("max_refresh_fraction must be in (0, 1]")
        self.keys = list(keys)
        self.loader = loader
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
        return CacheSlot(value=self.loader(key), source_key=key, remaining_life=self._life())

    @property
    def max_refresh_per_step(self):
        return max(1, int(self.size * self.max_refresh_fraction))

    def resident(self):
        """Current resident values (pass to the sampler each step)."""
        return [s.value for s in self.slots]

    def sample(self, rng=None):
        r = rng or self.rng
        return self.slots[r.randrange(len(self.slots))].value

    def step(self):
        """Age one step, refresh up to max_refresh_per_step expired slots. Returns #replaced."""
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
                key, value = self._prefetch_queue.get_nowait()
                return CacheSlot(value=value, source_key=key, remaining_life=self._life())
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
                value = self.loader(key)
                while not self._stop:
                    try:
                        self._prefetch_queue.put((key, value), timeout=0.1)
                        break
                    except queue.Full:
                        time.sleep(0.01)

        self._workers = [Thread(target=run, daemon=True) for _ in range(workers)]
        for w in self._workers:
            w.start()

    def stop_prefetch(self):
        self._stop = True
        for w in self._workers:
            w.join(timeout=0.2)
        self._workers = []
