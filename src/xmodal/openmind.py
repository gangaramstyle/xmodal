"""OpenMind BIDS anat scans, streamed from Cloudflare R2 (bucket 'openmind') on demand (option-b: no bulk
download; pair with a rotating cache to bound memory).

The provenance hinge needs (modality, patient) labels: OpenMind is BIDS-organized as
`ds<ID>/.../sub-XX/anat/sub-XX_<suffix>.nii`, so we key
  - **modality/series** = (dataset, sequence-suffix)  -> same dataset+sequence, different subject is the
    "same-modality / different-patient" positive (same scanner+protocol = same effective series),
  - **patient**         = (dataset, subject).
Geometry: slabs are thin along the native acquisition through-plane (the odd-spacing voxel axis), and the
scan's world-thin axis is set so patch shape can be embedded in PATIENT space (see sampling.world_shape_*).
Creds come from env: OM_KEY / OM_SECRET / OM_ENDPOINT.
"""
from __future__ import annotations

import os
import re
import tempfile

import numpy as np

_BIDS_RE = re.compile(r"(ds\d+).*/(sub-[^/_]+)_([A-Za-z0-9\-]+)\.nii(?:\.gz)?$")
_DEFAULT_ENDPOINT = "https://366d3f9f6455a4cea2ab98aa0a2764da.r2.cloudflarestorage.com"


def om_client():
    import boto3
    return boto3.client(
        "s3", endpoint_url=os.environ.get("OM_ENDPOINT", _DEFAULT_ENDPOINT),
        aws_access_key_id=os.environ["OM_KEY"], aws_secret_access_key=os.environ["OM_SECRET"],
        region_name="auto")


def index_openmind(client=None, bucket="openmind", suffixes=("T1w", "T2w", "FLAIR"), min_mb=1.0):
    """List BIDS .nii scans -> (records, n_modalities, n_patients). Each record carries key + dataset/subject/
    seq + a GLOBAL modality_id over (dataset,seq) and patient_id over (dataset,subject). `suffixes=None` keeps
    every BIDS suffix; `min_mb` drops tiny/degenerate files."""
    client = client or om_client()
    recs = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for o in page.get("Contents", []):
            m = _BIDS_RE.search(o["Key"])
            if not m or o["Size"] < min_mb * 1e6:
                continue
            seq = m.group(3).split("_")[-1]                       # BIDS suffix (T1w, T2w, ...)
            if suffixes and seq not in suffixes:
                continue
            recs.append(dict(key=o["Key"], dataset=m.group(1), subject=m.group(2), seq=seq, size=o["Size"]))
    mod_map, pat_map = {}, {}
    for r in recs:
        r["modality_id"] = mod_map.setdefault((r["dataset"], r["seq"]), len(mod_map))
        r["patient_id"] = pat_map.setdefault((r["dataset"], r["subject"]), len(pat_map))
    return recs, len(mod_map), len(pat_map)


_MANIFEST_URL = "https://huggingface.co/datasets/MIC-DKFZ/OpenMind/resolve/main/openneuro_metadata.csv"
# structural MR contrasts to train on; extend as needed. Non-anat (func/dwi/pet) never enter the anat manifest join.
DEFAULT_MODALITIES = ("T1w", "T2w", "FLAIR", "MP2RAGE", "T1map", "PDw", "FLASH", "SWI", "T2starw",
                      "inplaneT2", "inplaneT1", "UNIT1", "T2starmap", "angio")


def index_openmind_manifest(client=None, bucket="openmind", manifest_url=_MANIFEST_URL, manifest_csv="/tmp/om_meta.csv",
                            modalities=DEFAULT_MODALITIES, min_mb=1.0):
    """Index OpenMind via the openneuro_metadata.csv MANIFEST joined to the actual R2 objects — reaches ~2.4x more
    images than the BIDS-suffix regex (incl. multi-entity filenames + all structural contrasts) with clean MR-technique
    labels + scanner metadata. Returns (records, n_modalities, n_patients). Each record: key/image_path/dataset/subject/
    modality/size + scanner (manufacturer/model/field)/quality/brain_extract, and GLOBAL modality_id over
    (dataset,modality) [= the series label] + patient_id over (dataset,subject). R2 key = '<ds>/OpenMind/<image_path>'."""
    import csv as _csv, io as _io, urllib.request
    client = client or om_client()
    keyset = {}                                                    # normalized image_path -> (key, size)
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for o in page.get("Contents", []):
            mm = re.match(r"ds\d+/OpenMind/(.*)", o["Key"])
            keyset[mm.group(1) if mm else o["Key"]] = (o["Key"], o["Size"])
    if manifest_csv and os.path.exists(manifest_csv):
        rows = list(_csv.DictReader(open(manifest_csv)))
    else:
        raw = urllib.request.urlopen(manifest_url, timeout=180).read().decode()
        if manifest_csv:
            open(manifest_csv, "w").write(raw)
        rows = list(_csv.DictReader(_io.StringIO(raw)))
    modset = set(modalities) if modalities else None
    recs = []
    for r in rows:
        p = r["image_path"]
        if p not in keyset:
            continue
        key, size = keyset[p]
        if size < min_mb * 1e6 or (modset and r["modality"] not in modset):
            continue
        parts = p.split("/")
        recs.append(dict(key=key, image_path=p, dataset=parts[0], subject=parts[1], modality=r["modality"], size=size,
                         manufacturer=r.get("manufacturer", ""), model=r.get("model_name", ""),
                         field=r.get("magnetic_field_strength", ""), quality=r.get("image_quality_score", ""),
                         brain_extract=r.get("is_brain_extract", "")))
    mod_map, pat_map = {}, {}
    for r in recs:
        r["modality_id"] = mod_map.setdefault((r["dataset"], r["modality"]), len(mod_map))  # series = (dataset, modality)
        r["patient_id"] = pat_map.setdefault((r["dataset"], r["subject"]), len(pat_map))
    return recs, len(mod_map), len(pat_map)


def _plane_from_header(client, key, bucket="openmind", nbytes=65536, iso_ratio=1.3):
    """Native acquisition plane from just the NIfTI header (one small Range GET, no full download). Decompresses
    the first gzip block to read the 352-byte header affine. Returns 'axial'|'coronal'|'sagittal' or None."""
    import zlib, io as _io, nibabel as nib
    body = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{nbytes - 1}")["Body"].read()
    if key.endswith(".gz"):
        try:
            body = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(body, 4096)
        except Exception:
            return None
    if len(body) < 352:
        return None
    try:
        R = np.asarray(nib.Nifti1Header.from_fileobj(_io.BytesIO(body[:352])).get_best_affine(), float)[:3, :3]
    except Exception:
        return None
    sp = np.linalg.norm(R, axis=0)
    thick = int(np.argmax(np.abs(R[2, :3]))) if sp.max() / max(sp.min(), 1e-6) < iso_ratio else int(np.argmax(np.abs(sp - np.median(sp))))
    return {0: "sagittal", 1: "coronal", 2: "axial"}[int(np.argmax(np.abs(R[:, thick])))]


def probe_planes(recs, client=None, cache_path=None, workers=24):
    """Attach rec['plane'] (native acquisition plane) to each record via a threaded header-only probe. Cached to
    `cache_path` (JSON key->plane) so it's computed once and reused across runs."""
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    client = client or om_client()
    cache = {}
    if cache_path and os.path.exists(cache_path):
        cache = json.load(open(cache_path))
    todo = [r for r in recs if r["key"] not in cache]
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_plane_from_header, client, r["key"]): r for r in todo}
            for f in as_completed(futs):
                cache[futs[f]["key"]] = f.result() or "axial"
        if cache_path:
            json.dump(cache, open(cache_path, "w"))
    for r in recs:
        r["plane"] = cache.get(r["key"], "axial")
    return recs


def sampling_weights(recs, contrast_alpha=0.5, plane_boost=None):
    """Per-record sampling weight = contrast inverse-frequency (count**-alpha; alpha=0 uniform, 1 full-inverse) x a
    per-plane boost (default: axial 1, coronal 3, sagittal 6 to counter OpenMind's axial dominance). Feed to a weighted
    pool draw so rare contrasts + non-axial acquisitions are over-represented."""
    from collections import Counter
    plane_boost = plane_boost or {"axial": 1.0, "coronal": 3.0, "sagittal": 6.0}
    cc = Counter(r["modality"] for r in recs)
    w = np.array([cc[r["modality"]] ** (-contrast_alpha) * plane_boost.get(r.get("plane", "axial"), 1.0) for r in recs], float)
    return w / w.sum()


def load_openmind_raw(rec, client=None):
    """Stream one scan's bytes -> (volume_np [D,H,W] float32, affine 4x4, spacing) on CPU. Thread-safe part
    (no GPU) for a prefetch worker."""
    import nibabel as nib
    client = client or om_client()
    raw = client.get_object(Bucket="openmind", Key=rec["key"])["Body"].read()
    suf = ".nii.gz" if rec["key"].endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as f:
        f.write(raw); path = f.name
    try:
        img = nib.load(path)
        vol = np.asanyarray(img.dataobj).astype(np.float32)
        if vol.ndim > 3:
            vol = vol.reshape(*vol.shape[:3], -1)[..., 0]        # drop 4th dim (e.g. a single time/echo)
        vol = np.nan_to_num(vol, copy=False)
        return vol, np.asarray(img.affine, np.float32), tuple(float(z) for z in img.header.get_zooms()[:3])
    finally:
        os.unlink(path)


def place_openmind(raw, rec, device="cuda", iso_ratio=1.3):
    """(volume, affine, spacing) + record -> CachedScan on `device` with native-plane thick axis and
    provenance labels (series_idx = modality_id, patient = str(patient_id)). ISOTROPIC 3D scans (no true
    acquisition plane, max/min spacing < iso_ratio) default to AXIAL: thin along the voxel axis most aligned
    with world S-I."""
    from xmodal import sampling as S
    vol, aff, spacing = raw
    sp = np.asarray(spacing, float)
    if float(sp.max() / max(sp.min(), 1e-6)) < iso_ratio:
        thick = int(np.argmax(np.abs(np.asarray(aff)[2, :3])))   # world-Z-aligned voxel axis (axial default)
    else:
        thick = S.native_thru_plane(spacing)
    return S.to_device_scan(vol, aff, modality=rec.get("modality", rec.get("seq", "?")), device=device, thick_axis=thick,
                            series_idx=rec["modality_id"], patient=str(rec["patient_id"]))


def load_openmind_scan(rec, device="cuda", client=None):
    """Convenience: stream + place in one call (single-volume CachedScan)."""
    return place_openmind(load_openmind_raw(rec, client=client), rec, device=device)
