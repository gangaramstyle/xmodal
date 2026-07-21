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


def place_openmind(raw, rec, device="cuda"):
    """(volume, affine, spacing) + record -> CachedScan on `device` with native-plane thick axis and
    provenance labels (series_idx = modality_id, patient = str(patient_id))."""
    from xmodal import sampling as S
    vol, aff, spacing = raw
    thick = S.native_thru_plane(spacing)
    return S.to_device_scan(vol, aff, modality=rec["seq"], device=device, thick_axis=thick,
                            series_idx=rec["modality_id"], patient=str(rec["patient_id"]))


def load_openmind_scan(rec, device="cuda", client=None):
    """Convenience: stream + place in one call (single-volume CachedScan)."""
    return place_openmind(load_openmind_raw(rec, client=client), rec, device=device)
