"""Download the BraTS-2026 challenge segmentation data from Synapse and unzip. Idempotent
(skips a track whose unzip dir already exists + is non-empty). Auth from ~/.synapseConfig
(NEVER commit the token). Run on a CPU node (I/O + unzip only).

    python scripts/cubic/fetch_mets.py [DEST]     # default DEST=~/xmodal/data/brats26
"""
from __future__ import annotations

import os
import sys
import zipfile

import synapseclient

DEST = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/xmodal/data/brats26")

# name -> (synapse id, human description). Covers the 3 seg tracks we target: METS, PED, GoAT.
ENTITIES = {
    "mets_train":  ("syn64919665", "BraTS-2025 MET training scans (~31 GB)"),
    "mets_labels": ("syn65888166", "BraTS-2025 MET corrected GT labels"),
    "ped_train":   ("syn74837563", "BraTS-2026 PED training (~24 GB)"),
    "ped_batch2":  ("syn74916879", "BraTS-2026 PED training batch2 (~3 GB)"),
    "goat_gt":     ("syn60084146", "BraTS-GoAT 2024 training WITH ground truth (~14 GB)"),
    "goat_nogt":   ("syn60084765", "BraTS-GoAT 2024 training WITHOUT ground truth (~21 GB)"),
}

os.makedirs(DEST, exist_ok=True)
syn = synapseclient.Synapse()
syn.login()
print("logged in as", syn.getUserProfile().get("userName"), flush=True)

for name, (sid, desc) in ENTITIES.items():
    out = os.path.join(DEST, name)
    if os.path.isdir(out) and os.listdir(out):
        print(f"[{name}] already present ({len(os.listdir(out))} entries) — skip", flush=True)
        continue
    print(f"[{name}] {desc} — downloading {sid} ...", flush=True)
    e = syn.get(sid, downloadLocation=DEST)
    zp = e.path
    print(f"[{name}] got {os.path.basename(zp)} ({os.path.getsize(zp) / 1e9:.1f} GB); unzipping ...", flush=True)
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(out)
    print(f"[{name}] unzipped -> {out} ({len(os.listdir(out))} entries)", flush=True)

print("DONE", flush=True)
