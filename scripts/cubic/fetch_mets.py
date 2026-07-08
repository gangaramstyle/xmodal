"""Download BraTS-2025 MET challenge training data + corrected labels from Synapse and unzip.

Auth comes from ~/.synapseConfig (NEVER commit the token). Run on a CPU node (I/O + unzip only).

    python scripts/cubic/fetch_mets.py [DEST]
"""
from __future__ import annotations

import os
import sys
import zipfile

import synapseclient

DEST = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/xmodal/data/mets")
# BraTS-2025 MET (labeled training set the 2026 METS task inherits) + corrected GT labels.
ENTITIES = {"train": "syn64919665", "labels": "syn65888166"}  # validation (no GT) = syn64919141

os.makedirs(DEST, exist_ok=True)
syn = synapseclient.Synapse()
syn.login()
print("logged in as", syn.getUserProfile().get("userName"), flush=True)

for name, sid in ENTITIES.items():
    print(f"[{name}] downloading {sid} -> {DEST} ...", flush=True)
    e = syn.get(sid, downloadLocation=DEST)
    zp = e.path
    print(f"[{name}] got {zp} ({os.path.getsize(zp) / 1e9:.1f} GB); unzipping ...", flush=True)
    out = os.path.join(DEST, name)
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(out)
    print(f"[{name}] unzipped -> {out} ({len(os.listdir(out))} entries)", flush=True)

print("DONE", flush=True)
