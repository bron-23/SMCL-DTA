#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd

target_csv = "/data_C/sdb1/lww/mmatt_independent_tests_raw_aligned/step9_c_full_surface/C_missing_target_ids.csv"
out_dir = Path("/data_C/sdb1/lww/mmatt_independent_tests_raw_aligned/step9_c_full_surface/alphafold_structures")
out_dir.mkdir(parents=True, exist_ok=True)

targets = pd.read_csv(target_csv)["uniprot_id"].astype(str).tolist()

def url_exists(url, timeout=20):
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status < 400
    except Exception:
        return False

def download(url, path, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = r.read()
    path.write_bytes(data)
    return len(data)

records = []

for uid in targets:
    print("=" * 100)
    print("[TARGET]", uid)

    pdb_path = out_dir / f"{uid}.pdb"
    cif_path = out_dir / f"{uid}.cif"
    meta_path = out_dir / f"{uid}.alphafold_api.json"

    if pdb_path.exists() and pdb_path.stat().st_size > 1000:
        print("[SKIP] existing PDB:", pdb_path)
        records.append({"uniprot_id": uid, "status": "exists", "pdb_path": str(pdb_path), "cif_path": str(cif_path) if cif_path.exists() else ""})
        continue

    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uid}"

    status = "failed"
    pdb_url = ""
    cif_url = ""
    error = ""

    try:
        with urllib.request.urlopen(api_url, timeout=60) as r:
            meta = json.loads(r.read().decode("utf-8"))
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if isinstance(meta, list) and len(meta) > 0:
            rec = meta[0]
            pdb_url = rec.get("pdbUrl", "") or rec.get("bcifUrl", "")
            cif_url = rec.get("cifUrl", "") or rec.get("mmcifUrl", "")

            if pdb_url:
                n = download(pdb_url, pdb_path)
                print("[OK] PDB downloaded:", pdb_path, "bytes=", n)
                status = "pdb_downloaded"

            if cif_url:
                try:
                    n = download(cif_url, cif_path)
                    print("[OK] CIF downloaded:", cif_path, "bytes=", n)
                except Exception as e:
                    print("[WARN] CIF download failed:", e)

    except Exception as e:
        error = repr(e)
        print("[WARN] API download failed:", error)

    # Fallback direct URLs.
    if not pdb_path.exists() or pdb_path.stat().st_size < 1000:
        for version in ["v4", "v3", "v2", "v1"]:
            direct = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_{version}.pdb"
            try:
                n = download(direct, pdb_path)
                print("[OK] fallback PDB downloaded:", direct, "bytes=", n)
                pdb_url = direct
                status = f"fallback_{version}"
                break
            except Exception as e:
                error = repr(e)

    records.append({
        "uniprot_id": uid,
        "status": status,
        "pdb_url": pdb_url,
        "cif_url": cif_url,
        "pdb_path": str(pdb_path) if pdb_path.exists() else "",
        "cif_path": str(cif_path) if cif_path.exists() else "",
        "error": error,
    })

    time.sleep(0.5)

df = pd.DataFrame(records)
df.to_csv(out_dir / "alphafold_download_manifest.csv", index=False)

print("=" * 100)
print(df.to_string(index=False))
print("[OUT]", out_dir / "alphafold_download_manifest.csv")
