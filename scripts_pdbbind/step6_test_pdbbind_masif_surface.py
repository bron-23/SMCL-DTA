#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_preprocessing_module(path):
    path = str(path)
    module_dir = os.path.dirname(path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location("preprocessing_suf_pdbbind", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    manifest = "/data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_pilot_manifest.csv"
    preprocessing_suf = "/home/lww/learn_project/MGraphDTA-dev/regression/preprocessing_suf.py"

    out_base = Path("/data_C/sdb1/lww/pdbbind/processed/masif_test")
    out_base.mkdir(parents=True, exist_ok=True)

    module = load_preprocessing_module(preprocessing_suf)

    # Important: redirect all temporary/cache directories to PDBbind-specific paths.
    module.PROTEIN_SURFACE_CACHE_DIR = str(out_base / "protein_surface_cache")
    os.makedirs(module.PROTEIN_SURFACE_CACHE_DIR, exist_ok=True)

    module.MASIF_OPTS["tmp_dir"] = str(out_base / "tmp")
    module.MASIF_OPTS["raw_pdb_dir"] = "/data_C/sdb1/lww/pdbbind/raw/v2016"
    module.MASIF_OPTS["ply_chain_dir"] = str(out_base / "ply_chain")
    module.MASIF_OPTS["pdb_chain_dir"] = str(out_base / "pdb_chain")

    for k in ["tmp_dir", "ply_chain_dir", "pdb_chain_dir"]:
        os.makedirs(module.MASIF_OPTS[k], exist_ok=True)

    df = pd.read_csv(manifest)

    # Use a few small/medium examples first.
    test_df = df.sample(n=5, random_state=42).copy()

    print("=" * 100)
    print("[INFO] Testing PDBbind MaSIF protein_surface extraction")
    print("[INFO] module:", preprocessing_suf)
    print("[INFO] manifest:", manifest)
    print("[INFO] cache:", module.PROTEIN_SURFACE_CACHE_DIR)
    print("=" * 100)

    for _, row in test_df.iterrows():
        pdb_id = str(row["pdb_id"]).lower()
        seq = str(row["protein_sequence"])

        # Prefer pocket for PDBbind pilot.
        pocket_pdb = row.get("pocket_pdb", "")
        protein_pdb = row.get("protein_pdb", "")

        pdb_file = pocket_pdb if isinstance(pocket_pdb, str) and os.path.exists(pocket_pdb) else protein_pdb

        print(f"\n[TEST] {pdb_id}")
        print("pdb_file:", pdb_file)
        print("label:", row["label"])
        print("seq_len:", len(seq))

        try:
            feat = module.extract_protein_features(
                protein_id=f"pdbbind_{pdb_id}_pocket",
                protein_sequence=seq,
                pdb_file=pdb_file,
                include_masif=True,
                chain_id=None,
            )

            masif = feat.get("masif", None)
            if masif is None:
                print("[FAIL] no masif key")
                continue

            arr = np.asarray(masif)
            print("[OK] masif shape:", arr.shape)
            print("[OK] finite:", np.isfinite(arr).all())
            print("[OK] abs_sum:", float(np.abs(arr).sum()))
            print("[OK] mean/std:", float(arr.mean()), float(arr.std()))

        except Exception as e:
            print("[ERROR]", type(e).__name__, str(e))


if __name__ == "__main__":
    main()
