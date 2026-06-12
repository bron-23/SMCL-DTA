#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import pandas as pd


def detect_id_col(df):
    candidates = [
        "pdb_id", "pdbid", "pdb", "PDB", "PDB_ID", "PDB_code",
        "pdb_code", "code", "id", "ID", "complex_id", "complex"
    ]
    lower_map = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    # 如果找不到，尝试第一列
    return df.columns[0]


def norm_pdb_id(x):
    s = str(x).strip().lower()
    m = re.search(r"[0-9a-z]{4}", s)
    if m:
        return m.group(0)
    return s[:4]


def load_ids(csv_path):
    df = pd.read_csv(csv_path)
    col = detect_id_col(df)
    ids = df[col].map(norm_pdb_id).tolist()
    return df, col, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gign_data_dir", required=True)
    parser.add_argument(
        "--our_manifest",
        default="/data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_valid_resplit_manifest_with_seq_valid.csv",
    )
    args = parser.parse_args()

    gign_dir = Path(args.gign_data_dir)
    our_manifest = Path(args.our_manifest)

    ours = pd.read_csv(our_manifest)
    our_col = detect_id_col(ours)
    ours["_pdb_id_norm"] = ours[our_col].map(norm_pdb_id)
    our_ids = set(ours["_pdb_id_norm"])

    print("=" * 100)
    print("[OUR MANIFEST]")
    print("path:", our_manifest)
    print("shape:", ours.shape)
    print("id_col:", our_col)
    print("unique ids:", len(our_ids))
    if "split" in ours.columns:
        print(ours["split"].value_counts())

    split_files = {
        "train": "train.csv",
        "valid": "valid.csv",
        "test2016": "test2016.csv",
        "test2013": "test2013.csv",
        "test2019": "test2019.csv",
    }

    all_sets = {}

    for split, fname in split_files.items():
        p = gign_dir / fname
        print("=" * 100)
        print("[GIGN]", split, p)

        if not p.exists():
            print("MISSING")
            continue

        df, id_col, ids = load_ids(p)
        ids_set = set(ids)
        overlap = ids_set & our_ids
        missing = ids_set - our_ids

        all_sets[split] = ids_set

        print("shape:", df.shape)
        print("id_col:", id_col)
        print("unique ids:", len(ids_set))
        print("overlap with our v2016 manifest:", len(overlap))
        print("missing from our v2016 manifest:", len(missing))

        print("head ids:", ids[:10])
        if len(missing) > 0:
            print("missing examples:", sorted(list(missing))[:30])

    print("=" * 100)
    print("[LEAKAGE CHECK]")
    names = list(all_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = all_sets[a] & all_sets[b]
            print(f"{a} ∩ {b}: {len(inter)}")
            if inter:
                print(" examples:", sorted(list(inter))[:20])


if __name__ == "__main__":
    main()
