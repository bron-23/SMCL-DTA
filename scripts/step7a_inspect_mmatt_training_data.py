#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 7A: Inspect candidate MMAtt-DTA training data files.

Usage:
python scripts/step7a_inspect_mmatt_training_data.py \
  --input /path/to/mmatt_training_file.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def read_table(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in [".csv"]:
        return pd.read_csv(path)
    elif suffix in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t")
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif suffix in [".pkl", ".pickle"]:
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    path = Path(args.input)

    print("=" * 100)
    print(f"[INFO] Inspecting: {path}")
    print("=" * 100)

    df = read_table(path)

    print(f"[INFO] Shape: {df.shape}")
    print("\n[INFO] Columns:")
    for i, c in enumerate(df.columns):
        print(f"{i:03d}: {c}")

    print("\n[INFO] Head:")
    print(df.head())

    print("\n[INFO] Missing values:")
    print(df.isna().sum().sort_values(ascending=False).head(30))

    lower_cols = {c.lower(): c for c in df.columns}

    candidate_smiles = [c for c in df.columns if any(k in c.lower() for k in ["smiles", "canonical"])]
    candidate_target = [c for c in df.columns if any(k in c.lower() for k in ["uniprot", "target", "protein"])]
    candidate_label = [c for c in df.columns if any(k in c.lower() for k in ["pchembl", "label", "affinity", "value", "activity", "standard"])]
    candidate_class = [c for c in df.columns if any(k in c.lower() for k in ["class", "family", "superfamily"])]

    print("\n[INFO] Candidate SMILES columns:")
    print(candidate_smiles)

    print("\n[INFO] Candidate target/protein columns:")
    print(candidate_target)

    print("\n[INFO] Candidate label columns:")
    print(candidate_label)

    print("\n[INFO] Candidate protein class columns:")
    print(candidate_class)

    for c in candidate_class:
        try:
            print(f"\n[INFO] Value counts for class-like column: {c}")
            print(df[c].value_counts(dropna=False).head(30))
        except Exception as e:
            print(f"[WARN] Could not summarize {c}: {e}")

    for c in candidate_label:
        try:
            numeric = pd.to_numeric(df[c], errors="coerce")
            if numeric.notna().sum() > 0:
                print(f"\n[INFO] Numeric summary for label-like column: {c}")
                print(numeric.describe())
        except Exception as e:
            print(f"[WARN] Could not summarize {c}: {e}")

    if args.out:
        Path(args.out).write_text("Inspection completed. See terminal output.\n", encoding="utf-8")


if __name__ == "__main__":
    main()