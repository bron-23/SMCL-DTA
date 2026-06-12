#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-split RDKit-valid PDBbind v2016 manifest.

Goal:
  - Keep all core2016 samples as external test.
  - Re-split non-core valid samples into train and val.
  - Ensure val has exactly 1000 samples.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--valid_manifest",
        default="/data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_final_manifest_with_smiles_valid.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="/data_C/sdb1/lww/pdbbind/manifests",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.valid_manifest)

    core = df[df["split"] == "core2016"].copy()
    pool = df[df["split"] != "core2016"].copy()

    train, val = train_test_split(
        pool,
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
    )

    train = train.copy()
    val = val.copy()
    core = core.copy()

    train["split"] = "train"
    val["split"] = "val"
    core["split"] = "core2016"

    final = pd.concat([train, val, core], ignore_index=True)

    out_final = out_dir / "pdbbind2016_valid_resplit_manifest.csv"
    out_train = out_dir / "pdbbind2016_valid_train_manifest.csv"
    out_val = out_dir / "pdbbind2016_valid_val_manifest.csv"
    out_core = out_dir / "pdbbind2016_valid_core2016_manifest.csv"

    final.to_csv(out_final, index=False)
    train.to_csv(out_train, index=False)
    val.to_csv(out_val, index=False)
    core.to_csv(out_core, index=False)

    print("=" * 100)
    print("[SUMMARY]")
    print("valid total:", len(df))
    print("train:", len(train))
    print("val:", len(val))
    print("core2016:", len(core))
    print()
    print(final["split"].value_counts())
    print()
    print("[LABEL STATS]")
    print(final.groupby("split")["label"].describe())
    print()
    print("[OUT]", out_final)
    print("[OUT]", out_train)
    print("[OUT]", out_val)
    print("[OUT]", out_core)


if __name__ == "__main__":
    main()







