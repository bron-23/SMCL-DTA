#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a small pilot manifest for PDBbind v2016.

Input:
  pdbbind2016_valid_resplit_manifest_with_seq_valid.csv

Output:
  pdbbind2016_pilot_manifest.csv
  pdbbind2016_pilot_train_manifest.csv
  pdbbind2016_pilot_val_manifest.csv
  pdbbind2016_pilot_core2016_manifest.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="/data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_valid_resplit_manifest_with_seq_valid.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="/data_C/sdb1/lww/pdbbind/manifests",
    )
    parser.add_argument("--train_n", type=int, default=2000)
    parser.add_argument("--val_n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    core = df[df["split"] == "core2016"].copy()

    if len(train) < args.train_n:
        raise ValueError(f"Not enough train samples: {len(train)} < {args.train_n}")
    if len(val) < args.val_n:
        raise ValueError(f"Not enough val samples: {len(val)} < {args.val_n}")

    train_pilot = train.sample(n=args.train_n, random_state=args.seed).copy()
    val_pilot = val.sample(n=args.val_n, random_state=args.seed).copy()
    core_pilot = core.copy()

    train_pilot["split"] = "train"
    val_pilot["split"] = "val"
    core_pilot["split"] = "core2016"

    pilot = pd.concat([train_pilot, val_pilot, core_pilot], ignore_index=True)

    out_all = out_dir / "pdbbind2016_pilot_manifest.csv"
    out_train = out_dir / "pdbbind2016_pilot_train_manifest.csv"
    out_val = out_dir / "pdbbind2016_pilot_val_manifest.csv"
    out_core = out_dir / "pdbbind2016_pilot_core2016_manifest.csv"

    pilot.to_csv(out_all, index=False)
    train_pilot.to_csv(out_train, index=False)
    val_pilot.to_csv(out_val, index=False)
    core_pilot.to_csv(out_core, index=False)

    print("=" * 100)
    print("[SUMMARY]")
    print(pilot["split"].value_counts())
    print()
    print("[LABEL STATS]")
    print(pilot.groupby("split")["label"].describe())
    print()
    print("[SEQ LEN STATS]")
    print(pilot.groupby("split")["protein_seq_len"].describe())
    print()
    print("[OUT]", out_all)
    print("[OUT]", out_train)
    print("[OUT]", out_val)
    print("[OUT]", out_core)


if __name__ == "__main__":
    main()
