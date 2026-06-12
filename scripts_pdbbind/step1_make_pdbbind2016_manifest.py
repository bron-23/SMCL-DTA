#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create PDBbind v2016 manifest for SMCL-DTA processing.

Input:
  /data_C/sdb1/lww/pdbbind/raw/v2016/
  /data_C/sdb1/lww/pdbbind/raw/v2016/index/

Output:
  /data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_all_manifest.csv
  /data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_trainval_manifest.csv
  /data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_core_manifest.csv
  /data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_train_manifest.csv
  /data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_val_manifest.csv

Notes:
  - Label is PDBbind pKa / pKd / pKi value from the index file.
  - Core set is excluded from train/val.
  - Train/val split is random but fixed by seed.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_index_file(index_file: Path, source_name: str):
    rows = []
    with open(index_file, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            pdb_id = parts[0].lower()

            # PDBbind data index format is usually:
            # PDBcode resolution release_year -logKd/Ki Kd/Ki ligand_name
            try:
                label = float(parts[3])
            except Exception:
                continue

            rows.append({
                "pdb_id": pdb_id,
                "label": label,
                "index_source": source_name,
                "raw_index_line": line,
            })

    return pd.DataFrame(rows)


def add_paths(df: pd.DataFrame, v2016_dir: Path):
    protein_paths = []
    pocket_paths = []
    ligand_sdf_paths = []
    ligand_mol2_paths = []
    has_all = []

    for pdb_id in df["pdb_id"]:
        d = v2016_dir / pdb_id
        protein = d / f"{pdb_id}_protein.pdb"
        pocket = d / f"{pdb_id}_pocket.pdb"
        ligand_sdf = d / f"{pdb_id}_ligand.sdf"
        ligand_mol2 = d / f"{pdb_id}_ligand.mol2"

        protein_paths.append(str(protein))
        pocket_paths.append(str(pocket))
        ligand_sdf_paths.append(str(ligand_sdf))
        ligand_mol2_paths.append(str(ligand_mol2))

        has_all.append(
            protein.exists()
            and pocket.exists()
            and (ligand_sdf.exists() or ligand_mol2.exists())
        )

    df["protein_pdb"] = protein_paths
    df["pocket_pdb"] = pocket_paths
    df["ligand_sdf"] = ligand_sdf_paths
    df["ligand_mol2"] = ligand_mol2_paths
    df["has_required_files"] = has_all
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data_C/sdb1/lww/pdbbind")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=int, default=1000)
    args = parser.parse_args()

    root = Path(args.root)
    v2016_dir = root / "raw" / "v2016"
    index_dir = v2016_dir / "index"
    out_dir = root / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    general_file = index_dir / "INDEX_general_PL_data.2016"
    refined_file = index_dir / "INDEX_refined_data.2016"
    core_file = index_dir / "INDEX_core_data.2016"

    print("[INFO] general:", general_file)
    print("[INFO] refined:", refined_file)
    print("[INFO] core:", core_file)

    general = parse_index_file(general_file, "general")
    refined = parse_index_file(refined_file, "refined")
    core = parse_index_file(core_file, "core")

    print("[INFO] parsed general:", len(general))
    print("[INFO] parsed refined:", len(refined))
    print("[INFO] parsed core:", len(core))

    # Combine general + refined, then remove duplicates by pdb_id.
    # Refined/core are subsets of general in many PDBbind releases.
    # For the full v2016 train/val pool, use general PL entries and mark membership.
    all_df = general.copy()
    all_df["is_refined"] = all_df["pdb_id"].isin(set(refined["pdb_id"]))
    all_df["is_core"] = all_df["pdb_id"].isin(set(core["pdb_id"]))

    all_df = add_paths(all_df, v2016_dir)

    # Core set for external testing.
    core_df = all_df[all_df["is_core"]].copy()
    core_df["split"] = "core2016"

    # Train/val pool excludes core.
    trainval = all_df[~all_df["is_core"]].copy()
    trainval = trainval[trainval["has_required_files"]].copy()

    # Fixed 1000 validation samples, following the common GIGN-style split size.
    if len(trainval) <= args.val_size:
        raise ValueError(f"Trainval too small: {len(trainval)} <= val_size {args.val_size}")

    train_df, val_df = train_test_split(
        trainval,
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
    )
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"

    final_df = pd.concat([train_df, val_df, core_df], ignore_index=True)

    # Write files.
    all_path = out_dir / "pdbbind2016_all_manifest.csv"
    trainval_path = out_dir / "pdbbind2016_trainval_manifest.csv"
    core_path = out_dir / "pdbbind2016_core_manifest.csv"
    train_path = out_dir / "pdbbind2016_train_manifest.csv"
    val_path = out_dir / "pdbbind2016_val_manifest.csv"
    final_path = out_dir / "pdbbind2016_final_manifest.csv"

    all_df.to_csv(all_path, index=False)
    trainval.to_csv(trainval_path, index=False)
    core_df.to_csv(core_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    final_df.to_csv(final_path, index=False)

    print("=" * 100)
    print("[SUMMARY]")
    print("all_df:", len(all_df))
    print("has_required_files:", int(all_df["has_required_files"].sum()))
    print("refined:", int(all_df["is_refined"].sum()))
    print("core:", int(all_df["is_core"].sum()))
    print("trainval feature-file complete:", len(trainval))
    print("train:", len(train_df))
    print("val:", len(val_df))
    print("core2016:", len(core_df))
    print()
    print(final_df["split"].value_counts())
    print()
    print("[OUT]", all_path)
    print("[OUT]", trainval_path)
    print("[OUT]", core_path)
    print("[OUT]", train_path)
    print("[OUT]", val_path)
    print("[OUT]", final_path)

    # Quick label stats.
    print()
    print("[LABEL STATS]")
    print(final_df.groupby("split")["label"].describe())


if __name__ == "__main__":
    main()