#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3: Prepare SMCL-DTA-compatible raw CSV for MMAtt-DTA S1 kinase external validation.

Input:
    external_validation/mmatt_s1/mmatt_s1_kinase_with_sequence.csv

Outputs:
    external_validation/mmatt_s1/mmatt_s1_kinase_smcl_ready.csv
    external_validation/mmatt_s1/mmatt_s1_kinase_invalid_smiles.csv
    external_validation/mmatt_s1/mmatt_s1_step3_summary.txt

Also creates:
    external_validation/mmatt_s1/smcl_raw/data_test.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem


def is_valid_smiles(smiles: str) -> bool:
    if pd.isna(smiles):
        return False
    smiles = str(smiles).strip()
    if smiles == "" or smiles.lower() == "nan":
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="external_validation/mmatt_s1/mmatt_s1_kinase_with_sequence.csv",
        help="Input CSV from Step 2"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="external_validation/mmatt_s1",
        help="Output directory"
    )
    parser.add_argument(
        "--canonicalize",
        action="store_true",
        help="Whether to canonicalize SMILES using RDKit"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading: {input_path}")
    df = pd.read_csv(input_path)

    required_cols = ["compound_iso_smiles", "target_sequence", "affinity"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"[INFO] Original rows: {len(df)}")

    # Basic cleaning
    df = df.copy()
    df["compound_iso_smiles"] = df["compound_iso_smiles"].astype(str).str.strip()
    df["target_sequence"] = df["target_sequence"].astype(str).str.strip()
    df["affinity"] = pd.to_numeric(df["affinity"], errors="coerce")

    df = df.dropna(subset=["compound_iso_smiles", "target_sequence", "affinity"])
    df = df[
        (df["compound_iso_smiles"] != "") &
        (df["compound_iso_smiles"].str.lower() != "nan") &
        (df["target_sequence"] != "") &
        (df["target_sequence"].str.lower() != "nan")
    ].copy()

    print(f"[INFO] Rows after basic cleaning: {len(df)}")

    # Validate SMILES
    print("[INFO] Validating SMILES with RDKit...")
    valid_mask = df["compound_iso_smiles"].apply(is_valid_smiles)

    valid_df = df[valid_mask].copy()
    invalid_df = df[~valid_mask].copy()

    print(f"[INFO] Valid SMILES rows: {len(valid_df)}")
    print(f"[INFO] Invalid SMILES rows: {len(invalid_df)}")

    # Canonicalize if requested
    if args.canonicalize:
        print("[INFO] Canonicalizing SMILES...")
        valid_df["compound_iso_smiles"] = valid_df["compound_iso_smiles"].apply(canonicalize_smiles)
        valid_df = valid_df[valid_df["compound_iso_smiles"] != ""].copy()

    # Reorder to SMCL-DTA-compatible format
    preferred_cols = [
        "compound_iso_smiles",
        "target_sequence",
        "affinity",
        "target_id",
        "target_name",
        "protein_class",
        "standard_type",
        "standard_value",
    ]

    existing_cols = [c for c in preferred_cols if c in valid_df.columns]
    remaining_cols = [c for c in valid_df.columns if c not in existing_cols]
    valid_df = valid_df[existing_cols + remaining_cols]

    # Save full SMCL-ready file
    ready_path = out_dir / "mmatt_s1_kinase_smcl_ready.csv"
    valid_df.to_csv(ready_path, index=False)

    invalid_path = out_dir / "mmatt_s1_kinase_invalid_smiles.csv"
    invalid_df.to_csv(invalid_path, index=False)

    # Create SMCL raw folder for later preprocessing
    smcl_raw_dir = out_dir / "smcl_raw"
    smcl_raw_dir.mkdir(parents=True, exist_ok=True)

    # External validation is test-only
    data_test_path = smcl_raw_dir / "data_test.csv"
    valid_df.to_csv(data_test_path, index=False)

    # Also create empty train file for compatibility with some old preprocessors
    data_train_path = smcl_raw_dir / "data_train.csv"
    valid_df.iloc[:0].to_csv(data_train_path, index=False)

    # Summary
    summary_lines = []
    summary_lines.append("MMAtt-DTA S1 Step 3 SMCL raw preparation summary")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Input file: {input_path}")
    summary_lines.append(f"Original rows: {len(df)}")
    summary_lines.append(f"Valid SMILES rows: {len(valid_df)}")
    summary_lines.append(f"Invalid SMILES rows: {len(invalid_df)}")
    summary_lines.append("")
    summary_lines.append(f"Unique compounds: {valid_df['compound_iso_smiles'].nunique()}")
    summary_lines.append(f"Unique targets: {valid_df['target_id'].nunique() if 'target_id' in valid_df.columns else 'NA'}")
    summary_lines.append("")
    summary_lines.append("Affinity statistics")
    summary_lines.append("-" * 70)
    summary_lines.append(str(valid_df["affinity"].describe()))
    summary_lines.append("")
    summary_lines.append("Sequence length statistics")
    summary_lines.append("-" * 70)
    summary_lines.append(str(valid_df["target_sequence"].astype(str).str.len().describe()))
    summary_lines.append("")
    summary_lines.append("Output files")
    summary_lines.append("-" * 70)
    summary_lines.append(str(ready_path))
    summary_lines.append(str(invalid_path))
    summary_lines.append(str(data_test_path))
    summary_lines.append(str(data_train_path))

    summary_path = out_dir / "mmatt_s1_step3_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("[DONE] Step 3 finished.")
    print(f"[OUT] SMCL-ready full file: {ready_path}")
    print(f"[OUT] Invalid SMILES file: {invalid_path}")
    print(f"[OUT] SMCL raw test file: {data_test_path}")
    print(f"[OUT] Empty train file: {data_train_path}")
    print(f"[OUT] Summary: {summary_path}")


if __name__ == "__main__":
    main()