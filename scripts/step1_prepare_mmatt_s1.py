#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1: Prepare MMAtt-DTA Supplementary File S1 for SMCL-DTA external validation.

Input:
    Supplementary File 1.csv from MMAtt-DTA

Outputs:
    external_validation/mmatt_s1/mmatt_s1_all_clean.csv
    external_validation/mmatt_s1/mmatt_s1_kinase.csv
    external_validation/mmatt_s1/mmatt_s1_targets.csv
    external_validation/mmatt_s1/mmatt_s1_compounds.csv
    external_validation/mmatt_s1/mmatt_s1_summary.txt
"""

import argparse
from pathlib import Path

import pandas as pd


def safe_join_unique(values):
    values = [str(v) for v in values if pd.notna(v)]
    values = sorted(set(values))
    return ";".join(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to MMAtt-DTA Supplementary File 1.csv"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="external_validation/mmatt_s1",
        help="Output directory"
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="pchembl_value",
        help="Regression label column. Default: pchembl_value"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading: {input_path}")
    df = pd.read_csv(input_path)

    print("[INFO] Original columns:")
    print(list(df.columns))

    required_cols = [
        "canonical_smiles",
        "uniprot_id",
        args.label_col,
        "standard_type",
        "standard_value",
        "protein_class",
        "pref_name",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in S1 file: {missing_cols}")

    # Basic cleaning
    df = df.copy()
    df["canonical_smiles"] = df["canonical_smiles"].astype(str).str.strip()
    df["uniprot_id"] = df["uniprot_id"].astype(str).str.strip()
    df["protein_class"] = df["protein_class"].astype(str).str.strip()
    df["pref_name"] = df["pref_name"].astype(str).str.strip()
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")

    # Convert to SMCL-DTA-friendly raw format
    clean = pd.DataFrame({
        "compound_iso_smiles": df["canonical_smiles"],
        "target_id": df["uniprot_id"],
        "affinity": df[args.label_col],
        "standard_type": df["standard_type"],
        "standard_value": df["standard_value"],
        "protein_class": df["protein_class"],
        "target_name": df["pref_name"],
    })

    # Remove invalid rows
    clean = clean.dropna(subset=["compound_iso_smiles", "target_id", "affinity"])
    clean = clean[
        (clean["compound_iso_smiles"] != "") &
        (clean["compound_iso_smiles"].str.lower() != "nan") &
        (clean["target_id"] != "") &
        (clean["target_id"].str.lower() != "nan")
    ].copy()

    print(f"[INFO] Valid rows before deduplication: {len(clean)}")

    # Deduplicate compound-target pairs by median affinity
    clean_dedup = (
        clean
        .groupby(["compound_iso_smiles", "target_id"], as_index=False)
        .agg({
            "affinity": "median",
            "standard_type": safe_join_unique,
            "standard_value": "median",
            "protein_class": safe_join_unique,
            "target_name": safe_join_unique,
        })
    )

    print(f"[INFO] Valid rows after deduplication: {len(clean_dedup)}")

    # Save all cleaned data
    all_clean_path = out_dir / "mmatt_s1_all_clean.csv"
    clean_dedup.to_csv(all_clean_path, index=False)

    # Extract kinase subset
    kinase = clean_dedup[
        clean_dedup["protein_class"].str.contains("kinase", case=False, na=False)
    ].copy()

    kinase_path = out_dir / "mmatt_s1_kinase.csv"
    kinase.to_csv(kinase_path, index=False)

    # Target list
    targets = (
        clean_dedup[["target_id", "target_name", "protein_class"]]
        .drop_duplicates()
        .sort_values("target_id")
    )
    targets_path = out_dir / "mmatt_s1_targets.csv"
    targets.to_csv(targets_path, index=False)

    # Compound list
    compounds = (
        clean_dedup[["compound_iso_smiles"]]
        .drop_duplicates()
        .sort_values("compound_iso_smiles")
    )
    compounds_path = out_dir / "mmatt_s1_compounds.csv"
    compounds.to_csv(compounds_path, index=False)

    # Summary
    summary_lines = []
    summary_lines.append("MMAtt-DTA S1 preprocessing summary")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Input file: {input_path}")
    summary_lines.append(f"Original rows: {len(df)}")
    summary_lines.append(f"Valid rows before deduplication: {len(clean)}")
    summary_lines.append(f"Valid rows after deduplication: {len(clean_dedup)}")
    summary_lines.append("")
    summary_lines.append(f"Unique compounds: {clean_dedup['compound_iso_smiles'].nunique()}")
    summary_lines.append(f"Unique targets: {clean_dedup['target_id'].nunique()}")
    summary_lines.append("")
    summary_lines.append("Kinase subset")
    summary_lines.append("-" * 60)
    summary_lines.append(f"Kinase rows: {len(kinase)}")
    summary_lines.append(f"Kinase unique compounds: {kinase['compound_iso_smiles'].nunique()}")
    summary_lines.append(f"Kinase unique targets: {kinase['target_id'].nunique()}")
    summary_lines.append("")
    summary_lines.append("Protein class counts")
    summary_lines.append("-" * 60)
    summary_lines.append(str(clean_dedup["protein_class"].value_counts().head(30)))
    summary_lines.append("")
    summary_lines.append("Standard type counts")
    summary_lines.append("-" * 60)
    summary_lines.append(str(clean_dedup["standard_type"].value_counts().head(30)))
    summary_lines.append("")
    summary_lines.append("Output files")
    summary_lines.append("-" * 60)
    summary_lines.append(str(all_clean_path))
    summary_lines.append(str(kinase_path))
    summary_lines.append(str(targets_path))
    summary_lines.append(str(compounds_path))

    summary_path = out_dir / "mmatt_s1_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("[DONE] Step 1 finished.")
    print(f"[OUT] All cleaned data: {all_clean_path}")
    print(f"[OUT] Kinase subset: {kinase_path}")
    print(f"[OUT] Target list: {targets_path}")
    print(f"[OUT] Compound list: {compounds_path}")
    print(f"[OUT] Summary: {summary_path}")


if __name__ == "__main__":
    main()