#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 5A: Check whether MMAtt-DTA S1 external samples can reuse existing
surface/MaSIF features from Davis/KIBA processed datasets.

This script does NOT generate new surface features.
It only checks overlap of:
    external target_id vs existing protein_id
    external compound_iso_smiles vs existing raw SMILES
"""

import argparse
import glob
from pathlib import Path

import pandas as pd
import torch


def load_pt_protein_ids(pt_path):
    try:
        data, slices = torch.load(pt_path, weights_only=False)
    except TypeError:
        data, slices = torch.load(pt_path)

    if not hasattr(data, "protein_id"):
        return set()

    protein_ids = data.protein_id

    if isinstance(protein_ids, list):
        return set(str(x).strip() for x in protein_ids if str(x).strip())

    try:
        return set(str(x).strip() for x in protein_ids.tolist() if str(x).strip())
    except Exception:
        return set()


def load_raw_smiles_from_dataset_root(dataset_root):
    dataset_root = Path(dataset_root)
    candidate_files = []

    # Common raw locations
    candidate_files.extend(glob.glob(str(dataset_root / "raw" / "*.csv")))
    candidate_files.extend(glob.glob(str(dataset_root / "*.csv")))

    smiles = set()

    for csv_path in candidate_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if "compound_iso_smiles" in df.columns:
            values = (
                df["compound_iso_smiles"]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )
            smiles.update([x for x in values if x and x.lower() != "nan"])

    return smiles, candidate_files


def collect_existing_features(dataset_roots):
    all_protein_ids = set()
    all_smiles = set()
    all_pt_files = []
    all_csv_files = []

    for root in dataset_roots:
        root = Path(root)

        pt_files = glob.glob(str(root / "processed" / "*surface_masif.pt"))
        pt_files += glob.glob(str(root / "**" / "processed" / "*surface_masif.pt"), recursive=True)

        for pt_path in sorted(set(pt_files)):
            all_pt_files.append(pt_path)
            protein_ids = load_pt_protein_ids(pt_path)
            all_protein_ids.update(protein_ids)

        smiles, csv_files = load_raw_smiles_from_dataset_root(root)
        all_smiles.update(smiles)
        all_csv_files.extend(csv_files)

    return all_protein_ids, all_smiles, all_pt_files, all_csv_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external_csv",
        type=str,
        default="external_validation/mmatt_s1/mmatt_s1_kinase_smcl_ready.csv",
        help="External validation CSV from Step 3"
    )
    parser.add_argument(
        "--dataset_roots",
        type=str,
        nargs="+",
        required=True,
        help="Existing dataset roots, e.g. data/kiba/cold data/davis/cold"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="external_validation/mmatt_s1/step5a_surface_reuse_summary.txt",
        help="Output summary path"
    )
    args = parser.parse_args()

    external_csv = Path(args.external_csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading external CSV: {external_csv}")
    ext = pd.read_csv(external_csv)

    if "target_id" not in ext.columns:
        raise ValueError("External CSV must contain target_id.")
    if "compound_iso_smiles" not in ext.columns:
        raise ValueError("External CSV must contain compound_iso_smiles.")

    ext["target_id"] = ext["target_id"].astype(str).str.strip()
    ext["compound_iso_smiles"] = ext["compound_iso_smiles"].astype(str).str.strip()

    ext_targets = set(ext["target_id"].dropna().tolist())
    ext_smiles = set(ext["compound_iso_smiles"].dropna().tolist())

    print("[INFO] Collecting existing surface/MaSIF feature identifiers...")
    existing_proteins, existing_smiles, pt_files, csv_files = collect_existing_features(args.dataset_roots)

    target_overlap = ext_targets & existing_proteins
    smiles_overlap = ext_smiles & existing_smiles

    ext_target_overlap_rows = ext[ext["target_id"].isin(target_overlap)]
    ext_smiles_overlap_rows = ext[ext["compound_iso_smiles"].isin(smiles_overlap)]
    ext_both_overlap_rows = ext[
        ext["target_id"].isin(target_overlap)
        & ext["compound_iso_smiles"].isin(smiles_overlap)
    ]

    summary = []
    summary.append("Step 5A surface/MaSIF reuse feasibility summary")
    summary.append("=" * 80)
    summary.append(f"External CSV: {external_csv}")
    summary.append(f"External rows: {len(ext)}")
    summary.append(f"External unique targets: {len(ext_targets)}")
    summary.append(f"External unique compounds: {len(ext_smiles)}")
    summary.append("")
    summary.append("Existing datasets searched")
    summary.append("-" * 80)
    for root in args.dataset_roots:
        summary.append(str(root))
    summary.append("")
    summary.append("Surface/MaSIF .pt files found")
    summary.append("-" * 80)
    for p in sorted(set(pt_files)):
        summary.append(p)
    summary.append("")
    summary.append("Raw CSV files found for SMILES overlap")
    summary.append("-" * 80)
    for p in sorted(set(csv_files)):
        summary.append(p)
    summary.append("")
    summary.append("Existing feature identifier counts")
    summary.append("-" * 80)
    summary.append(f"Existing unique protein IDs: {len(existing_proteins)}")
    summary.append(f"Existing unique SMILES: {len(existing_smiles)}")
    summary.append("")
    summary.append("Overlap with external MMAtt-DTA kinase subset")
    summary.append("-" * 80)
    summary.append(f"Target overlap unique: {len(target_overlap)} / {len(ext_targets)}")
    summary.append(f"Target overlap rows: {len(ext_target_overlap_rows)} / {len(ext)}")
    summary.append(f"Compound overlap unique: {len(smiles_overlap)} / {len(ext_smiles)}")
    summary.append(f"Compound overlap rows: {len(ext_smiles_overlap_rows)} / {len(ext)}")
    summary.append(f"Rows with both target and compound overlap: {len(ext_both_overlap_rows)} / {len(ext)}")
    summary.append("")
    summary.append("Overlapped targets")
    summary.append("-" * 80)
    summary.append(",".join(sorted(target_overlap)))
    summary.append("")
    summary.append("Interpretation")
    summary.append("-" * 80)
    if len(ext_both_overlap_rows) == len(ext):
        summary.append("All external samples can potentially reuse existing surface features.")
    elif len(ext_both_overlap_rows) > 0:
        summary.append("Only part of the external dataset can reuse existing surface features.")
        summary.append("New ligand/protein surface features are still required for full external validation.")
    else:
        summary.append("Existing surface features cannot cover the external dataset.")
        summary.append("New ligand and/or protein surface features must be generated.")

    out_path.write_text("\n".join(summary), encoding="utf-8")

    print("[DONE] Step 5A finished.")
    print(f"[OUT] Summary: {out_path}")


if __name__ == "__main__":
    main()