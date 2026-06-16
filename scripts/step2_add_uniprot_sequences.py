#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Add UniProt protein sequences to the MMAtt-DTA S1 kinase subset.

Input:
    external_validation/mmatt_s1/mmatt_s1_kinase.csv

Outputs:
    external_validation/mmatt_s1/mmatt_s1_kinase_with_sequence.csv
    external_validation/mmatt_s1/mmatt_s1_kinase_missing_sequence.csv
    external_validation/mmatt_s1/mmatt_s1_uniprot_sequence_cache.csv
    external_validation/mmatt_s1/mmatt_s1_sequence_summary.txt
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests


def parse_fasta_sequence(fasta_text: str) -> str:
    lines = fasta_text.strip().splitlines()
    seq_lines = [line.strip() for line in lines if line and not line.startswith(">")]
    return "".join(seq_lines)


def fetch_uniprot_sequence(uniprot_id: str, timeout: int = 20) -> str:
    """
    Fetch FASTA sequence from UniProt REST endpoint.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url, timeout=timeout)

    if response.status_code != 200:
        return ""

    sequence = parse_fasta_sequence(response.text)
    return sequence


def load_existing_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}

    cache_df = pd.read_csv(cache_path)
    if "target_id" not in cache_df.columns or "target_sequence" not in cache_df.columns:
        return {}

    cache = {}
    for _, row in cache_df.iterrows():
        tid = str(row["target_id"]).strip()
        seq = str(row["target_sequence"]).strip()
        if tid and seq and seq.lower() != "nan":
            cache[tid] = seq

    return cache


def save_cache(cache: dict, cache_path: Path):
    cache_df = pd.DataFrame(
        [{"target_id": k, "target_sequence": v} for k, v in sorted(cache.items())]
    )
    cache_df.to_csv(cache_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="external_validation/mmatt_s1/mmatt_s1_kinase.csv",
        help="Input kinase subset CSV"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="external_validation/mmatt_s1",
        help="Output directory"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep time between UniProt requests"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "mmatt_s1_uniprot_sequence_cache.csv"

    print(f"[INFO] Loading kinase subset: {input_path}")
    df = pd.read_csv(input_path)

    if "target_id" not in df.columns:
        raise ValueError("Input CSV must contain 'target_id' column.")

    unique_targets = sorted(df["target_id"].dropna().astype(str).str.strip().unique())
    print(f"[INFO] Unique targets to process: {len(unique_targets)}")

    cache = load_existing_cache(cache_path)
    print(f"[INFO] Loaded cached sequences: {len(cache)}")

    missing_targets = [tid for tid in unique_targets if tid not in cache]
    print(f"[INFO] Targets missing in cache: {len(missing_targets)}")

    failed = []

    for idx, tid in enumerate(missing_targets, start=1):
        print(f"[FETCH] {idx}/{len(missing_targets)} {tid}")

        seq = ""
        try:
            seq = fetch_uniprot_sequence(tid)
        except Exception as e:
            print(f"[WARN] Failed to fetch {tid}: {e}")

        if seq:
            cache[tid] = seq
        else:
            failed.append(tid)

        # Save cache every 20 targets
        if idx % 20 == 0:
            save_cache(cache, cache_path)

        time.sleep(args.sleep)

    save_cache(cache, cache_path)

    print(f"[INFO] Final cached sequences: {len(cache)}")
    print(f"[INFO] Failed targets: {len(failed)}")

    # Add sequence column
    df["target_id"] = df["target_id"].astype(str).str.strip()
    df["target_sequence"] = df["target_id"].map(cache)

    with_seq = df.dropna(subset=["target_sequence"]).copy()
    with_seq = with_seq[
        (with_seq["target_sequence"] != "") &
        (with_seq["target_sequence"].str.lower() != "nan")
    ].copy()

    missing_seq = df[
        df["target_sequence"].isna() |
        (df["target_sequence"] == "") |
        (df["target_sequence"].astype(str).str.lower() == "nan")
    ].copy()

    # Reorder columns for SMCL-DTA preprocessing
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

    existing_cols = [c for c in preferred_cols if c in with_seq.columns]
    remaining_cols = [c for c in with_seq.columns if c not in existing_cols]
    with_seq = with_seq[existing_cols + remaining_cols]

    out_with_seq = out_dir / "mmatt_s1_kinase_with_sequence.csv"
    out_missing = out_dir / "mmatt_s1_kinase_missing_sequence.csv"

    with_seq.to_csv(out_with_seq, index=False)
    missing_seq.to_csv(out_missing, index=False)

    summary_lines = []
    summary_lines.append("MMAtt-DTA S1 sequence mapping summary")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Input file: {input_path}")
    summary_lines.append(f"Original kinase rows: {len(df)}")
    summary_lines.append(f"Unique kinase targets: {len(unique_targets)}")
    summary_lines.append(f"Cached target sequences: {len(cache)}")
    summary_lines.append(f"Rows with sequence: {len(with_seq)}")
    summary_lines.append(f"Rows missing sequence: {len(missing_seq)}")
    summary_lines.append("")
    summary_lines.append(f"Unique compounds with sequence: {with_seq['compound_iso_smiles'].nunique()}")
    summary_lines.append(f"Unique targets with sequence: {with_seq['target_id'].nunique()}")
    summary_lines.append("")
    summary_lines.append("Sequence length statistics")
    summary_lines.append("-" * 60)
    if len(with_seq) > 0:
        seq_lengths = with_seq["target_sequence"].astype(str).str.len()
        summary_lines.append(str(seq_lengths.describe()))
    else:
        summary_lines.append("No valid sequences found.")
    summary_lines.append("")
    summary_lines.append("Output files")
    summary_lines.append("-" * 60)
    summary_lines.append(str(out_with_seq))
    summary_lines.append(str(out_missing))
    summary_lines.append(str(cache_path))

    summary_path = out_dir / "mmatt_s1_sequence_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("[DONE] Step 2 finished.")
    print(f"[OUT] Kinase with sequence: {out_with_seq}")
    print(f"[OUT] Missing sequence rows: {out_missing}")
    print(f"[OUT] Sequence cache: {cache_path}")
    print(f"[OUT] Summary: {summary_path}")


if __name__ == "__main__":
    main()