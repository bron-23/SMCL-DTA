#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 7C: Add UniProt protein sequences to MMAtt-DTA kinase SMCL raw CSVs.

Inputs:
    mmatt_kinase_train_raw.csv
    mmatt_kinase_val_raw.csv

Outputs:
    mmatt_kinase_train_with_sequence.csv
    mmatt_kinase_val_with_sequence.csv
    mmatt_kinase_missing_sequence.csv
    mmatt_kinase_sequence_cache.csv
    step7c_sequence_summary.txt

Strategy:
    1. Load any existing local sequence cache if provided.
    2. Reuse cached target_sequence where available.
    3. Download missing UniProt FASTA sequences if --allow_download is set.
"""

import argparse
import time
import urllib.request
from pathlib import Path

import pandas as pd


def read_cache_file(path):
    path = Path(path)
    if not path.exists():
        return {}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    cols = {c.lower(): c for c in df.columns}

    id_col = None
    seq_col = None

    for cand in ["target_id", "uniprot_id", "protein_id", "accession"]:
        if cand in cols:
            id_col = cols[cand]
            break

    for cand in ["target_sequence", "sequence", "protein_sequence"]:
        if cand in cols:
            seq_col = cols[cand]
            break

    if id_col is None or seq_col is None:
        return {}

    cache = {}
    for _, row in df[[id_col, seq_col]].dropna().iterrows():
        tid = str(row[id_col]).strip()
        seq = str(row[seq_col]).strip()
        if tid and seq and seq.lower() != "nan":
            cache[tid] = seq

    return cache


def fetch_uniprot_fasta(uniprot_id, sleep_sec=0.1):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            text = response.read().decode("utf-8")
    except Exception as e:
        print(f"[WARN] Failed to fetch {uniprot_id}: {e}")
        return None

    if not text.startswith(">"):
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith(">")]
    seq = "".join(lines).strip()

    time.sleep(sleep_sec)

    if seq:
        return seq

    return None


def add_sequences(df, seq_cache):
    out = df.copy()
    out["target_sequence"] = out["target_id"].map(seq_cache)
    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=str,
        default="/data_C/sdb1/lww/mmatt_training/smcl_raw/mmatt_kinase_train_raw.csv",
    )
    parser.add_argument(
        "--val",
        type=str,
        default="/data_C/sdb1/lww/mmatt_training/smcl_raw/mmatt_kinase_val_raw.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data_C/sdb1/lww/mmatt_training/smcl_sequence",
    )
    parser.add_argument(
        "--cache_files",
        type=str,
        nargs="*",
        default=[
            "/home/lww/learn_project/mydta/external_validation/mmatt_s1/mmatt_s1_targets.csv",
            "/home/lww/learn_project/mydta/external_validation/mmatt_s1/mmatt_s1_kinase_with_sequence.csv",
            "/data_C/sdb1/lww/mmatt_s1_surface_overlap/mmatt_s1_kinase_surface_masif_overlap_rows.csv",
        ],
    )
    parser.add_argument(
        "--allow_download",
        action="store_true",
        help="Download missing sequences from UniProt REST API.",
    )
    parser.add_argument(
        "--sleep_sec",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)

    required = ["compound_iso_smiles", "target_id", "affinity"]
    for name, df in [("train", train_df), ("val", val_df)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} file missing required columns: {missing}")

    all_targets = sorted(set(train_df["target_id"].astype(str)) | set(val_df["target_id"].astype(str)))

    seq_cache = {}

    print("[INFO] Loading local cache files...")
    for cf in args.cache_files:
        loaded = read_cache_file(cf)
        if loaded:
            print(f"[INFO] Loaded {len(loaded)} sequences from {cf}")
            seq_cache.update(loaded)
        else:
            print(f"[INFO] No usable sequence cache from {cf}")

    before_download = sum(t in seq_cache for t in all_targets)
    missing_targets = [t for t in all_targets if t not in seq_cache]

    print(f"[INFO] Unique targets: {len(all_targets)}")
    print(f"[INFO] Cached before download: {before_download}")
    print(f"[INFO] Missing before download: {len(missing_targets)}")

    if args.allow_download and missing_targets:
        print("[INFO] Downloading missing UniProt sequences...")
        for idx, tid in enumerate(missing_targets, start=1):
            seq = fetch_uniprot_fasta(tid, sleep_sec=args.sleep_sec)
            if seq:
                seq_cache[tid] = seq

            if idx % 20 == 0:
                print(f"[INFO] Download progress: {idx}/{len(missing_targets)}")

    train_out = add_sequences(train_df, seq_cache)
    val_out = add_sequences(val_df, seq_cache)

    train_missing = train_out[train_out["target_sequence"].isna()].copy()
    val_missing = val_out[val_out["target_sequence"].isna()].copy()

    train_with_seq = train_out.dropna(subset=["target_sequence"]).copy()
    val_with_seq = val_out.dropna(subset=["target_sequence"]).copy()

    train_out_path = out_dir / "mmatt_kinase_train_with_sequence.csv"
    val_out_path = out_dir / "mmatt_kinase_val_with_sequence.csv"
    missing_path = out_dir / "mmatt_kinase_missing_sequence.csv"
    cache_path = out_dir / "mmatt_kinase_sequence_cache.csv"
    summary_path = out_dir / "step7c_sequence_summary.txt"

    train_with_seq.to_csv(train_out_path, index=False)
    val_with_seq.to_csv(val_out_path, index=False)

    missing_df = pd.concat(
        [
            train_missing.assign(split="train"),
            val_missing.assign(split="val"),
        ],
        ignore_index=True,
    )
    missing_df.to_csv(missing_path, index=False)

    cache_df = pd.DataFrame(
        [{"target_id": k, "target_sequence": v} for k, v in sorted(seq_cache.items())]
    )
    cache_df.to_csv(cache_path, index=False)

    final_cached = sum(t in seq_cache for t in all_targets)
    final_missing_targets = [t for t in all_targets if t not in seq_cache]

    summary = []
    summary.append("Step 7C MMAtt-DTA kinase sequence mapping summary")
    summary.append("=" * 80)
    summary.append(f"Train input: {args.train}")
    summary.append(f"Validation input: {args.val}")
    summary.append("")
    summary.append(f"Original train rows: {len(train_df)}")
    summary.append(f"Original validation rows: {len(val_df)}")
    summary.append(f"Unique targets total: {len(all_targets)}")
    summary.append("")
    summary.append(f"Cached targets before download: {before_download}")
    summary.append(f"Missing targets before download: {len(missing_targets)}")
    summary.append(f"Cached targets after download: {final_cached}")
    summary.append(f"Missing targets after download: {len(final_missing_targets)}")
    summary.append("")
    summary.append(f"Train rows with sequence: {len(train_with_seq)}")
    summary.append(f"Train rows missing sequence: {len(train_missing)}")
    summary.append(f"Validation rows with sequence: {len(val_with_seq)}")
    summary.append(f"Validation rows missing sequence: {len(val_missing)}")
    summary.append("")
    summary.append("Sequence length statistics")
    summary.append("-" * 80)
    if len(train_with_seq) > 0:
        seq_lens = pd.concat(
            [
                train_with_seq["target_sequence"].str.len(),
                val_with_seq["target_sequence"].str.len(),
            ],
            ignore_index=True,
        )
        summary.append(str(seq_lens.describe()))
    else:
        summary.append("No sequence available.")
    summary.append("")
    summary.append("Missing target IDs")
    summary.append("-" * 80)
    summary.append(",".join(final_missing_targets))
    summary.append("")
    summary.append("Output files")
    summary.append("-" * 80)
    summary.append(str(train_out_path))
    summary.append(str(val_out_path))
    summary.append(str(missing_path))
    summary.append(str(cache_path))

    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))


if __name__ == "__main__":
    main()