#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add protein sequences to PDBbind v2016 manifest.

Input:
  /data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_valid_resplit_manifest.csv

Output:
  pdbbind2016_valid_resplit_manifest_with_seq.csv
  pdbbind2016_valid_resplit_manifest_with_seq_valid.csv
  pdbbind2016_sequence_failures.csv

Notes:
  - Sequences are extracted from *_protein.pdb.
  - Multiple chains are concatenated with no separator.
  - Common modified residues are mapped to canonical amino acids when possible.
"""

import argparse
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser


AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",

    # Common modified residues mapped to canonical residues.
    "MSE": "M",  # Selenomethionine
    "SEC": "C",
    "PYL": "K",
    "HYP": "P",
    "TPO": "T",
    "SEP": "S",
    "PTR": "Y",
    "CSO": "C",
    "CME": "C",
    "KCX": "K",
    "MLY": "K",
    "M3L": "K",
    "HIC": "H",
    "HIP": "H",
    "HID": "H",
    "HIE": "H",
    "ASH": "D",
    "GLH": "E",
    "CYX": "C",
    "CYM": "C",
}


def extract_sequence_from_pdb(pdb_path: str):
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        return None, "missing_pdb"

    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    except Exception as exc:
        return None, f"parse_error:{type(exc).__name__}"

    chain_seqs = []

    try:
        model = next(structure.get_models())
    except StopIteration:
        return None, "no_model"

    for chain in model:
        residues = []
        seen_res_ids = set()

        for res in chain:
            hetflag, resseq, icode = res.id

            # Keep standard residues and common modified amino acids.
            resname = res.resname.strip().upper()

            # Avoid duplicated residue IDs within a chain.
            rid = (chain.id, resseq, icode, resname)
            if rid in seen_res_ids:
                continue
            seen_res_ids.add(rid)

            if resname in AA3_TO_AA1:
                residues.append(AA3_TO_AA1[resname])
            else:
                # Skip waters, ions, cofactors, and unknown hetero residues.
                continue

        if residues:
            chain_seqs.append("".join(residues))

    if not chain_seqs:
        return None, "empty_sequence"

    seq = "".join(chain_seqs)

    # Basic sanity check.
    if len(seq) < 20:
        return None, f"too_short:{len(seq)}"

    return seq, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="/data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_valid_resplit_manifest.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="/data_C/sdb1/lww/pdbbind/manifests",
    )
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest)

    seqs = []
    status = []
    lengths = []

    for i, row in df.iterrows():
        seq, st = extract_sequence_from_pdb(row["protein_pdb"])
        seqs.append(seq)
        status.append(st)
        lengths.append(len(seq) if seq is not None else None)

        if (i + 1) % 1000 == 0:
            print(f"[INFO] processed {i + 1}/{len(df)}")

    df["protein_sequence"] = seqs
    df["protein_seq_status"] = status
    df["protein_seq_len"] = lengths
    df["protein_seq_ok"] = df["protein_sequence"].notna()

    out_all = out_dir / "pdbbind2016_valid_resplit_manifest_with_seq.csv"
    out_valid = out_dir / "pdbbind2016_valid_resplit_manifest_with_seq_valid.csv"
    out_fail = out_dir / "pdbbind2016_sequence_failures.csv"

    df.to_csv(out_all, index=False)
    df[df["protein_seq_ok"]].to_csv(out_valid, index=False)
    df[~df["protein_seq_ok"]].to_csv(out_fail, index=False)

    print("=" * 100)
    print("[SUMMARY]")
    print("total:", len(df))
    print("sequence valid:", int(df["protein_seq_ok"].sum()))
    print("sequence failed:", int((~df["protein_seq_ok"]).sum()))
    print()
    print("[VALID BY SPLIT]")
    print(df[df["protein_seq_ok"]]["split"].value_counts())
    print()
    print("[FAILED BY SPLIT]")
    print(df[~df["protein_seq_ok"]]["split"].value_counts())
    print()
    print("[STATUS COUNTS]")
    print(df["protein_seq_status"].value_counts(dropna=False).head(20))
    print()
    print("[SEQUENCE LENGTH STATS]")
    print(df[df["protein_seq_ok"]].groupby("split")["protein_seq_len"].describe())
    print()
    print("[OUT]", out_all)
    print("[OUT]", out_valid)
    print("[OUT]", out_fail)


if __name__ == "__main__":
    main()
