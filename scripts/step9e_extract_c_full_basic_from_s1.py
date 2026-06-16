#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset


class LoadedPTDataset(InMemoryDataset):
    def __init__(self, pt_path):
        super().__init__(".")
        self.data, self.slices = torch.load(pt_path, map_location="cpu", weights_only=False)

    def len(self):
        return int(self.slices["y"].numel() - 1)


def save_data_list(data_list, out_path):
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), out_path)


def add_occurrence_index(df, key_cols):
    df = df.copy()
    df["_occ"] = df.groupby(key_cols, dropna=False).cumcount()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full_s1_csv",
        default="/home/lww/learn_project/mydta/external_validation/mmatt_s1/mmatt_s1_kinase_smcl_ready.csv",
    )
    parser.add_argument(
        "--full_s1_basic_pt",
        default="/home/lww/learn_project/mydta/external_validation/mmatt_s1/smcl_processed_basic/processed_data_mmatt_s1_kinase_basic.pt",
    )
    parser.add_argument(
        "--c_smcl_raw",
        default="/data_C/sdb1/lww/mmatt_independent_tests_raw_aligned/step9_c_full_surface/C_new_compound_new_target_full_607_smcl_raw.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="/data_C/sdb1/lww/mmatt_independent_tests_raw_aligned/step9_c_full_surface/c_full_basic",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(args.full_s1_csv)
    c = pd.read_csv(args.c_smcl_raw)

    print("[INFO] full S1 CSV rows:", len(full))
    print("[INFO] C raw rows:", len(c))

    full_ds = LoadedPTDataset(args.full_s1_basic_pt)
    print("[INFO] full S1 basic pt samples:", len(full_ds))

    if len(full) != len(full_ds):
        raise ValueError(f"full CSV rows {len(full)} != full basic pt samples {len(full_ds)}")

    full = full.copy()
    full["_full_index"] = range(len(full))
    c = c.copy()
    c["_c_order"] = range(len(c))

    full["affinity_key"] = pd.to_numeric(full["affinity"], errors="coerce").round(6)
    c["affinity_key"] = pd.to_numeric(c["affinity"], errors="coerce").round(6)

    full["smiles_key"] = full["compound_iso_smiles"].astype(str)
    c["smiles_key"] = c["compound_iso_smiles"].astype(str)

    full["target_key"] = full["target_id"].astype(str)
    c["target_key"] = c["target_id"].astype(str)

    key_cols = ["smiles_key", "target_key", "affinity_key"]

    full2 = add_occurrence_index(full, key_cols)
    c2 = add_occurrence_index(c, key_cols)

    merged = c2.merge(
        full2[
            key_cols
            + ["_occ", "_full_index", "target_sequence"]
        ],
        on=key_cols + ["_occ"],
        how="left",
    )

    matched = merged["_full_index"].notna().sum()
    print("[MATCH] C rows matched to full S1:", matched, "/", len(c))

    if matched != len(c):
        bad = merged[merged["_full_index"].isna()].copy()
        bad.to_csv(out_dir / "unmatched_c_rows.csv", index=False)
        print("[OUT] unmatched:", out_dir / "unmatched_c_rows.csv")
        raise RuntimeError("Some C rows could not be matched to full S1 CSV/basic pt.")

    merged = merged.sort_values("_c_order").reset_index(drop=True)
    full_indices = merged["_full_index"].astype(int).tolist()

    c_with_seq = pd.DataFrame({
        "compound_iso_smiles": merged["compound_iso_smiles"],
        "target_id": merged["target_id"],
        "affinity": merged["affinity"],
        "protein_class": merged["protein_class"],
        "target_sequence": merged["target_sequence"],
        "_full_s1_index": full_indices,
    })

    out_csv = out_dir / "C_new_compound_new_target_full_607_with_sequence.csv"
    c_with_seq.to_csv(out_csv, index=False)

    data_list = [full_ds.get(i) for i in full_indices]
    out_pt = out_dir / "processed_data_C_new_compound_new_target_full_607_basic.pt"
    save_data_list(data_list, out_pt)

    summary = []
    summary.append("Step 9E C full basic extraction summary")
    summary.append("=" * 100)
    summary.append(f"Full S1 CSV: {args.full_s1_csv}")
    summary.append(f"Full S1 basic pt: {args.full_s1_basic_pt}")
    summary.append(f"C raw: {args.c_smcl_raw}")
    summary.append(f"Full S1 rows: {len(full)}")
    summary.append(f"C rows: {len(c)}")
    summary.append(f"Matched C rows: {matched}/{len(c)}")
    summary.append("")
    summary.append("Output files")
    summary.append("-" * 100)
    summary.append(str(out_csv))
    summary.append(str(out_pt))

    out_summary = out_dir / "step9e_c_full_basic_extraction_summary.txt"
    out_summary.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))
    print("[DONE]")


if __name__ == "__main__":
    main()