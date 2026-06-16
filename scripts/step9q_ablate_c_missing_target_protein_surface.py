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
        if "y" in self.slices:
            return int(self.slices["y"].numel() - 1)
        key = list(self.slices.keys())[0]
        return int(self.slices[key].numel() - 1)


def save_data_list(data_list, out_path):
    data, slices = InMemoryDataset.collate(data_list)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save((data, slices), out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--missing_targets_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ds = LoadedPTDataset(args.input_pt)
    rows = pd.read_csv(args.rows_csv)

    missing_df = pd.read_csv(args.missing_targets_csv)
    if "uniprot_id" in missing_df.columns:
        missing_targets = set(missing_df["uniprot_id"].astype(str))
    else:
        missing_targets = set(missing_df.iloc[:, 0].astype(str))

    if len(rows) != len(ds):
        raise RuntimeError(f"rows_csv length {len(rows)} != pt length {len(ds)}")

    target_col = "target_id" if "target_id" in rows.columns else "uniprot_id"
    if target_col not in rows.columns:
        raise RuntimeError("rows_csv must contain target_id or uniprot_id")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = ["zero", "noise", "scale10"]
    data_lists = {m: [] for m in modes}
    changed_counts = {m: 0 for m in modes}

    for i in range(len(ds)):
        target_id = str(rows.iloc[i][target_col])
        is_missing_target = target_id in missing_targets

        for mode in modes:
            d = ds.get(i)

            if is_missing_target and hasattr(d, "protein_surface") and d.protein_surface is not None:
                x = d.protein_surface
                if mode == "zero":
                    d.protein_surface = torch.zeros_like(x)
                elif mode == "noise":
                    d.protein_surface = torch.randn_like(x)
                elif mode == "scale10":
                    d.protein_surface = x * 10.0
                changed_counts[mode] += 1

            data_lists[mode].append(d)

    out_paths = {}
    for mode in modes:
        out_pt = out_dir / f"processed_data_C_full_missing_target_protein_surface_{mode}.pt"
        save_data_list(data_lists[mode], out_pt)
        out_paths[mode] = out_pt

    lines = []
    lines.append("Step 9Q C missing-target protein_surface ablation summary")
    lines.append("=" * 100)
    lines.append(f"Input pt: {args.input_pt}")
    lines.append(f"Rows csv: {args.rows_csv}")
    lines.append(f"Samples: {len(ds)}")
    lines.append(f"Missing targets: {len(missing_targets)}")
    lines.append("")
    for mode in modes:
        lines.append(f"{mode}: changed samples = {changed_counts[mode]}")
        lines.append(f"{mode}: out_pt = {out_paths[mode]}")
    lines.append("")
    lines.append("[DONE]")

    summary_path = out_dir / "step9q_ablation_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
