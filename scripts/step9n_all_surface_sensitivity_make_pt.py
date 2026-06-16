#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset


class LoadedPTDataset(InMemoryDataset):
    def __init__(self, pt_path):
        super().__init__(".")
        try:
            self.data, self.slices = torch.load(pt_path, map_location="cpu", weights_only=False)
        except TypeError:
            self.data, self.slices = torch.load(pt_path, map_location="cpu")

    def len(self):
        if "y" in self.slices:
            return int(self.slices["y"].numel() - 1)
        key = list(self.slices.keys())[0]
        return int(self.slices[key].numel() - 1)


def save_data_list(data_list, out_path):
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), out_path)


def tensor_keys(data):
    keys = list(data.keys()) if callable(data.keys) else list(data.keys)
    return [k for k in keys if torch.is_tensor(data[k])]


def modify_surface_fields(data, mode):
    """
    Modify protein_surface, ligand_surface, ligand_global if they exist.
    """
    changed = []

    surface_keys = [
        "protein_surface",
        "ligand_surface",
        "ligand_global",
        "protein_surface_x",
        "ligand_surface_x",
        "ligand_global_x",
    ]

    keys = tensor_keys(data)

    for key in surface_keys:
        if key not in keys:
            continue

        x = data[key]

        if mode == "zero":
            data[key] = torch.zeros_like(x)
        elif mode == "scale10":
            data[key] = x * 10.0
        elif mode == "noise":
            # Same shape, roughly standard normal. Fixed generator handled outside by torch manual seed.
            data[key] = torch.randn_like(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        changed.append((key, tuple(x.shape)))

    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = LoadedPTDataset(args.input_pt)

    modes = ["zero", "scale10", "noise"]
    data_lists = {m: [] for m in modes}
    changed_counts = {m: 0 for m in modes}
    changed_key_counts = {m: {} for m in modes}

    print("=" * 100)
    print("[INFO] Input pt:", args.input_pt)
    print("[INFO] Samples:", len(ds))

    first = ds.get(0)
    print("[INFO] First sample tensor keys and shapes:")
    for k in tensor_keys(first):
        print(" ", k, tuple(first[k].shape))

    for i in range(len(ds)):
        for mode in modes:
            d = ds.get(i)
            changed = modify_surface_fields(d, mode)
            if changed:
                changed_counts[mode] += 1
                for key, shape in changed:
                    changed_key_counts[mode][key] = changed_key_counts[mode].get(key, 0) + 1
            data_lists[mode].append(d)

    out_paths = {}
    for mode in modes:
        out_pt = out_dir / f"processed_data_C_full_all_surface_{mode}.pt"
        save_data_list(data_lists[mode], out_pt)
        out_paths[mode] = out_pt

    print("=" * 100)
    print("[RESULT]")
    for mode in modes:
        print(f"[{mode}] samples_changed:", changed_counts[mode])
        print(f"[{mode}] key_counts:", changed_key_counts[mode])
        print(f"[{mode}] out_pt:", out_paths[mode])

    summary = []
    summary.append("Step 9N all-surface sensitivity pt generation summary")
    summary.append("=" * 100)
    summary.append(f"Input pt: {args.input_pt}")
    summary.append(f"Samples: {len(ds)}")
    summary.append("")
    for mode in modes:
        summary.append(f"{mode}")
        summary.append("-" * 100)
        summary.append(f"samples_changed: {changed_counts[mode]}")
        summary.append(f"key_counts: {changed_key_counts[mode]}")
        summary.append(f"out_pt: {out_paths[mode]}")
        summary.append("")

    summary_path = out_dir / "step9n_all_surface_sensitivity_make_pt_summary.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    print("[OUT]", summary_path)
    print("[DONE]")


if __name__ == "__main__":
    main()
