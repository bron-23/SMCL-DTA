#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import json

import torch
from torch_geometric.data import InMemoryDataset


SURFACE_KEYS = ["protein_surface", "ligand_surface", "ligand_global"]


class DS(InMemoryDataset):
    def __init__(self, pt_path):
        super().__init__(".")
        self.data, self.slices = torch.load(pt_path, map_location="cpu", weights_only=False)


def load_list(pt):
    ds = DS(pt)
    return [ds.get(i) for i in range(len(ds))]


def save_list(data_list, out_pt):
    collater = InMemoryDataset()
    data, slices = collater.collate(data_list)
    out_pt = Path(out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save((data, slices), out_pt)


def collect_train_stats(data_list, key):
    vals = []
    for d in data_list:
        v = getattr(d, key)
        v = torch.nan_to_num(v.float(), nan=0.0, posinf=0.0, neginf=0.0)

        if v.dim() == 1:
            vals.append(v.view(1, -1))
        else:
            vals.append(v.reshape(-1, v.shape[-1]))

    x = torch.cat(vals, dim=0)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    return mean, std


def transform_list(data_list, stats, clamp_abs=10.0):
    out = []

    for d in data_list:
        for key in SURFACE_KEYS:
            v = getattr(d, key)
            v = torch.nan_to_num(v.float(), nan=0.0, posinf=0.0, neginf=0.0)

            mean, std = stats[key]

            if v.dim() == 1:
                v2 = (v - mean) / std
            else:
                v2 = (v - mean.view(1, -1)) / std.view(1, -1)

            v2 = torch.clamp(v2, -clamp_abs, clamp_abs)
            setattr(d, key, v2.float())

        out.append(d)

    return out


def check_pt(pt_path):
    ds = DS(pt_path)
    print("check:", pt_path, "samples:", len(ds))
    d = ds.get(0)
    for key in SURFACE_KEYS:
        v = getattr(d, key)
        print(" ", key, tuple(v.shape), "finite:", bool(torch.isfinite(v).all()), "absmax:", float(v.abs().max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--clamp_abs", type=float, default=10.0)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "train": "processed_data_train_surface_masif.pt",
        "val": "processed_data_test1_surface_masif.pt",
        "core2016": "processed_data_test2_surface_masif.pt",
    }

    for split, fname in files.items():
        p = in_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"missing input file for {split}: {p}")

    print("=" * 100)
    print("[INFO] loading input datasets")
    data = {split: load_list(in_dir / fname) for split, fname in files.items()}
    for split in files:
        print(split, len(data[split]))

    print("=" * 100)
    print("[INFO] collecting train statistics")
    stats = {}
    stat_json = {}

    for key in SURFACE_KEYS:
        mean, std = collect_train_stats(data["train"], key)
        stats[key] = (mean, std)
        stat_json[key] = {"mean": mean.tolist(), "std": std.tolist()}

        print("-" * 100)
        print(key)
        print("dim:", mean.numel())
        print("mean min/max:", float(mean.min()), float(mean.max()))
        print("std min/max:", float(std.min()), float(std.max()))

    print("=" * 100)
    print("[INFO] transforming and saving")
    for split, fname in files.items():
        transformed = transform_list(data[split], stats, clamp_abs=args.clamp_abs)
        out_pt = out_dir / fname
        save_list(transformed, out_pt)
        print("[DONE]", split, "samples:", len(transformed), "out:", out_pt)

    stats_path = out_dir.parent / "surface_standardization_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stat_json, f)

    print("=" * 100)
    print("[INFO] saved stats:", stats_path)
    for split, fname in files.items():
        check_pt(out_dir / fname)

    print("=" * 100)
    print("[DONE] standardization completed successfully")


if __name__ == "__main__":
    main()
