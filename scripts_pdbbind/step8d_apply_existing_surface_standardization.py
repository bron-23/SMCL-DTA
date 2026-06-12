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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--stats_json",
        default="/data_C/sdb1/lww/pdbbind/smcl_gign_exact_filtered_std/surface_standardization_stats.json",
    )
    parser.add_argument("--clamp_abs", type=float, default=10.0)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.stats_json, "r") as f:
        raw = json.load(f)

    stats = {}
    for key in SURFACE_KEYS:
        mean = torch.tensor(raw[key]["mean"], dtype=torch.float)
        std = torch.tensor(raw[key]["std"], dtype=torch.float).clamp_min(1e-6)
        stats[key] = (mean, std)
        print(key, "dim:", mean.numel())

    files = [
        "processed_data_train_surface_masif.pt",
        "processed_data_test1_surface_masif.pt",
        "processed_data_test2_surface_masif.pt",
    ]

    for fname in files:
        in_pt = in_dir / fname
        if not in_pt.exists():
            print("[SKIP missing]", in_pt)
            continue

        data_list = load_list(in_pt)
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

                v2 = torch.clamp(v2, -args.clamp_abs, args.clamp_abs)
                setattr(d, key, v2.float())

            out.append(d)

        out_pt = out_dir / fname
        save_list(out, out_pt)
        print("[DONE]", fname, "samples:", len(out), "out:", out_pt)

        d0 = out[0]
        for key in SURFACE_KEYS:
            v = getattr(d0, key)
            print(" ", key, tuple(v.shape), "finite:", bool(torch.isfinite(v).all()), "absmax:", float(v.abs().max()))


if __name__ == "__main__":
    main()
