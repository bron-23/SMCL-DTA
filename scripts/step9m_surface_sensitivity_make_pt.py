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
        return int(self.slices["y"].numel() - 1)


def save_data_list(data_list, out_path):
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--out_zero_pt", required=True)
    parser.add_argument("--out_scale10_pt", required=True)
    args = parser.parse_args()

    ds = LoadedPTDataset(args.input_pt)

    zero_list = []
    scale_list = []

    changed = 0

    for i in range(len(ds)):
        d0 = ds.get(i)
        d1 = ds.get(i)

        if hasattr(d0, "protein_surface") and d0.protein_surface is not None:
            d0.protein_surface = torch.zeros_like(d0.protein_surface)
            d1.protein_surface = d1.protein_surface * 10.0
            changed += 1

        zero_list.append(d0)
        scale_list.append(d1)

    Path(args.out_zero_pt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_scale10_pt).parent.mkdir(parents=True, exist_ok=True)

    save_data_list(zero_list, args.out_zero_pt)
    save_data_list(scale_list, args.out_scale10_pt)

    print("Input samples:", len(ds))
    print("Samples with protein_surface changed:", changed)
    print("Zero pt:", args.out_zero_pt)
    print("Scale10 pt:", args.out_scale10_pt)


if __name__ == "__main__":
    main()