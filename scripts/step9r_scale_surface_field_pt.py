#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--output_pt", required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--field", default="protein_surface")
    args = parser.parse_args()

    ds = LoadedPTDataset(args.input_pt)
    data_list = []
    changed = 0

    for i in range(len(ds)):
        d = ds.get(i)
        if hasattr(d, args.field) and getattr(d, args.field) is not None:
            x = getattr(d, args.field)
            setattr(d, args.field, x * args.scale)
            changed += 1
        data_list.append(d)

    data, slices = InMemoryDataset.collate(data_list)
    out_path = Path(args.output_pt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((data, slices), out_path)

    print("input_pt:", args.input_pt)
    print("output_pt:", args.output_pt)
    print("field:", args.field)
    print("scale:", args.scale)
    print("samples:", len(ds))
    print("changed:", changed)


if __name__ == "__main__":
    main()
