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


def sanitize_tensor(x):
    if not torch.is_tensor(x):
        return x, 0

    bad_mask = ~torch.isfinite(x)
    bad_count = int(bad_mask.sum().item())

    if bad_count == 0:
        return x, 0

    y = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return y, bad_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--output_pt", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    ds = LoadedPTDataset(args.input_pt)

    data_list = []
    total_bad_by_key = {}
    sample_bad_count = 0

    for i in range(len(ds)):
        d = ds.get(i)
        sample_had_bad = False

        for key in list(d.keys()):
            value = d[key]
            if torch.is_tensor(value):
                fixed, bad_count = sanitize_tensor(value)
                if bad_count > 0:
                    d[key] = fixed
                    total_bad_by_key[key] = total_bad_by_key.get(key, 0) + bad_count
                    sample_had_bad = True

        if sample_had_bad:
            sample_bad_count += 1

        data_list.append(d)

    data, slices = InMemoryDataset.collate(data_list)

    out_path = Path(args.output_pt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((data, slices), out_path)

    lines = []
    lines.append("Step 7G sanitize surface pt summary")
    lines.append("=" * 100)
    lines.append(f"Input pt: {args.input_pt}")
    lines.append(f"Output pt: {args.output_pt}")
    lines.append(f"Samples: {len(ds)}")
    lines.append(f"Samples with nonfinite tensors: {sample_bad_count}")
    lines.append("")
    lines.append("Nonfinite value counts by tensor key")
    lines.append("-" * 100)
    if total_bad_by_key:
        for k, v in sorted(total_bad_by_key.items()):
            lines.append(f"{k}: {v}")
    else:
        lines.append("None")
    lines.append("")
    lines.append("[DONE]")

    Path(args.summary).write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
