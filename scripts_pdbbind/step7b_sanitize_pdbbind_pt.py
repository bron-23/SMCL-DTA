#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

import torch
from torch_geometric.data import InMemoryDataset


FLOAT_KEYS = [
    "x",
    "edge_attr",
    "y",
    "ligand_surface",
    "ligand_global",
    "protein_surface",
]


class DS(InMemoryDataset):
    def __init__(self, pt_path):
        super().__init__(".")
        self.data, self.slices = torch.load(pt_path, map_location="cpu", weights_only=False)


def sanitize_tensor(x, clamp_abs=None):
    if not torch.is_tensor(x):
        return x
    if not x.is_floating_point():
        return x

    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if clamp_abs is not None and clamp_abs > 0:
        x = torch.clamp(x, min=-clamp_abs, max=clamp_abs)

    return x


def sanitize_pt(in_pt, out_pt, clamp_abs=None):
    ds = DS(in_pt)
    data_list = []

    bad_before = []
    bad_after = []

    for i in range(len(ds)):
        d = ds.get(i)

        for k in FLOAT_KEYS:
            if hasattr(d, k):
                v = getattr(d, k)
                if torch.is_tensor(v) and v.is_floating_point():
                    if not torch.isfinite(v).all():
                        bad_before.append((i, k))

                    v2 = sanitize_tensor(v, clamp_abs=clamp_abs)
                    setattr(d, k, v2)

                    if not torch.isfinite(v2).all():
                        bad_after.append((i, k))

        data_list.append(d)

    collater = InMemoryDataset()
    data, slices = collater.collate(data_list)

    out_pt = Path(out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save((data, slices), out_pt)

    print("=" * 100)
    print("in_pt:", in_pt)
    print("out_pt:", out_pt)
    print("samples:", len(data_list))
    print("bad_before:", bad_before[:20], "count:", len(bad_before))
    print("bad_after:", bad_after[:20], "count:", len(bad_after))

    # quick stats
    ds2 = DS(out_pt)
    for k in FLOAT_KEYS:
        vals = []
        for i in range(len(ds2)):
            d = ds2.get(i)
            if hasattr(d, k):
                v = getattr(d, k)
                if torch.is_tensor(v) and v.is_floating_point():
                    vals.append(float(v.abs().max()))
        if vals:
            t = torch.tensor(vals)
            print(k, "max_abs min/mean/max:", float(t.min()), float(t.mean()), float(t.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="/data_C/sdb1/lww/pdbbind/smcl_pilot/processed")
    parser.add_argument("--out_dir", default="/data_C/sdb1/lww/pdbbind/smcl_pilot_sanitized/processed")
    parser.add_argument("--clamp_abs", type=float, default=5000.0)
    args = parser.parse_args()

    mapping = {
        "processed_data_train_surface_masif.pt": "processed_data_train_surface_masif.pt",
        "processed_data_test1_surface_masif.pt": "processed_data_test1_surface_masif.pt",
        "processed_data_test2_surface_masif.pt": "processed_data_test2_surface_masif.pt",
    }

    for src_name, dst_name in mapping.items():
        sanitize_pt(
            Path(args.in_dir) / src_name,
            Path(args.out_dir) / dst_name,
            clamp_abs=args.clamp_abs,
        )


if __name__ == "__main__":
    main()
