#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset


class DS(InMemoryDataset):
    def __init__(self, pt_path):
        super().__init__(".")
        self.data, self.slices = torch.load(pt_path, map_location="cpu", weights_only=False)


def is_good_sample(d, eps=1e-8):
    checks = {}

    for key in ["ligand_surface", "ligand_global", "protein_surface", "y", "x", "edge_attr"]:
        v = getattr(d, key, None)

        if v is None or not torch.is_tensor(v):
            checks[key] = False
            continue

        if v.is_floating_point():
            checks[key] = bool(torch.isfinite(v).all())
        else:
            checks[key] = True

    checks["ligand_surface_nonzero"] = float(d.ligand_surface.float().abs().sum()) > eps
    checks["ligand_global_nonzero"] = float(d.ligand_global.float().abs().sum()) > eps
    checks["protein_surface_nonzero"] = float(d.protein_surface.float().abs().sum()) > eps

    good = all(checks.values())
    return good, checks


def filter_pt(in_pt, out_pt, report_csv):
    ds = DS(in_pt)

    kept = []
    rows = []

    for i in range(len(ds)):
        d = ds.get(i)
        good, checks = is_good_sample(d)

        row = {
            "idx": i,
            "protein_id": str(getattr(d, "protein_id", "")),
            "y": float(d.y.view(-1)[0]) if hasattr(d, "y") else None,
            "keep": good,
            "ligand_surface_abs_sum": float(d.ligand_surface.float().abs().sum()) if hasattr(d, "ligand_surface") else None,
            "ligand_global_abs_sum": float(d.ligand_global.float().abs().sum()) if hasattr(d, "ligand_global") else None,
            "protein_surface_abs_sum": float(d.protein_surface.float().abs().sum()) if hasattr(d, "protein_surface") else None,
        }
        row.update(checks)
        rows.append(row)

        if good:
            kept.append(d)

    collater = InMemoryDataset()
    data, slices = collater.collate(kept)

    out_pt = Path(out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save((data, slices), out_pt)

    report = pd.DataFrame(rows)
    report.to_csv(report_csv, index=False)

    print("=" * 100)
    print("in_pt:", in_pt)
    print("out_pt:", out_pt)
    print("report:", report_csv)
    print("original:", len(ds))
    print("kept:", len(kept))
    print("removed:", len(ds) - len(kept))
    if len(report) > 0:
        print("removed samples:")
        print(report[~report["keep"]][["idx", "protein_id", "y", "ligand_surface_abs_sum", "ligand_global_abs_sum", "protein_surface_abs_sum"]].head(50).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="/data_C/sdb1/lww/pdbbind/smcl_pilot_sanitized/processed")
    parser.add_argument("--out_dir", default="/data_C/sdb1/lww/pdbbind/smcl_pilot_filtered/processed")
    args = parser.parse_args()

    files = {
        "train": "processed_data_train_surface_masif.pt",
        "val": "processed_data_test1_surface_masif.pt",
        "core2016": "processed_data_test2_surface_masif.pt",
    }

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, fname in files.items():
        filter_pt(
            in_pt=in_dir / fname,
            out_pt=out_dir / fname,
            report_csv=out_dir.parent / f"{split}_filter_report.csv",
        )


if __name__ == "__main__":
    main()
