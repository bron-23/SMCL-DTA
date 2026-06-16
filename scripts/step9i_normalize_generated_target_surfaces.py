#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data


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


def get_surface(data):
    for key in ["protein_surface", "protein_surface_x", "protein_surface_feat"]:
        if key in data and torch.is_tensor(data[key]):
            return data[key].detach().cpu().float()
    for key in data.keys():
        low = str(key).lower()
        if "protein" in low and "surface" in low and torch.is_tensor(data[key]):
            return data[key].detach().cpu().float()
    return None


def get_target_id(data):
    for key in ["target_id", "protein_id", "uniprot_id"]:
        if key in data:
            v = data[key]
            if isinstance(v, (list, tuple)):
                v = v[0]
            return str(v)
    return ""


def load_unique_surfaces(pt_files):
    surfaces = {}
    for pt in pt_files:
        print("[LOAD REF]", pt)
        ds = LoadedPTDataset(pt)
        for i in range(len(ds)):
            d = ds.get(i)
            tid = get_target_id(d)
            ps = get_surface(d)
            if tid and ps is not None and ps.ndim == 2:
                # keep first occurrence per target to avoid overweighting frequent targets
                surfaces.setdefault(tid, ps)
    return surfaces


def tensor_channel_stats(tensors):
    arr = torch.cat([t.reshape(-1, t.shape[-1]) for t in tensors], dim=0).numpy().astype(np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def normalize_surface(ps, gen_mean, gen_std, ref_mean, ref_std):
    x = ps.numpy().astype(np.float64)
    y = (x - gen_mean) / gen_std * ref_std + ref_mean
    return torch.tensor(y.astype(np.float32))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--generated_pt",
        default="/data_C/sdb1/lww/mmatt_independent_tests_raw_aligned/step9_c_full_surface/C_missing_targets_surface_masif.pt",
    )
    parser.add_argument(
        "--out_pt",
        default="/data_C/sdb1/lww/mmatt_independent_tests_raw_aligned/step9_c_full_surface/C_missing_targets_surface_masif_normalized_ref_existing.pt",
    )
    parser.add_argument(
        "--out_dir",
        default="/data_C/sdb1/lww/mmatt_independent_tests_raw_aligned/step9_c_full_surface/surface_normalization_ref_existing",
    )
    parser.add_argument(
        "--reference_surface_pts",
        nargs="+",
        required=True,
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("[STEP 9I] Normalize generated C target surfaces")
    print("[GENERATED]", args.generated_pt)

    # 1. Reference distribution from existing KIBA/Davis reused surfaces.
    ref_surfaces = load_unique_surfaces(args.reference_surface_pts)
    print("[INFO] unique reference targets:", len(ref_surfaces))

    if len(ref_surfaces) == 0:
        raise RuntimeError("No reference protein_surface found.")

    ref_mean, ref_std = tensor_channel_stats(list(ref_surfaces.values()))

    # 2. Generated distribution.
    gen_ds = LoadedPTDataset(args.generated_pt)
    gen_data_list = [gen_ds.get(i) for i in range(len(gen_ds))]

    gen_surfaces = []
    for d in gen_data_list:
        ps = get_surface(d)
        if ps is None:
            raise RuntimeError(f"Missing protein_surface for generated target: {get_target_id(d)}")
        gen_surfaces.append(ps)

    gen_mean, gen_std = tensor_channel_stats(gen_surfaces)

    print("[REF mean]", ref_mean)
    print("[REF std ]", ref_std)
    print("[GEN mean]", gen_mean)
    print("[GEN std ]", gen_std)

    # 3. Normalize generated surfaces.
    norm_data_list = []
    records = []

    for d in gen_data_list:
        tid = get_target_id(d)
        ps = get_surface(d)
        ps_norm = normalize_surface(ps, gen_mean, gen_std, ref_mean, ref_std)

        new_d = Data()
        for key in d.keys():
            new_d[key] = d[key]

        # overwrite all possible protein surface keys that exist
        if "protein_surface" in new_d:
            new_d.protein_surface = ps_norm
        else:
            new_d.protein_surface = ps_norm

        before = ps.numpy()
        after = ps_norm.numpy()

        records.append({
            "target_id": tid,
            "before_mean": float(before.mean()),
            "before_std": float(before.std()),
            "before_abs_mean": float(np.abs(before).mean()),
            "before_l2": float(np.sqrt((before ** 2).sum())),
            "after_mean": float(after.mean()),
            "after_std": float(after.std()),
            "after_abs_mean": float(np.abs(after).mean()),
            "after_l2": float(np.sqrt((after ** 2).sum())),
        })

        norm_data_list.append(new_d)

    data, slices = InMemoryDataset.collate(norm_data_list)
    torch.save((data, slices), args.out_pt)

    manifest = pd.DataFrame(records)
    manifest_path = out_dir / "normalized_generated_surface_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    stats = pd.DataFrame({
        "channel": list(range(len(ref_mean))),
        "ref_mean": ref_mean,
        "ref_std": ref_std,
        "gen_mean": gen_mean,
        "gen_std": gen_std,
    })
    stats_path = out_dir / "channelwise_ref_vs_generated_stats.csv"
    stats.to_csv(stats_path, index=False)

    summary = []
    summary.append("Step 9I generated target surface normalization summary")
    summary.append("=" * 100)
    summary.append(f"Generated input pt: {args.generated_pt}")
    summary.append(f"Normalized output pt: {args.out_pt}")
    summary.append(f"Reference unique targets: {len(ref_surfaces)}")
    summary.append(f"Generated targets: {len(gen_data_list)}")
    summary.append("")
    summary.append("Group-level before/after")
    summary.append("-" * 100)
    summary.append(manifest[[
        "before_mean", "before_std", "before_abs_mean", "before_l2",
        "after_mean", "after_std", "after_abs_mean", "after_l2",
    ]].mean().to_string())
    summary.append("")
    summary.append("Output files")
    summary.append("-" * 100)
    summary.append(str(args.out_pt))
    summary.append(str(manifest_path))
    summary.append(str(stats_path))

    summary_path = out_dir / "step9i_surface_normalization_summary.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))
    print("[DONE]")


if __name__ == "__main__":
    main()