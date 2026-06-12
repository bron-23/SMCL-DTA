#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


def norm_id(x):
    return str(x).strip().lower()[:4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gign_data_dir",
        default="/data_C/sdb1/lww/pdbbind/gign_repo/GNN_DTI/data",
    )
    parser.add_argument(
        "--our_manifest",
        default="/data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_valid_resplit_manifest_with_seq_valid.csv",
    )
    parser.add_argument(
        "--out_manifest",
        default="/data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_gign_exact_manifest.csv",
    )
    parser.add_argument(
        "--out_missing",
        default="/data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_gign_exact_missing.csv",
    )
    args = parser.parse_args()

    gign_dir = Path(args.gign_data_dir)
    our_manifest = Path(args.our_manifest)

    ours = pd.read_csv(our_manifest)
    ours["_pdbid_norm"] = ours["pdb_id"].map(norm_id)

    # 用 GIGN 标签覆盖，确保 label 与 GIGN split 文件一致
    split_files = {
        "train": "train.csv",
        "val": "valid.csv",
        "core2016": "test2016.csv",
    }

    all_rows = []
    missing_rows = []

    for split, fname in split_files.items():
        gdf = pd.read_csv(gign_dir / fname)
        gdf["_pdbid_norm"] = gdf["pdbid"].map(norm_id)
        gdf["_gign_order"] = range(len(gdf))
        gdf = gdf.rename(columns={"-logKd/Ki": "gign_label"})

        merged = gdf.merge(
            ours,
            on="_pdbid_norm",
            how="left",
            suffixes=("_gign", "_ours"),
            indicator=True,
        )

        hit = merged[merged["_merge"] == "both"].copy()
        miss = merged[merged["_merge"] == "left_only"].copy()

        print("=" * 100)
        print(split)
        print("GIGN rows:", len(gdf))
        print("matched:", len(hit))
        print("missing:", len(miss))

        if len(miss) > 0:
            print("missing pdbids:", miss["pdbid"].head(50).tolist())

        # 保持 GIGN 原始顺序
        hit = hit.sort_values("_gign_order")

        # 统一字段
        hit["split"] = split
        hit["label"] = hit["gign_label"].astype(float)
        hit["pdb_id"] = hit["_pdbid_norm"]

        all_rows.append(hit)

        if len(miss) > 0:
            miss_out = pd.DataFrame({
                "split": split,
                "pdbid": miss["pdbid"],
                "gign_label": miss["gign_label"],
            })
            missing_rows.append(miss_out)

    out = pd.concat(all_rows, ignore_index=True)

    # 去掉 merge 中产生的辅助列，保留你原 manifest 的关键字段
    drop_cols = [
        "_merge", "_pdbid_norm", "_gign_order",
        "pdbid", "gign_label",
        "split_gign", "split_ours",
    ]
    for c in drop_cols:
        if c in out.columns:
            out = out.drop(columns=[c])

    # 确保 split/label/pdb_id 在前面
    front = ["pdb_id", "split", "label"]
    cols = front + [c for c in out.columns if c not in front]
    out = out[cols]

    out_path = Path(args.out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    if missing_rows:
        miss_df = pd.concat(missing_rows, ignore_index=True)
    else:
        miss_df = pd.DataFrame(columns=["split", "pdbid", "gign_label"])

    miss_path = Path(args.out_missing)
    miss_df.to_csv(miss_path, index=False)

    print("=" * 100)
    print("[DONE]")
    print("out_manifest:", out_path)
    print("shape:", out.shape)
    print(out["split"].value_counts())
    print("out_missing:", miss_path)
    print("missing shape:", miss_df.shape)

    # leakage check
    print("=" * 100)
    print("[LEAKAGE CHECK]")
    for a in ["train", "val", "core2016"]:
        for b in ["train", "val", "core2016"]:
            if a >= b:
                continue
            A = set(out.loc[out["split"] == a, "pdb_id"])
            B = set(out.loc[out["split"] == b, "pdb_id"])
            inter = A & B
            print(f"{a} ∩ {b}: {len(inter)}")


if __name__ == "__main__":
    main()
