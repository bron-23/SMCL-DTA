#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


def norm_id(x):
    return str(x).strip().lower()[:4]


gign_test2013 = Path("/data_C/sdb1/lww/pdbbind/gign_repo/GNN_DTI/data/test2013.csv")
our_manifest = Path("/data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_valid_resplit_manifest_with_seq_valid.csv")
out_manifest = Path("/data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_gign_test2013_manifest.csv")

gdf = pd.read_csv(gign_test2013)
ours = pd.read_csv(our_manifest)

gdf["_pdbid_norm"] = gdf["pdbid"].map(norm_id)
ours["_pdbid_norm"] = ours["pdb_id"].map(norm_id)

merged = gdf.merge(
    ours,
    on="_pdbid_norm",
    how="left",
    suffixes=("_gign", "_ours"),
    indicator=True,
)

hit = merged[merged["_merge"] == "both"].copy()
miss = merged[merged["_merge"] == "left_only"].copy()

print("GIGN test2013:", len(gdf))
print("matched:", len(hit))
print("missing:", len(miss))
if len(miss):
    print(miss[["pdbid", "-logKd/Ki"]].to_string(index=False))

# 为了兼容现有 step7 builder，把同一批 test2013 复制成 train / val / core2016 三个 split。
# 之后只使用 core2016 输出作为 test2013 评估集。
rows = []
for split in ["train", "val", "core2016"]:
    tmp = hit.copy()
    tmp["split"] = split
    tmp["label"] = tmp["-logKd/Ki"].astype(float)
    tmp["pdb_id"] = tmp["_pdbid_norm"]
    rows.append(tmp)

out = pd.concat(rows, ignore_index=True)

drop_cols = ["_merge", "_pdbid_norm", "pdbid", "-logKd/Ki"]
for c in drop_cols:
    if c in out.columns:
        out = out.drop(columns=[c])

front = ["pdb_id", "split", "label"]
cols = front + [c for c in out.columns if c not in front]
out = out[cols]

out_manifest.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_manifest, index=False)

print("out:", out_manifest)
print(out["split"].value_counts())
