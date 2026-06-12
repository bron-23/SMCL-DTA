#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
from torch_geometric.loader import DataLoader

# 保证 src 能被 import
PROJECT_ROOT = "/home/lww/learn_project/mydta"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in [PROJECT_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset import GNNDataset

try:
    from model_0428_16_dual import MGraphDTA
    print("[INFO] Imported MGraphDTA from root/model_0428_16_dual.py")
except Exception:
    from src.model_0428_16_dual import MGraphDTA
    print("[INFO] Imported MGraphDTA from src/model_0428_16_dual.py")


def summarize_output(out):
    print("[INFO] forward output type:", type(out))

    if torch.is_tensor(out):
        print("[INFO] tensor output:", tuple(out.shape), out.dtype)
        return

    if isinstance(out, (tuple, list)):
        print("[INFO] tuple/list length:", len(out))
        for i, item in enumerate(out):
            if torch.is_tensor(item):
                print(f"  output[{i}]: tensor {tuple(item.shape)} {item.dtype}")
            else:
                print(f"  output[{i}]: {type(item)}")
        return

    if isinstance(out, dict):
        print("[INFO] dict keys:", out.keys())
        for k, v in out.items():
            if torch.is_tensor(v):
                print(f"  {k}: tensor {tuple(v.shape)} {v.dtype}")
            else:
                print(f"  {k}: {type(v)}")
        return

    print("[INFO] raw output:", out)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = "/data_C/sdb1/lww/pdbbind/smcl_pilot"

    train_set = GNNDataset(root, types="train", use_surface=True, use_masif=True)
    loader = DataLoader(train_set, batch_size=8, shuffle=False)

    batch = next(iter(loader)).to(device)

    print("[INFO] batch loaded")
    print("x:", batch.x.shape)
    print("edge_index:", batch.edge_index.shape)
    print("edge_attr:", batch.edge_attr.shape)
    print("target:", batch.target.shape)
    print("y:", batch.y.shape)
    print("protein_surface:", batch.protein_surface.shape)
    print("ligand_surface:", batch.ligand_surface.shape)
    print("ligand_global:", batch.ligand_global.shape)

    # 使用现有训练脚本的参数
    model = MGraphDTA(
        3,
        25 + 1,
        embedding_size=128,
        filter_num=32,
        out_dim=1,
        use_surface=True,
    ).to(device)

    model.eval()

    with torch.no_grad():
        try:
            out = model(batch, apply_masking=False)
        except TypeError:
            out = model(batch)

    summarize_output(out)

    print("[DONE] forward smoke test completed.")


if __name__ == "__main__":
    main()
