#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = "/home/lww/learn_project/mydta"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in [PROJECT_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset import GNNDataset
from model_0428_16_dual import MGraphDTA


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "spearman": float(spearmanr(y_true, y_pred)[0]),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).view(-1)
        y = batch.y.view(-1).float()

        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return compute_metrics(y_true, y_pred), y_true, y_pred


def build_model_from_ckpt(ckpt, device):
    args = ckpt.get("args", {})

    model = MGraphDTA(
        3,
        25 + 1,
        embedding_size=int(args.get("embedding_size", 128)),
        filter_num=int(args.get("filter_num", 32)),
        out_dim=1,
        mask_rate=float(args.get("mask_rate", 0.05)),
        temperature=float(args.get("temperature", 0.1)),
        disable_masking=False,
        cl_mode="regression",
        cl_similarity_threshold=float(args.get("cl_similarity_threshold", 0.5)),
        use_surface=True,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    return model


def main():
    root = "/data_C/sdb1/lww/pdbbind/smcl_gign_test2013_filtered_std"
    run_base = Path("/data_C/sdb1/lww/pdbbind/runs")
    seeds = [42, 43, 44, 45, 46]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test = GNNDataset(root, types="test2", use_surface=True, use_masif=True)
    loader = DataLoader(test, batch_size=128, shuffle=False, num_workers=0)

    print("test2013 samples:", len(test))
    print("device:", device)

    rows = []

    pred_dir = run_base / "gign_exact_test2013_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        ckpt_path = run_base / f"gign_exact_smcl_seed{seed}_e200_pat60" / "best_model.pt"

        if not ckpt_path.exists():
            print("[MISSING]", seed, ckpt_path)
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        model = build_model_from_ckpt(ckpt, device)

        metrics, y_true, y_pred = evaluate(model, loader, device)

        row = {
            "seed": seed,
            "best_epoch": int(ckpt["epoch"]),
            **metrics,
        }
        rows.append(row)

        pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
        }).to_csv(pred_dir / f"test2013_seed{seed}_predictions.csv", index=False)

        print(seed, row)

    df = pd.DataFrame(rows)
    out_csv = run_base / "gign_exact_smcl_test2013_5seeds_summary.csv"
    out_txt = run_base / "gign_exact_smcl_test2013_5seeds_summary.txt"

    df.to_csv(out_csv, index=False)

    lines = []
    lines.append("=" * 100)
    lines.append("[TEST2013 PER-SEED RESULTS]")
    lines.append(df.to_string(index=False))
    lines.append("=" * 100)
    lines.append("[TEST2013 MEAN ± STD]")

    for c in ["rmse", "mae", "pearson", "spearman"]:
        mean = df[c].mean()
        std = df[c].std(ddof=1) if len(df) > 1 else 0.0
        lines.append(f"test2013_{c}: {mean:.4f} ± {std:.4f}")

    text = "\n".join(lines)
    print(text)

    with open(out_txt, "w") as f:
        f.write(text + "\n")

    print("saved:", out_csv)
    print("saved:", out_txt)


if __name__ == "__main__":
    main()
