#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader


class LoadedPTDataset(InMemoryDataset):
    def __init__(self, pt_path):
        super().__init__(".")
        self.data, self.slices = torch.load(pt_path, map_location="cpu")

    def len(self):
        return int(self.slices["y"].numel() - 1)


def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location("model_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(model_py):
    module = load_module_from_path(model_py)

    candidates = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj.__module__ == module.__name__:
            candidates.append((name, obj))

    priority = []
    for name, obj in candidates:
        lname = name.lower()
        if "mgraph" in lname or "dta" in lname or "net" in lname:
            priority.append((name, obj))

    if not priority:
        priority = candidates

    init_trials = [
    {"block_num": 3, "vocab_protein_size": 26, "embedding_size": 128, "use_surface": True},
    {"block_num": 3, "vocab_size": 26, "embedding_size": 128, "use_surface": True},
    {"block_num": 3, "vocab_protein_size": 26, "embedding_size": 128},
    {"block_num": 3, "vocab_size": 26, "embedding_size": 128},
    {},
]

    errors = []
    for name, cls in priority:
        for kwargs in init_trials:
            try:
                model = cls(**kwargs)
                print(f"[INFO] Built model class: {name}, kwargs={kwargs}")
                return model
            except Exception as e:
                errors.append(f"{name} kwargs={kwargs}: {repr(e)}")

    raise RuntimeError("Failed to instantiate model:\n" + "\n".join(errors[:20]))


def load_finetuned_checkpoint(model, ckpt_path):
    print(f"[INFO] Loading fine-tuned checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    model.load_state_dict(sd, strict=True)
    return model


def call_model(model, batch):
    out = model(batch)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.view(-1)


def spearman_corr(y_true, y_pred):
    a = pd.Series(y_true).rank(method="average").to_numpy()
    b = pd.Series(y_pred).rank(method="average").to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pearson_corr(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    return {
        "n_total": int(len(mask)),
        "n_valid": int(len(y_true)),
        "n_removed_nonfinite": int(len(mask) - len(y_true)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "spearman": spearman_corr(y_true, y_pred),
        "pearson": pearson_corr(y_true, y_pred),
        "y_true_mean": float(np.mean(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
        "y_true_std": float(np.std(y_true)),
        "y_pred_std": float(np.std(y_pred)),
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    ys = []
    ps = []

    for batch in loader:
        batch = batch.to(device)
        y = batch.y.view(-1).float()
        pred = call_model(model, batch)

        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    return np.concatenate(ys), np.concatenate(ps)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pt", required=True)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--model_py", default="/home/lww/learn_project/mydta/src/model_0428_16_dual.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[INFO] Device: {device}")

    print("[INFO] Loading test dataset...")
    ds = LoadedPTDataset(args.pt)
    print(f"[INFO] Test samples: {len(ds)}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("[INFO] Building model...")
    model = build_model(args.model_py)
    model = load_finetuned_checkpoint(model, args.checkpoint)
    model = model.to(device)

    print("[INFO] Predicting...")
    y_true, y_pred = predict(model, loader, device)

    metrics = compute_metrics(y_true, y_pred)

    print("[RESULT]")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    rows = pd.read_csv(args.rows_csv)
    if len(rows) != len(y_true):
        print(f"[WARN] rows_csv length {len(rows)} != predictions length {len(y_true)}")

    n = min(len(rows), len(y_true))
    rows = rows.iloc[:n].copy()
    rows["y_true"] = y_true[:n]
    rows["y_pred"] = y_pred[:n]
    rows["abs_error"] = np.abs(rows["y_true"] - rows["y_pred"])
    rows["squared_error"] = (rows["y_true"] - rows["y_pred"]) ** 2

    pred_csv = out_dir / "predictions.csv"
    metrics_json = out_dir / "metrics.json"

    rows.to_csv(pred_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[OUT] Predictions CSV: {pred_csv}")
    print(f"[OUT] Metrics JSON: {metrics_json}")
    print("[DONE] Inference finished.")


if __name__ == "__main__":
    main()