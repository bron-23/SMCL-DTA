#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 7F: Fine-tune SMCL-DTA on MMAtt-DTA kinase pChEMBL data from KIBA checkpoint.

Inputs:
    train surface/MaSIF .pt
    val surface/MaSIF .pt
    KIBA-trained checkpoint
    model_0428_16_dual.py

Output:
    best fine-tuned checkpoint
    per-epoch metrics
"""

import argparse
import importlib.util
import inspect
import json
import math
import os
import random
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
        loaded = torch.load(pt_path, map_location="cpu")
        self.data, self.slices = loaded

    def len(self):
        return int(self.slices["y"].numel() - 1)


class LimitedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, limit):
        self.base_dataset = base_dataset
        self.limit = min(limit, len(base_dataset)) if limit and limit > 0 else len(base_dataset)

    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        return self.base_dataset.get(idx)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    if not priority:
        raise RuntimeError(f"No nn.Module class found in {model_py}")

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

    raise RuntimeError("Failed to instantiate model. Errors:\n" + "\n".join(errors[:20]))


def unwrap_checkpoint(raw):
    if isinstance(raw, dict):
        for key in ["model_state_dict", "state_dict", "net", "model", "module"]:
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    return raw


def normalize_state_dict_keys(sd):
    out = {}
    for k, v in sd.items():
        nk = k
        for prefix in ["module.", "model.", "net."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        out[nk] = v
    return out


def load_checkpoint_into_model(model, checkpoint_path):
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location="cpu")
    sd = unwrap_checkpoint(raw)
    sd = normalize_state_dict_keys(sd)

    model_sd = model.state_dict()
    compatible = {}
    skipped = []

    for k, v in sd.items():
        if k in model_sd and hasattr(v, "shape") and model_sd[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)

    load_ratio = len(compatible) / max(1, len(model_sd))
    print(f"[INFO] Compatible checkpoint tensors: {len(compatible)} / model tensors {len(model_sd)}")
    print(f"[INFO] Load ratio: {load_ratio:.3f}")
    print(f"[INFO] Skipped checkpoint tensors: {len(skipped)}")

    if len(compatible) < 10 or load_ratio < 0.5:
        raise RuntimeError(
            "Too few tensors matched between checkpoint and model. "
            "This likely means model class/constructor is wrong."
        )

    model_sd.update(compatible)
    model.load_state_dict(model_sd, strict=True)
    return model


def call_model(model, batch):
    out = model(batch)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.view(-1)


def fast_spearman(y_true, y_pred):
    a = pd.Series(y_true).rank(method="average").to_numpy()
    b = pd.Series(y_pred).rank(method="average").to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pearson(y_true, y_pred):
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
    sp = fast_spearman(y_true, y_pred)
    pr = pearson(y_true, y_pred)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "n": int(len(y_true)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "spearman": sp,
        "pearson": pr,
        "r2": r2,
        "y_true_mean": float(np.mean(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
        "y_true_std": float(np.std(y_true)),
        "y_pred_std": float(np.std(y_pred)),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys = []
    ps = []

    for batch in loader:
        batch = batch.to(device)
        y = batch.y.view(-1).float()
        pred = call_model(model, batch)

        mask = torch.isfinite(y) & torch.isfinite(pred)
        ys.append(y[mask].detach().cpu().numpy())
        ps.append(pred[mask].detach().cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return compute_metrics(y_true, y_pred)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    losses = []
    total_n = 0

    for batch in loader:
        batch = batch.to(device)
        y = batch.y.view(-1).float()

        optimizer.zero_grad(set_to_none=True)
        pred = call_model(model, batch)

        mask = torch.isfinite(y) & torch.isfinite(pred)
        if mask.sum() == 0:
            continue

        loss = criterion(pred[mask], y[mask])
        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        n = int(mask.sum().item())
        losses.append(float(loss.item()) * n)
        total_n += n

    return float(sum(losses) / max(1, total_n))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_pt", type=str, required=True)
    parser.add_argument("--val_pt", type=str, required=True)
    parser.add_argument("--model_py", type=str, default="/home/lww/learn_project/mydta/src/model_0428_16_dual.py")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_val", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[INFO] Device: {device}")

    print("[INFO] Loading datasets...")
    train_base = LoadedPTDataset(args.train_pt)
    val_base = LoadedPTDataset(args.val_pt)

    train_ds = LimitedDataset(train_base, args.limit_train)
    val_ds = LimitedDataset(val_base, args.limit_val)

    print(f"[INFO] Train samples: {len(train_ds)}")
    print(f"[INFO] Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    print("[INFO] Building model...")
    model = build_model(args.model_py)
    model = load_checkpoint_into_model(model, args.checkpoint)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    history = []
    best_rmse = float("inf")
    best_epoch = -1
    bad_epochs = 0

    print("[INFO] Starting fine-tuning...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip)
        val_metrics = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            f"[EPOCH {epoch:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} "
            f"val_spearman={val_metrics['spearman']:.6f} "
            f"val_pearson={val_metrics['pearson']:.6f} "
            f"val_pred_mean={val_metrics['y_pred_mean']:.4f}"
        )

        pd.DataFrame(history).to_csv(out_dir / "finetune_history.csv", index=False)

        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_epoch = epoch
            bad_epochs = 0

            best_path = out_dir / "best_finetuned_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                best_path,
            )

            with open(out_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "best_rmse": best_rmse,
                        "val_metrics": val_metrics,
                        "checkpoint_init": args.checkpoint,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(f"[SAVE] Best model updated: epoch={epoch}, rmse={best_rmse:.6f}")
        else:
            bad_epochs += 1
            print(f"[INFO] No improvement. bad_epochs={bad_epochs}/{args.patience}")

        if args.patience > 0 and bad_epochs >= args.patience:
            print("[EARLY STOP] Patience reached.")
            break

    print("=" * 80)
    print("[DONE] Fine-tuning finished.")
    print(f"[BEST] epoch={best_epoch}, rmse={best_rmse:.6f}")
    print(f"[OUT] {out_dir / 'best_finetuned_model.pt'}")
    print(f"[OUT] {out_dir / 'finetune_history.csv'}")
    print(f"[OUT] {out_dir / 'best_metrics.json'}")


if __name__ == "__main__":
    main()