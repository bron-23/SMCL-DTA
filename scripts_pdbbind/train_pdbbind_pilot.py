#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None


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


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cuda_id):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_id}")
    return torch.device("cpu")


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    if len(y_true) < 2:
        pearson = float("nan")
        spearman = float("nan")
    else:
        if pearsonr is not None:
            pearson = float(pearsonr(y_true, y_pred)[0])
        else:
            pearson = float(np.corrcoef(y_true, y_pred)[0, 1])

        if spearmanr is not None:
            spearman = float(spearmanr(y_true, y_pred)[0])
        else:
            spearman = float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
    }


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        pred = model(batch)
        pred = pred.view(-1)
        y = batch.y.view(-1).float()

        loss = criterion(pred, y)
        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        n = y.numel()
        total_loss += float(loss.item()) * n
        total_n += n

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_n = 0

    ys = []
    preds = []

    for batch in loader:
        batch = batch.to(device)

        pred = model(batch).view(-1)
        y = batch.y.view(-1).float()

        loss = criterion(pred, y)

        n = y.numel()
        total_loss += float(loss.item()) * n
        total_n += n

        ys.append(y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_n, 1)
    return metrics, y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/data_C/sdb1/lww/pdbbind/smcl_pilot")
    parser.add_argument("--out_dir", type=str, default="/data_C/sdb1/lww/pdbbind/runs/pilot_smcl")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--filter_num", type=int, default=32)
    parser.add_argument("--mask_rate", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--cl_similarity_threshold", type=float, default=0.5)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.cuda)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "training_log.csv"
    best_ckpt = out_dir / "best_model.pt"
    last_ckpt = out_dir / "last_model.pt"
    pred_path = out_dir / "core2016_predictions_best.csv"

    print("=" * 100)
    print("[INFO] PDBbind pilot training")
    print("[INFO] root:", args.root)
    print("[INFO] out_dir:", out_dir)
    print("[INFO] device:", device)
    print("[INFO] seed:", args.seed)
    print("=" * 100)

    train_set = GNNDataset(args.root, types="train", use_surface=True, use_masif=True)
    val_set = GNNDataset(args.root, types="test1", use_surface=True, use_masif=True)
    core_set = GNNDataset(args.root, types="test2", use_surface=True, use_masif=True)

    print("[INFO] dataset sizes:")
    print("train:", len(train_set))
    print("val:", len(val_set))
    print("core2016:", len(core_set))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    core_loader = DataLoader(core_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MGraphDTA(
        3,
        25 + 1,
        embedding_size=args.embedding_size,
        filter_num=args.filter_num,
        out_dim=1,
        mask_rate=args.mask_rate,
        temperature=args.temperature,
        disable_masking=False,
        cl_mode="regression",
        cl_similarity_threshold=args.cl_similarity_threshold,
        use_surface=True,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=1e-6,
    )

    best_val_rmse = float("inf")
    best_epoch = -1
    bad_epochs = 0

    with open(log_path, "w") as f:
        f.write(
            "epoch,lr,train_loss,"
            "val_loss,val_rmse,val_mae,val_pearson,val_spearman,"
            "core_loss,core_rmse,core_mae,core_pearson,core_spearman\n"
        )

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=args.grad_clip,
        )

        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        core_metrics, _, _ = evaluate(model, core_loader, criterion, device)

        scheduler.step()

        line = (
            f"{epoch},{lr_now:.8g},{train_loss:.8f},"
            f"{val_metrics['loss']:.8f},{val_metrics['rmse']:.8f},{val_metrics['mae']:.8f},"
            f"{val_metrics['pearson']:.8f},{val_metrics['spearman']:.8f},"
            f"{core_metrics['loss']:.8f},{core_metrics['rmse']:.8f},{core_metrics['mae']:.8f},"
            f"{core_metrics['pearson']:.8f},{core_metrics['spearman']:.8f}"
        )

        with open(log_path, "a") as f:
            f.write(line + "\n")

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f}, val_mae={val_metrics['mae']:.4f}, "
            f"val_p={val_metrics['pearson']:.4f}, val_s={val_metrics['spearman']:.4f} | "
            f"core_rmse={core_metrics['rmse']:.4f}, core_mae={core_metrics['mae']:.4f}, "
            f"core_p={core_metrics['pearson']:.4f}, core_s={core_metrics['spearman']:.4f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "val_metrics": val_metrics,
                    "core_metrics": core_metrics,
                },
                best_ckpt,
            )
            print(f"[BEST] epoch={epoch}, val_rmse={best_val_rmse:.4f}, saved={best_ckpt}")
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"[EARLY STOP] no val RMSE improvement for {args.patience} epochs.")
            break

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        last_ckpt,
    )

    # Evaluate best checkpoint.
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
    core_metrics, y_core, p_core = evaluate(model, core_loader, criterion, device)

    import pandas as pd
    pd.DataFrame({"y_true": y_core, "y_pred": p_core}).to_csv(pred_path, index=False)

    elapsed = time.time() - start

    print("=" * 100)
    print("[FINAL BEST]")
    print("best_epoch:", best_epoch)
    print("best_val_rmse:", best_val_rmse)
    print("val:", val_metrics)
    print("core2016:", core_metrics)
    print("best_ckpt:", best_ckpt)
    print("predictions:", pred_path)
    print("log:", log_path)
    print(f"elapsed_sec: {elapsed:.1f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
