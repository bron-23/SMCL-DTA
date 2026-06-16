#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 6B: External validation inference on MMAtt-DTA S1 kinase subset.

Inputs:
    - external surface_masif .pt file
    - kept rows CSV from Step 5C
    - four KIBA checkpoints

Outputs:
    - external_validation_predictions.csv
    - external_validation_metrics.json

Notes:
    - Checkpoints are model.state_dict() OrderedDict files.
    - No isotonic calibration is fitted on the external test set.
    - Ensemble prediction is the mean of four checkpoint predictions.
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader


class LoadedPTDataset(InMemoryDataset):
    def __init__(self, pt_path):
        super().__init__(".")
        self.data, self.slices = torch.load(pt_path, weights_only=False)

    def len(self):
        return int(self.slices["y"].numel() - 1)


def extract_model_output(output):
    """Handle model outputs that may be tensor, tuple, list, or dict."""
    if isinstance(output, dict):
        for key in ["pred", "prediction", "y_pred", "output", "out"]:
            if key in output:
                output = output[key]
                break
        else:
            raise ValueError(f"Cannot find prediction tensor in model output dict keys: {output.keys()}")

    if isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported model output type: {type(output)}")

    return output.view(-1)


def infer_model_hparams_from_state_dict(state_dict):
    """Infer only the parameters needed to instantiate MGraphDTA."""
    embed_weight = state_dict["protein_encoder.embed.weight"]
    vocab_protein_size = int(embed_weight.shape[0])
    embedding_size = int(embed_weight.shape[1])

    block_ids = []
    prefix = "protein_encoder.block_list."
    for key in state_dict.keys():
        if key.startswith(prefix):
            rest = key[len(prefix):]
            block_id = rest.split(".")[0]
            if block_id.isdigit():
                block_ids.append(int(block_id))
    block_num = max(block_ids) + 1 if block_ids else 3

    return {
        "block_num": block_num,
        "vocab_protein_size": vocab_protein_size,
        "embedding_size": embedding_size,
    }


def load_model(model_path, checkpoint_path, device):
    sys.path.insert(0, str(Path(model_path).parent))

    import importlib.util
    spec = importlib.util.spec_from_file_location("model_0428_16_dual", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    MGraphDTA = model_module.MGraphDTA

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = infer_model_hparams_from_state_dict(state_dict)

    print(f"[INFO] Inferred model hparams: {hparams}")

    model = MGraphDTA(
        block_num=hparams["block_num"],
        vocab_protein_size=hparams["vocab_protein_size"],
        embedding_size=hparams["embedding_size"],
        use_surface=True,
        disable_masking=True,
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict_one_checkpoint(model, loader, device, max_batches=0):
    y_true_all = []
    y_pred_all = []

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        batch = batch.to(device)
        output = model(batch)
        pred = extract_model_output(output)

        y = batch.y.view(-1)

        y_true_all.append(y.detach().cpu())
        y_pred_all.append(pred.detach().cpu())

        if (batch_idx + 1) % 50 == 0:
            print(f"[INFO] Predicted batches: {batch_idx + 1}")

    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()

    return y_true, y_pred


def rank_average(values):
    """Average ranks, equivalent to scipy.stats.rankdata(method='average')."""
    return pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)


def pearson_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_corr(y_true, y_pred):
    return pearson_corr(rank_average(y_true), rank_average(y_pred))


def concordance_index_fast(y_true, y_pred):
    """
    Exact C-index using Fenwick tree.
    Comparable pairs are pairs with different y_true.
    Prediction ties count as 0.5.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    order = np.argsort(y_true, kind="mergesort")
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]

    unique_preds = np.unique(y_pred_sorted)
    pred_rank = {v: i + 1 for i, v in enumerate(unique_preds)}
    m = len(unique_preds)

    bit = np.zeros(m + 2, dtype=np.int64)

    def bit_add(i, delta):
        while i <= m:
            bit[i] += delta
            i += i & -i

    def bit_sum(i):
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    total_score = 0.0
    total_pairs = 0
    previous_count = 0

    n = len(y_true_sorted)
    start = 0

    while start < n:
        end = start + 1
        while end < n and y_true_sorted[end] == y_true_sorted[start]:
            end += 1

        group_preds = y_pred_sorted[start:end]

        for p in group_preds:
            r = pred_rank[p]
            less = bit_sum(r - 1)
            equal = bit_sum(r) - bit_sum(r - 1)

            total_score += less + 0.5 * equal
            total_pairs += previous_count

        for p in group_preds:
            bit_add(pred_rank[p], 1)

        previous_count += (end - start)
        start = end

    if total_pairs == 0:
        return float("nan")

    return float(total_score / total_pairs)


def compute_metrics(y_true, y_pred):
    y_true_raw = np.asarray(y_true, dtype=np.float64)
    y_pred_raw = np.asarray(y_pred, dtype=np.float64)

    finite_mask = np.isfinite(y_true_raw) & np.isfinite(y_pred_raw)

    n_total = int(len(y_true_raw))
    n_valid = int(np.sum(finite_mask))
    n_removed = int(n_total - n_valid)

    if n_valid < 2:
        return {
            "n_total": n_total,
            "n_valid": n_valid,
            "n_removed_nonfinite": n_removed,
            "mse": float("nan"),
            "rmse": float("nan"),
            "ci": float("nan"),
            "spearman": float("nan"),
            "pearson": float("nan"),
            "y_true_mean": float("nan"),
            "y_pred_mean": float("nan"),
            "y_true_std": float("nan"),
            "y_pred_std": float("nan"),
        }

    y_true = y_true_raw[finite_mask]
    y_pred = y_pred_raw[finite_mask]

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_removed_nonfinite": n_removed,
        "mse": mse,
        "rmse": rmse,
        "ci": concordance_index_fast(y_true, y_pred),
        "spearman": spearman_corr(y_true, y_pred),
        "pearson": pearson_corr(y_true, y_pred),
        "y_true_mean": float(np.mean(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
        "y_true_std": float(np.std(y_true)),
        "y_pred_std": float(np.std(y_pred)),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pt",
        type=str,
        default="/data_C/sdb1/lww/mmatt_s1_surface_overlap/processed_data_mmatt_s1_kinase_surface_masif_overlap.pt",
    )
    parser.add_argument(
        "--rows_csv",
        type=str,
        default="/data_C/sdb1/lww/mmatt_s1_surface_overlap/mmatt_s1_kinase_surface_masif_overlap_rows.csv",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/lww/learn_project/mydta/src/model_0428_16_dual.py",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/home/lww/learn_project/mydta/checkpoints",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data_C/sdb1/lww/mmatt_s1_surface_overlap/step6_external_validation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="Debug only. If >0, predict only the first N batches.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[INFO] Step 6 external validation inference")
    print("=" * 80)
    print(f"[INFO] PT: {args.pt}")
    print(f"[INFO] Rows CSV: {args.rows_csv}")
    print(f"[INFO] Model: {args.model_path}")
    print(f"[INFO] Checkpoint dir: {args.checkpoint_dir}")
    print(f"[INFO] Device: {args.device}")

    device = torch.device(args.device)

    print("[INFO] Loading external dataset...")
    dataset = LoadedPTDataset(args.pt)
    print(f"[INFO] Dataset length: {len(dataset)}")

    rows_df = pd.read_csv(args.rows_csv)
    if args.max_batches > 0:
        max_samples = min(len(dataset), args.max_batches * args.batch_size)
        rows_df = rows_df.iloc[:max_samples].copy()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    checkpoint_paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")))
    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(f"No .pt checkpoints found in {args.checkpoint_dir}")

    print(f"[INFO] Found checkpoints: {len(checkpoint_paths)}")
    for p in checkpoint_paths:
        print(f"  - {p}")

    all_preds = []
    y_true_reference = None
    metrics = {}

    for ckpt_path in checkpoint_paths:
        ckpt_name = Path(ckpt_path).stem
        print("\n" + "=" * 80)
        print(f"[INFO] Predicting with checkpoint: {ckpt_name}")
        print("=" * 80)

        model = load_model(args.model_path, ckpt_path, device)
        y_true, y_pred = predict_one_checkpoint(
            model=model,
            loader=loader,
            device=device,
            max_batches=args.max_batches,
        )

        if y_true_reference is None:
            y_true_reference = y_true
        else:
            if len(y_true) != len(y_true_reference):
                raise ValueError("Different prediction lengths across checkpoints.")
            if not np.allclose(y_true, y_true_reference):
                raise ValueError("y_true differs across checkpoints.")

        all_preds.append(y_pred)

        ckpt_metrics = compute_metrics(y_true, y_pred)
        metrics[ckpt_name] = ckpt_metrics

        print(f"[RESULT] {ckpt_name}")
        print(json.dumps(ckpt_metrics, indent=2))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred_matrix = np.vstack(all_preds)
    
    with np.errstate(all="ignore"):
        ensemble_pred = np.nanmean(pred_matrix, axis=0)

# If all checkpoints are NaN for a sample, keep it as NaN.
    all_nan_mask = np.all(~np.isfinite(pred_matrix), axis=0)
    ensemble_pred[all_nan_mask] = np.nan
    
    ensemble_metrics = compute_metrics(y_true_reference, ensemble_pred)
    metrics["ensemble_mean"] = ensemble_metrics

    print("\n" + "=" * 80)
    print("[RESULT] Ensemble mean")
    print("=" * 80)
    print(json.dumps(ensemble_metrics, indent=2))

    # Save prediction CSV
    out_pred_df = rows_df.copy()
    n_pred = len(y_true_reference)
    out_pred_df = out_pred_df.iloc[:n_pred].copy()

    out_pred_df["y_true"] = y_true_reference

    for ckpt_path, pred in zip(checkpoint_paths, all_preds):
        safe_name = Path(ckpt_path).stem
        safe_name = safe_name.replace(" ", "_").replace(",", "").replace(":", "").replace("[", "").replace("]", "")
        out_pred_df[f"pred_{safe_name}"] = pred

    out_pred_df["pred_ensemble_mean"] = ensemble_pred

    pred_csv_path = out_dir / "external_validation_predictions.csv"
    out_pred_df.to_csv(pred_csv_path, index=False)

    metrics_path = out_dir / "external_validation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n[DONE] Step 6 finished.")
    print(f"[OUT] Predictions CSV: {pred_csv_path}")
    print(f"[OUT] Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()