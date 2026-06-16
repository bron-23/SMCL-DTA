#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


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


def compute_metrics(df):
    y_true = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    return {
        "n": int(len(y_true)),
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


def add_occurrence_index(df, key_cols):
    df = df.copy()
    df["_occ"] = df.groupby(key_cols, dropna=False).cumcount()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_csv",
        default="/data_C/sdb1/lww/mmatt_s1_surface_overlap/step8_finetuned_epoch1400_inference/predictions.csv",
    )
    parser.add_argument(
        "--aligned_raw_csv",
        default="/data_C/sdb1/lww/mmatt_independent_tests_raw_aligned/kinase_independent_all_scenarios_aligned_raw.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="/data_C/sdb1/lww/mmatt_independent_tests_raw_aligned/step8c_scenario_scores_epoch1400",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.predictions_csv)
    aligned = pd.read_csv(args.aligned_raw_csv)

    print("=" * 100)
    print("[INFO] Loading predictions and aligned official scenarios")
    print("[INFO] predictions:", args.predictions_csv)
    print("[INFO] predictions rows:", len(pred))
    print("[INFO] prediction columns:", pred.columns.tolist())
    print("[INFO] aligned raw:", args.aligned_raw_csv)
    print("[INFO] aligned raw rows:", len(aligned))
    print("[INFO] aligned raw columns:", aligned.columns.tolist())

    # Standardize prediction columns.
    pred2 = pred.copy()
    if "compound_iso_smiles" not in pred2.columns:
        raise ValueError("predictions CSV must contain compound_iso_smiles")
    if "target_id" not in pred2.columns:
        raise ValueError("predictions CSV must contain target_id")
    if "affinity" not in pred2.columns:
        raise ValueError("predictions CSV must contain affinity")
    if "y_pred" not in pred2.columns:
        raise ValueError("predictions CSV must contain y_pred")

    pred2["smiles_key"] = pred2["compound_iso_smiles"].astype(str)
    pred2["target_key"] = pred2["target_id"].astype(str)
    pred2["affinity_key"] = pd.to_numeric(pred2["affinity"], errors="coerce").round(6)

    # If y_true exists, use it. Otherwise use affinity.
    if "y_true" not in pred2.columns:
        pred2["y_true"] = pred2["affinity"]

    # Standardize aligned raw columns.
    aligned2 = aligned.copy()
    aligned2["smiles_key"] = aligned2["canonical_smiles"].astype(str)
    aligned2["target_key"] = aligned2["uniprot_id"].astype(str)
    aligned2["affinity_key"] = pd.to_numeric(aligned2["pchembl_value"], errors="coerce").round(6)

    key_cols = ["smiles_key", "target_key", "affinity_key"]

    # Add occurrence index to handle duplicate identical SMILES-target-affinity rows.
    pred2 = add_occurrence_index(pred2, key_cols)
    aligned2 = add_occurrence_index(aligned2, key_cols)

    merge_cols = key_cols + ["_occ"]

    merged = pred2.merge(
        aligned2[
            merge_cols
            + [
                "scenario",
                "scenario_row",
                "original_raw_index",
                "standard_inchi_key",
                "canonical_smiles",
                "uniprot_id",
                "pref_name",
                "pchembl_value",
            ]
        ],
        on=merge_cols,
        how="left",
        suffixes=("", "_official"),
    )

    matched = merged["scenario"].notna().sum()
    unmatched = merged["scenario"].isna().sum()

    print("=" * 100)
    print("[MATCH]")
    print("Prediction rows:", len(pred2))
    print("Matched to official scenarios:", matched)
    print("Unmatched prediction rows:", unmatched)

    unmatched_df = merged[merged["scenario"].isna()].copy()
    unmatched_path = out_dir / "unmatched_predictions_not_in_official_scenarios.csv"
    unmatched_df.to_csv(unmatched_path, index=False)

    official_pred = merged[merged["scenario"].notna()].copy()
    official_pred_path = out_dir / "official_scenario_predictions.csv"
    official_pred.to_csv(official_pred_path, index=False)

    print("[OUT] official scenario predictions:", official_pred_path)
    print("[OUT] unmatched predictions:", unmatched_path)

    results = {}
    rows = []

    table_s4 = {
        "A_imputation": {"mmatt_spearman": 0.705, "mmatt_rmse": 0.86},
        "B_new_compound": {"mmatt_spearman": 0.294, "mmatt_rmse": 1.24},
        "C_new_compound_new_target": {"mmatt_spearman": 0.308, "mmatt_rmse": 1.05},
    }

    for scenario in ["A_imputation", "B_new_compound", "C_new_compound_new_target"]:
        sub = official_pred[official_pred["scenario"].eq(scenario)].copy()
        metrics = compute_metrics(sub)
        metrics.update(table_s4.get(scenario, {}))
        results[scenario] = metrics

        row = {"scenario": scenario, **metrics}
        if "mmatt_spearman" in row:
            row["delta_spearman_vs_mmatt"] = row["spearman"] - row["mmatt_spearman"]
            row["delta_rmse_vs_mmatt"] = row["rmse"] - row["mmatt_rmse"]
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_csv = out_dir / "scenario_metrics_vs_table_s4.csv"
    result_json = out_dir / "scenario_metrics_vs_table_s4.json"

    result_df.to_csv(result_csv, index=False)
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("[RESULT] Scenario metrics vs MMAtt-DTA Table S4")
    print(result_df.to_string(index=False))
    print("[OUT]", result_csv)
    print("[OUT]", result_json)

    summary = []
    summary.append("Step 8C independent scenario scoring summary")
    summary.append("=" * 100)
    summary.append(f"Predictions CSV: {args.predictions_csv}")
    summary.append(f"Aligned official raw CSV: {args.aligned_raw_csv}")
    summary.append(f"Prediction rows: {len(pred2)}")
    summary.append(f"Matched official scenario prediction rows: {matched}")
    summary.append(f"Unmatched prediction rows: {unmatched}")
    summary.append("")
    summary.append("Scenario metrics vs MMAtt-DTA Table S4")
    summary.append("-" * 100)
    summary.append(result_df.to_string(index=False))
    summary.append("")
    summary.append("Output files")
    summary.append("-" * 100)
    summary.append(str(official_pred_path))
    summary.append(str(unmatched_path))
    summary.append(str(result_csv))
    summary.append(str(result_json))

    summary_path = out_dir / "step8c_scenario_scoring_summary.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print("[OUT]", summary_path)
    print("[DONE]")


if __name__ == "__main__":
    main()