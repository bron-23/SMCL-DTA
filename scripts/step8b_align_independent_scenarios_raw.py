#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_scenario_csvs():
    paths = {
        "A_imputation": "/data_C/sdb1/lww/mmatt_training/data_unzipped/data/independent_test/bioactivity_imputation/kinase_bioactivity_imputation.csv",
        "B_new_compound": "/data_C/sdb1/lww/mmatt_training/data_unzipped/data/independent_test/new_compound/kinase_new_compound.csv",
        "C_new_compound_new_target": "/data_C/sdb1/lww/mmatt_training/data_unzipped/data/independent_test/new_compound_new_target/kinase_new_compound_new_target.csv",
    }

    out = []
    for scenario, path in paths.items():
        df = pd.read_csv(path)
        tmp = df[["pchembl_value"]].copy()
        tmp["scenario"] = scenario
        tmp["scenario_row"] = np.arange(len(tmp))
        tmp["scenario_file"] = path
        out.append(tmp)

    return pd.concat(out, ignore_index=True)


def align_by_order_allow_raw_deletions(raw_k, scen, tolerance=1e-8):
    """
    Align scenario pchembl sequence to raw kinase pchembl sequence.
    Assumption:
      scenario rows preserve raw order, but some raw rows are omitted from scenario files.
    """
    raw_vals = pd.to_numeric(raw_k["pchembl_value"], errors="coerce").to_numpy(dtype=float)
    scen_vals = pd.to_numeric(scen["pchembl_value"], errors="coerce").to_numpy(dtype=float)

    i = 0
    j = 0
    matched = []
    omitted_raw_indices = []

    while i < len(raw_vals) and j < len(scen_vals):
        rv = raw_vals[i]
        sv = scen_vals[j]

        if np.isclose(rv, sv, atol=tolerance, rtol=0, equal_nan=True):
            matched.append((i, j))
            i += 1
            j += 1
        else:
            # raw row is likely omitted from official scenario files
            omitted_raw_indices.append(i)
            i += 1

    # Remaining raw rows are also omitted.
    while i < len(raw_vals):
        omitted_raw_indices.append(i)
        i += 1

    return matched, omitted_raw_indices, j


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_independent",
        default="/data_C/sdb1/lww/mmatt_training/data_unzipped/data/independent_test/test_data_undivided_unfeaturized.csv",
    )
    parser.add_argument(
        "--out_root",
        default="/data_C/sdb1/lww/mmatt_independent_tests_raw_aligned",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.raw_independent)
    raw_k = raw[raw["protein_class"].astype(str).str.lower().eq("kinase")].copy().reset_index(drop=False)
    raw_k = raw_k.rename(columns={"index": "original_raw_index"})

    scen = read_scenario_csvs().reset_index(drop=True)

    print("=" * 100)
    print("[INFO] Raw kinase rows:", len(raw_k))
    print("[INFO] Scenario total rows:", len(scen))
    print("[INFO] Scenario counts:")
    print(scen["scenario"].value_counts())

    matched, omitted, consumed = align_by_order_allow_raw_deletions(raw_k, scen)

    print("=" * 100)
    print("[ALIGNMENT]")
    print("Matched scenario rows:", len(matched))
    print("Scenario rows consumed:", consumed, "/", len(scen))
    print("Omitted raw kinase rows:", len(omitted))
    print("Expected omitted:", len(raw_k) - len(scen))

    if consumed != len(scen):
        raise RuntimeError(f"Only consumed {consumed}/{len(scen)} scenario rows. Alignment failed.")

    if len(matched) != len(scen):
        raise RuntimeError(f"Matched {len(matched)} rows, expected {len(scen)}.")

    if len(omitted) != len(raw_k) - len(scen):
        print("[WARN] Omitted raw count differs from expected.")

    # Build aligned dataframe.
    raw_indices = [x[0] for x in matched]
    scen_indices = [x[1] for x in matched]

    aligned = raw_k.iloc[raw_indices].copy().reset_index(drop=True)
    scen_matched = scen.iloc[scen_indices].copy().reset_index(drop=True)

    aligned["scenario"] = scen_matched["scenario"]
    aligned["scenario_row"] = scen_matched["scenario_row"]
    aligned["scenario_pchembl_value"] = scen_matched["pchembl_value"]

    # Verify pchembl equality after alignment.
    ok = np.isclose(
        pd.to_numeric(aligned["pchembl_value"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(aligned["scenario_pchembl_value"], errors="coerce").to_numpy(dtype=float),
        atol=1e-8,
        rtol=0,
        equal_nan=True,
    )
    print("[VERIFY] pchembl matched:", int(ok.sum()), "/", len(ok), "ratio:", float(ok.mean()))

    if ok.sum() != len(ok):
        bad = np.where(~ok)[0][:20]
        print("[BAD EXAMPLES]")
        print(aligned.iloc[bad][["pchembl_value", "scenario_pchembl_value", "scenario", "scenario_row"]])
        raise RuntimeError("pchembl verification failed after alignment.")

    # Save all aligned rows.
    all_out = out_root / "kinase_independent_all_scenarios_aligned_raw.csv"
    aligned.to_csv(all_out, index=False)

    omitted_df = raw_k.iloc[omitted].copy()
    omitted_out = out_root / "kinase_independent_omitted_raw_rows_not_in_official_scenarios.csv"
    omitted_df.to_csv(omitted_out, index=False)

    # Save scenario-specific raw and SMCL raw.
    scenario_map = {
        "A_imputation": "A_imputation",
        "B_new_compound": "B_new_compound",
        "C_new_compound_new_target": "C_new_compound_new_target",
    }

    for scenario, dirname in scenario_map.items():
        sub = aligned[aligned["scenario"].eq(scenario)].copy()
        scenario_dir = out_root / dirname
        scenario_dir.mkdir(parents=True, exist_ok=True)

        raw_out = scenario_dir / f"kinase_{dirname}_raw_aligned.csv"
        sub.to_csv(raw_out, index=False)

        smcl = pd.DataFrame({
            "compound_iso_smiles": sub["canonical_smiles"],
            "target_id": sub["uniprot_id"],
            "affinity": sub["pchembl_value"],
            "protein_class": sub["protein_class"],
        })
        smcl_out = scenario_dir / f"kinase_{dirname}_smcl_raw.csv"
        smcl.to_csv(smcl_out, index=False)

        print("-" * 100)
        print(scenario)
        print("rows:", len(sub))
        print("unique compounds:", sub["canonical_smiles"].nunique())
        print("unique targets:", sub["uniprot_id"].nunique())
        print("raw_out:", raw_out)
        print("smcl_out:", smcl_out)

    summary = []
    summary.append("Step 8B independent scenario raw alignment summary")
    summary.append("=" * 100)
    summary.append(f"Raw independent: {args.raw_independent}")
    summary.append(f"Raw kinase rows: {len(raw_k)}")
    summary.append(f"Scenario total rows: {len(scen)}")
    summary.append(f"Matched rows: {len(matched)}")
    summary.append(f"Omitted raw kinase rows: {len(omitted)}")
    summary.append(f"pchembl matched after alignment: {int(ok.sum())}/{len(ok)}")
    summary.append("")
    summary.append("Scenario counts")
    summary.append("-" * 100)
    for k, v in aligned["scenario"].value_counts().items():
        summary.append(f"{k}: {v}")
    summary.append("")
    summary.append("Output files")
    summary.append("-" * 100)
    summary.append(str(all_out))
    summary.append(str(omitted_out))

    summary_path = out_root / "step8b_alignment_summary.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    print("=" * 100)
    print("\n".join(summary))
    print("[DONE]")


if __name__ == "__main__":
    main()