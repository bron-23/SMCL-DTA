#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

official_path = "/data_C/sdb1/lww/mmatt_independent_tests_raw_aligned/kinase_independent_all_scenarios_aligned_raw.csv"
pred_path = "/data_C/sdb1/lww/mmatt_independent_tests_raw_aligned/step8c_scenario_scores_epoch1400/official_scenario_predictions.csv"
out_dir = Path("/data_C/sdb1/lww/mmatt_independent_tests_raw_aligned/step9_c_full_surface")
out_dir.mkdir(parents=True, exist_ok=True)

official = pd.read_csv(official_path)
pred = pd.read_csv(pred_path)

c_off = official[official["scenario"] == "C_new_compound_new_target"].copy()
c_pred = pred[pred["scenario"] == "C_new_compound_new_target"].copy()

pred_target_col = "uniprot_id" if "uniprot_id" in c_pred.columns else "target_id"

official_targets = set(c_off["uniprot_id"].astype(str))
pred_targets = set(c_pred[pred_target_col].astype(str))

missing_targets = sorted(official_targets - pred_targets)
covered_targets = sorted(official_targets & pred_targets)

print("=" * 100)
print("[C official]")
print("rows:", len(c_off))
print("unique compounds:", c_off["canonical_smiles"].nunique())
print("unique targets:", c_off["uniprot_id"].nunique())

print("\n[C predicted currently]")
print("rows:", len(c_pred))
print("unique compounds:", c_pred["compound_iso_smiles"].nunique() if "compound_iso_smiles" in c_pred.columns else "NA")
print("unique targets:", len(pred_targets))

print("\nCovered C targets:")
print(covered_targets)

print("\nMissing C targets:")
print(missing_targets)

print("\n[C official rows by target]")
print(c_off["uniprot_id"].value_counts())

print("\n[C missing rows by target]")
missing_rows = c_off[c_off["uniprot_id"].astype(str).isin(missing_targets)].copy()
print(missing_rows["uniprot_id"].value_counts())

# Save outputs.
c_off.to_csv(out_dir / "C_official_607_raw.csv", index=False)
c_pred.to_csv(out_dir / "C_currently_predicted_12.csv", index=False)
missing_rows.to_csv(out_dir / "C_missing_rows_due_to_no_surface.csv", index=False)

pd.DataFrame({"uniprot_id": missing_targets}).to_csv(out_dir / "C_missing_target_ids.csv", index=False)
pd.DataFrame({"uniprot_id": covered_targets}).to_csv(out_dir / "C_covered_target_ids.csv", index=False)

# Also save full C SMCL raw for later reconstruction.
smcl = pd.DataFrame({
    "compound_iso_smiles": c_off["canonical_smiles"],
    "target_id": c_off["uniprot_id"],
    "affinity": c_off["pchembl_value"],
    "protein_class": c_off["protein_class"],
})
smcl.to_csv(out_dir / "C_new_compound_new_target_full_607_smcl_raw.csv", index=False)

summary = []
summary.append("Step 9A C scenario missing target summary")
summary.append("=" * 100)
summary.append(f"C official rows: {len(c_off)}")
summary.append(f"C current predicted rows: {len(c_pred)}")
summary.append(f"C official unique targets: {c_off['uniprot_id'].nunique()}")
summary.append(f"C currently covered targets: {len(covered_targets)}")
summary.append(f"C missing targets: {len(missing_targets)}")
summary.append("")
summary.append("Covered targets")
summary.append("-" * 100)
summary.extend(covered_targets)
summary.append("")
summary.append("Missing targets")
summary.append("-" * 100)
summary.extend(missing_targets)
summary.append("")
summary.append("Missing rows by target")
summary.append("-" * 100)
summary.append(missing_rows["uniprot_id"].value_counts().to_string())
summary.append("")
summary.append("Output files")
summary.append("-" * 100)
for fn in [
    "C_official_607_raw.csv",
    "C_currently_predicted_12.csv",
    "C_missing_rows_due_to_no_surface.csv",
    "C_missing_target_ids.csv",
    "C_covered_target_ids.csv",
    "C_new_compound_new_target_full_607_smcl_raw.csv",
]:
    summary.append(str(out_dir / fn))

(out_dir / "step9a_c_missing_target_summary.txt").write_text("\n".join(summary), encoding="utf-8")

print("\n[DONE]")
print(out_dir / "step9a_c_missing_target_summary.txt")
