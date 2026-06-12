#!/usr/bin/env bash
set -uo pipefail

cd /home/lww/learn_project/mydta

BIO_PY="/home/lww/anaconda3/envs/bio/bin/python"

ROOT="/data_C/sdb1/lww/pdbbind/smcl_gign_exact_filtered_std"
RUN_BASE="/data_C/sdb1/lww/pdbbind/runs"
MASTER_LOG="${RUN_BASE}/gign_exact_5seeds_master_inner.log"

CUDA_ID=0
EPOCHS=200
BATCH_SIZE=128
LR=3e-4
WEIGHT_DECAY=1e-4
PATIENCE=60
SEEDS=(42 43 44 45 46)

mkdir -p "${RUN_BASE}"

echo "====================================================================================================" | tee -a "${MASTER_LOG}"
echo "[START] GIGN exact SMCL-DTA 5-seed training" | tee -a "${MASTER_LOG}"
echo "BIO_PY=${BIO_PY}" | tee -a "${MASTER_LOG}"
echo "ROOT=${ROOT}" | tee -a "${MASTER_LOG}"
echo "CUDA_ID=${CUDA_ID}" | tee -a "${MASTER_LOG}"
echo "SEEDS=${SEEDS[*]}" | tee -a "${MASTER_LOG}"
date | tee -a "${MASTER_LOG}"
echo "====================================================================================================" | tee -a "${MASTER_LOG}"

echo "[CHECK] Python / torch environment" | tee -a "${MASTER_LOG}"
"${BIO_PY}" - <<'PY' 2>&1 | tee -a "${MASTER_LOG}"
import sys
import torch
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY

rc=${PIPESTATUS[0]}
if [ "${rc}" -ne 0 ]; then
  echo "[FATAL] bio python cannot import torch. Exit." | tee -a "${MASTER_LOG}"
  exit 1
fi

STATUS_FILE="${RUN_BASE}/gign_exact_5seeds_status.tsv"
echo -e "seed\tstatus\tout_dir" > "${STATUS_FILE}"

for seed in "${SEEDS[@]}"
do
  OUT_DIR="${RUN_BASE}/gign_exact_smcl_seed${seed}_e200_pat60"
  LOG_FILE="${RUN_BASE}/gign_exact_smcl_seed${seed}_e200_pat60.log"

  echo "====================================================================================================" | tee -a "${MASTER_LOG}"
  echo "[SEED ${seed}] START" | tee -a "${MASTER_LOG}"
  echo "[SEED ${seed}] out_dir=${OUT_DIR}" | tee -a "${MASTER_LOG}"
  date | tee -a "${MASTER_LOG}"

  mkdir -p "${OUT_DIR}"

  "${BIO_PY}" scripts_pdbbind/train_pdbbind_pilot.py \
    --root "${ROOT}" \
    --out_dir "${OUT_DIR}" \
    --cuda "${CUDA_ID}" \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --patience "${PATIENCE}" \
    2>&1 | tee "${LOG_FILE}"

  rc=${PIPESTATUS[0]}

  if [ "${rc}" -eq 0 ]; then
    echo "[SEED ${seed}] DONE" | tee -a "${MASTER_LOG}"
    echo -e "${seed}\tdone\t${OUT_DIR}" >> "${STATUS_FILE}"
  else
    echo "[SEED ${seed}] FAILED with exit code ${rc}" | tee -a "${MASTER_LOG}"
    echo -e "${seed}\tfailed_${rc}\t${OUT_DIR}" >> "${STATUS_FILE}"
    exit "${rc}"
  fi

  date | tee -a "${MASTER_LOG}"
done

echo "====================================================================================================" | tee -a "${MASTER_LOG}"
echo "[SUMMARY] Loading best_model.pt from finished runs" | tee -a "${MASTER_LOG}"

"${BIO_PY}" - <<'PY' 2>&1 | tee -a "${MASTER_LOG}"
import torch
from pathlib import Path
import pandas as pd

RUN_BASE = Path("/data_C/sdb1/lww/pdbbind/runs")
SEEDS = [42, 43, 44, 45, 46]

rows = []
for seed in SEEDS:
    r = RUN_BASE / f"gign_exact_smcl_seed{seed}_e200_pat60"
    ckpt = r / "best_model.pt"
    if not ckpt.exists():
        print(f"[MISSING] seed={seed}, ckpt={ckpt}")
        continue

    obj = torch.load(ckpt, map_location="cpu")
    row = {
        "seed": seed,
        "run": r.name,
        "best_epoch": int(obj["epoch"]),
    }

    for prefix, metrics in [("val", obj["val_metrics"]), ("test2016", obj["core_metrics"])]:
        for k, v in metrics.items():
            row[f"{prefix}_{k}"] = float(v)

    rows.append(row)

df = pd.DataFrame(rows)

out_csv = RUN_BASE / "gign_exact_smcl_5seeds_e200_pat60_summary.csv"
out_txt = RUN_BASE / "gign_exact_smcl_5seeds_e200_pat60_summary.txt"
df.to_csv(out_csv, index=False)

lines = []
lines.append("=" * 100)
lines.append("[PER-SEED RESULTS]")
lines.append(df.to_string(index=False) if len(df) else "No completed runs found.")
lines.append("=" * 100)
lines.append("[MEAN ± STD]")

metrics_to_report = [
    "val_rmse", "val_mae", "val_pearson", "val_spearman",
    "test2016_rmse", "test2016_mae", "test2016_pearson", "test2016_spearman",
]

if len(df):
    for c in metrics_to_report:
        if c in df.columns:
            mean = df[c].mean()
            std = df[c].std(ddof=1) if len(df) > 1 else 0.0
            lines.append(f"{c}: {mean:.4f} ± {std:.4f}")

    lines.append("=" * 100)
    lines.append("[MAIN COMPARISON METRICS]")
    for c in ["test2016_rmse", "test2016_pearson"]:
        mean = df[c].mean()
        std = df[c].std(ddof=1) if len(df) > 1 else 0.0
        lines.append(f"{c}: {mean:.4f} ± {std:.4f}")

lines.append("=" * 100)
lines.append(f"saved_csv: {out_csv}")
lines.append(f"saved_txt: {out_txt}")

text = "\n".join(lines)
print(text)

with open(out_txt, "w") as f:
    f.write(text + "\n")
PY

rc=${PIPESTATUS[0]}
if [ "${rc}" -ne 0 ]; then
  echo "[FATAL] summary failed." | tee -a "${MASTER_LOG}"
  exit "${rc}"
fi

echo "====================================================================================================" | tee -a "${MASTER_LOG}"
echo "[ALL DONE] 5-seed training and summary completed." | tee -a "${MASTER_LOG}"
date | tee -a "${MASTER_LOG}"
echo "Summary CSV: ${RUN_BASE}/gign_exact_smcl_5seeds_summary.csv" | tee -a "${MASTER_LOG}"
echo "Summary TXT: ${RUN_BASE}/gign_exact_smcl_5seeds_summary.txt" | tee -a "${MASTER_LOG}"
echo "====================================================================================================" | tee -a "${MASTER_LOG}"
