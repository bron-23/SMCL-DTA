#!/usr/bin/env bash
set -eo pipefail

cd /home/lww/learn_project/mydta

source /home/lww/anaconda3/etc/profile.d/conda.sh
conda activate bio

TRAIN_PT="/data_C/sdb1/lww/mmatt_training/smcl_processed_surface_masif_train_final/processed_data_mmatt_kinase_train_surface_masif.pt"
VAL_PT="/data_C/sdb1/lww/mmatt_training/smcl_processed_surface_masif_val_final/processed_data_mmatt_kinase_val_surface_masif.pt"
MODEL_PY="/home/lww/learn_project/mydta/src/model_0428_16_dual.py"
OUT_ROOT="/data_C/sdb1/lww/mmatt_training"

for EPOCH in 1317 1323 1344; do
  CKPT="${OUT_ROOT}/finetune_epoch${EPOCH}_full/best_finetuned_model.pt"

  if [[ ! -f "$CKPT" ]]; then
    echo "[ERROR] Missing checkpoint: $CKPT"
    exit 1
  fi

  OUT_DIR="${OUT_ROOT}/finetune_epoch${EPOCH}_extend70"
  LOG="${OUT_ROOT}/finetune_epoch${EPOCH}_extend70.log"

  mkdir -p "$OUT_DIR"

  echo "================================================================================"
  echo "[START] Extend fine-tuning epoch-${EPOCH}"
  echo "[CKPT] $CKPT"
  echo "[OUT]  $OUT_DIR"
  echo "[LOG]  $LOG"
  echo "================================================================================"

  python -u scripts/step7f_finetune_mmatt_from_kiba.py \
    --train_pt "$TRAIN_PT" \
    --val_pt "$VAL_PT" \
    --model_py "$MODEL_PY" \
    --checkpoint "$CKPT" \
    --out_dir "$OUT_DIR" \
    --epochs 70 \
    --batch_size 128 \
    --lr 5e-5 \
    --weight_decay 1e-5 \
    --patience 15 \
    --device cuda \
    > "$LOG" 2>&1

  echo "[DONE] Extend fine-tuning epoch-${EPOCH}"
  echo "[BEST]"
  cat "${OUT_DIR}/best_metrics.json"
  echo ""
done

echo "================================================================================"
echo "[ALL DONE] Remaining extend70 fine-tuning finished."
echo "================================================================================"
