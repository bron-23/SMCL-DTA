#!/usr/bin/env bash
set -eo pipefail

cd /home/lww/learn_project/mydta

export PS1="${PS1:-}"
source /home/lww/anaconda3/etc/profile.d/conda.sh
conda activate bio

TRAIN_PT="/data_C/sdb1/lww/mmatt_training/smcl_processed_surface_masif_train_final/processed_data_mmatt_kinase_train_surface_masif_sanitized.pt"
VAL_PT="/data_C/sdb1/lww/mmatt_training/smcl_processed_surface_masif_val_final/processed_data_mmatt_kinase_val_surface_masif_sanitized.pt"
MODEL_PY="/home/lww/learn_project/mydta/src/model_0428_16_dual.py"
CKPT_ROOT="/home/lww/learn_project/mydta/checkpoints"
OUT_ROOT="/data_C/sdb1/lww/mmatt_training"

mkdir -p "$OUT_ROOT"

echo "================================================================================"
echo "[INFO] Surface-enabled fine-tuning for remaining checkpoints"
echo "[TRAIN_PT] $TRAIN_PT"
echo "[VAL_PT]   $VAL_PT"
echo "[MODEL_PY] $MODEL_PY"
echo "================================================================================"

for EPOCH in 1317 1323 1344; do
  CKPT="$(find "$CKPT_ROOT" -maxdepth 1 -type f -name "epoch-${EPOCH}*.pt" | sort | head -n 1 || true)"

  if [[ -z "$CKPT" ]]; then
    echo "[ERROR] checkpoint for epoch-${EPOCH} not found in $CKPT_ROOT"
    exit 1
  fi

  OUT_DIR="${OUT_ROOT}/finetune_epoch${EPOCH}_surface_full_sanitized_lr1e4"
  LOG="${OUT_ROOT}/finetune_epoch${EPOCH}_surface_full_sanitized_lr1e4.log"

  mkdir -p "$OUT_DIR"

  echo "================================================================================"
  echo "[START] Surface-enabled fine-tuning epoch-${EPOCH}"
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
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --patience 10 \
    --device cuda \
    > "$LOG" 2>&1

  echo "[DONE] Surface-enabled fine-tuning epoch-${EPOCH}"
  echo "[BEST]"
  cat "${OUT_DIR}/best_metrics.json"
  echo ""
done

echo "================================================================================"
echo "[ALL DONE] Remaining surface-enabled fine-tuning finished."
echo "================================================================================"
