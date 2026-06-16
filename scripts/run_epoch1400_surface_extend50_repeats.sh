#!/usr/bin/env bash
set -eo pipefail

cd /home/lww/learn_project/mydta

export PS1="${PS1:-}"
source /home/lww/anaconda3/etc/profile.d/conda.sh
conda activate bio

# ========= fixed inputs =========
TRAIN_PT="/data_C/sdb1/lww/mmatt_training/smcl_processed_surface_masif_train_final/processed_data_mmatt_kinase_train_surface_masif_sanitized.pt"
VAL_PT="/data_C/sdb1/lww/mmatt_training/smcl_processed_surface_masif_val_final/processed_data_mmatt_kinase_val_surface_masif_sanitized.pt"

MODEL_PY="/home/lww/learn_project/mydta/src/model_0428_16_dual.py"

# 从 50 epoch surface 模型继续 extend，重复 5 次
INIT_CKPT="/data_C/sdb1/lww/mmatt_training/finetune_epoch1400_surface_full_sanitized_lr1e4/best_finetuned_model.pt"

S1_PT="/data_C/sdb1/lww/mmatt_s1_surface_overlap/processed_data_mmatt_s1_kinase_surface_masif_overlap_sanitized.pt"
S1_ROWS="/data_C/sdb1/lww/mmatt_s1_surface_overlap/mmatt_s1_kinase_surface_masif_overlap_rows.csv"

ALIGNED_RAW="/data_C/sdb1/lww/mmatt_independent_tests_raw_aligned/kinase_independent_all_scenarios_aligned_raw.csv"

C_BASE="/data_C/sdb1/lww/mmatt_independent_tests_raw_aligned/step9_c_full_surface"
C_PT="${C_BASE}/c_full_surface_masif/processed_data_C_full_surface_masif_sanitized.pt"
C_ROWS=$(ls "${C_BASE}/c_full_surface_masif"/*surface_masif*_rows.csv | head -n 1)

ROOT="/data_C/sdb1/lww/mmatt_training/repeats_epoch1400_surface_extend50_lr5e5"
mkdir -p "$ROOT"

echo "================================================================================"
echo "[INFO] Repeated surface-enabled extend fine-tuning"
echo "[ROOT] $ROOT"
echo "[INIT_CKPT] $INIT_CKPT"
echo "================================================================================"

for SEED in 1 2 3 4 5; do
  RUN_DIR="${ROOT}/seed_${SEED}"
  FT_DIR="${RUN_DIR}/finetune"
  S1_OUT="${RUN_DIR}/s1_inference"
  SCEN_OUT="${RUN_DIR}/scenario_scores"
  C_OUT="${RUN_DIR}/c_full_inference"

  mkdir -p "$RUN_DIR" "$FT_DIR" "$S1_OUT" "$SCEN_OUT" "$C_OUT"

  echo "================================================================================"
  echo "[START] seed=${SEED}"
  echo "================================================================================"

  # 1) fine-tune extend
  python -u scripts/run_seeded_step7f.py \
    --seed "$SEED" \
    --target_script scripts/step7f_finetune_mmatt_from_kiba.py \
    -- \
    --train_pt "$TRAIN_PT" \
    --val_pt "$VAL_PT" \
    --model_py "$MODEL_PY" \
    --checkpoint "$INIT_CKPT" \
    --out_dir "$FT_DIR" \
    --epochs 50 \
    --batch_size 64 \
    --lr 5e-5 \
    --weight_decay 1e-5 \
    --patience 15 \
    --device cuda \
    > "${RUN_DIR}/finetune.log" 2>&1

  echo "[DONE] fine-tune seed=${SEED}"

  CKPT="${FT_DIR}/best_finetuned_model.pt"

  # 2) S1 inference
  python scripts/step8a_infer_finetuned_on_external_pt.py \
    --pt "$S1_PT" \
    --rows_csv "$S1_ROWS" \
    --model_py "$MODEL_PY" \
    --checkpoint "$CKPT" \
    --out_dir "$S1_OUT" \
    --batch_size 128 \
    --device cuda \
    > "${RUN_DIR}/s1_inference.log" 2>&1

  echo "[DONE] S1 inference seed=${SEED}"

  # 3) A/B scenario scoring
  python scripts/step8c_score_independent_scenarios_from_s1_predictions.py \
    --predictions_csv "${S1_OUT}/predictions.csv" \
    --aligned_raw_csv "$ALIGNED_RAW" \
    --out_dir "$SCEN_OUT" \
    > "${RUN_DIR}/scenario_scoring.log" 2>&1

  echo "[DONE] scenario scoring seed=${SEED}"

  # 4) C full 607 inference
  python scripts/step8a_infer_finetuned_on_external_pt.py \
    --pt "$C_PT" \
    --rows_csv "$C_ROWS" \
    --model_py "$MODEL_PY" \
    --checkpoint "$CKPT" \
    --out_dir "$C_OUT" \
    --batch_size 128 \
    --device cuda \
    > "${RUN_DIR}/c_full_inference.log" 2>&1

  echo "[DONE] C full inference seed=${SEED}"

  echo "[BEST METRICS]"
  cat "${FT_DIR}/best_metrics.json"
  echo ""
done

echo "================================================================================"
echo "[ALL DONE] Repeats finished."
echo "================================================================================"
