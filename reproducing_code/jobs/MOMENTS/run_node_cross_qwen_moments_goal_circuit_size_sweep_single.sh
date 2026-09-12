#!/bin/bash
set -euo pipefail

LABEL="$1"
MODEL_NAME="$2"
MODEL_PATH="$3"
CF_MODE="$4"
DATA_FILE="$5"
SCORES_DIR="$6"

cd /home/eboccaletti/reproducing_code/vlm-circuits-analysis

SWEEP_ROOT="${SCORES_DIR}/cross_modality/circuit_size_sweep"
PAIRS=(
  "0.10 0.10 l010_vl010"
  "0.20 0.30 l020_vl030"
  "0.30 0.50 l030_vl050"
  "0.50 0.70 l050_vl070"
)

echo "=== Cross-modal circuit-size sweep: ${LABEL} ==="
echo "=== Model: ${MODEL_NAME} ==="
echo "=== Counterfactual mode: ${CF_MODE} ==="
echo "=== Dataset: ${DATA_FILE} ==="
echo "=== Scores: ${SCORES_DIR} ==="
echo "Host: $(hostname)"
echo "Start time: $(date)"
nvidia-smi || true

for pair in "${PAIRS[@]}"; do
  read -r L_PCT VL_PCT TAG <<< "${pair}"
  RESULT_FILE="${SWEEP_ROOT}/${TAG}/faithfulness_nodes_cross_interchanges_LD_${SLURM_JOB_ID}.pt"
  mkdir -p "$(dirname "${RESULT_FILE}")"

  echo "Running ${LABEL}: L=${L_PCT}, VL=${VL_PCT}"
  python script_node_cross_modality_analysis.py \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --task_name moments_goal \
    --moments_cf_mode "${CF_MODE}" \
    --moments_data_file "${DATA_FILE}" \
    --scores_dir "${SCORES_DIR}" \
    --output_file "${RESULT_FILE}" \
    --l_circuit_percentage "${L_PCT}" \
    --vl_circuit_percentage "${VL_PCT}"

  echo "Saved result: ${RESULT_FILE}"
done

echo "End time: $(date)"
