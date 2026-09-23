#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
: "${TRAIN_JSON:?Set TRAIN_JSON to proofwriter_all_train.jsonl}"
: "${EVAL_JSON:?Set EVAL_JSON explicitly to the intended dev or test file}"
: "${MODEL:?Set MODEL to the original Qwen model directory}"
EXTRA=()
[[ -n "${RESUME:-}" ]] && EXTRA+=(--resume "$RESUME")
echo "[data] train=$TRAIN_JSON eval=$EVAL_JSON"
torchrun --standalone --nproc_per_node="${NPROC:-4}" \
  -m cond_drift_reason.train_candidate_attraction \
  --train-json "$TRAIN_JSON" --test-json "$EVAL_JSON" \
  --teacher "$MODEL" --backbone "${BACKBONE:-$MODEL}" \
  --max-candidates "${MAX_CANDIDATES:-8}" \
  --query-len "${QUERY_LEN:-320}" --resp-len "${RESP_LEN:-224}" \
  --queries-per-step "${QPS:-8}" --steps "${STEPS:-60000}" --lr "${LR:-2e-4}" \
  --candidate-weight "${CANDIDATE_WEIGHT:-3}" --candidate-temp "${CANDIDATE_TEMP:-0.5}" \
  --repel-intra "${REPEL_INTRA:-0}" --eval-every "${EVAL_EVERY:-1000}" \
  --eval-queries "${EVAL_QUERIES:-1000}" --eval-train-queries "${EVAL_TRAIN_QUERIES:-256}" \
  --save-every "${SAVE_EVERY:-5000}" --fail-on-target-truncation \
  --ckpt-dir "${CKPT_DIR:-$ROOT/runs/proofwriter}" "${EXTRA[@]}" "$@"
