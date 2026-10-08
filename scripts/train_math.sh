#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
: "${TRAIN_JSON:?Set TRAIN_JSON to training question/answer JSONL}"
: "${EVAL_JSON:?Set EVAL_JSON explicitly; no automatic training-set fallback}"
: "${MODEL:?Set MODEL to the original frozen Qwen embedding model}"
: "${STEPS:?Set STEPS to the total ending step, including resumed steps}"
EXTRA=()
[[ -n "${RESUME:-}" ]] && EXTRA+=(--resume "$RESUME")
# Defaults below come from the equation curriculum launcher, not a verified
# final GSM8K-Aug experiment configuration. Override using env or CLI arguments.
torchrun --standalone --nproc_per_node="${NPROC:-4}" \
  -m tasks.reasoning.train \
  --train-json "$TRAIN_JSON" --test-json "$EVAL_JSON" \
  --teacher "$MODEL" --backbone "${BACKBONE:-$MODEL}" \
  --max-candidates "${MAX_CANDIDATES:-1}" \
  --query-len "${QUERY_LEN:-128}" --resp-len "${RESP_LEN:-48}" \
  --queries-per-step "${QPS:-32}" --steps "$STEPS" --lr "${LR:-1e-4}" \
  --candidate-temp "${CANDIDATE_TEMP:-0.15}" --candidate-weight "${CANDIDATE_WEIGHT:-1}" \
  --repel-intra "${REPEL_INTRA:-0}" --eval-every "${EVAL_EVERY:-1000}" \
  --eval-queries "${EVAL_QUERIES:-1000}" --eval-train-queries "${EVAL_TRAIN_QUERIES:-500}" \
  --save-every "${SAVE_EVERY:-5000}" --fail-on-target-truncation \
  --ckpt-dir "${CKPT_DIR:-$ROOT/runs/math}" "${EXTRA[@]}" "$@"
