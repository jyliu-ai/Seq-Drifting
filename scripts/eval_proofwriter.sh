#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
: "${CHECKPOINT:?Set CHECKPOINT}"
: "${TEST_JSON:?Set TEST_JSON to proofwriter_all_test.jsonl}"
: "${MODEL:?Set MODEL to the same frozen embedding model used for training}"
python -m cond_drift_reason.train_candidate_attraction \
  --eval-only --resume "$CHECKPOINT" --test-json "$TEST_JSON" \
  --teacher "$MODEL" --backbone "${BACKBONE:-$MODEL}" \
  --max-candidates "${MAX_CANDIDATES:-8}" \
  --query-len "${QUERY_LEN:-320}" --resp-len "${RESP_LEN:-224}" \
  --eval-queries "${EVAL_QUERIES:-2147483647}" \
  --eval-repeats "${REPEATS:-5}" --eval-seed "${SEED:-0}" \
  --fail-on-target-truncation --ckpt-dir "${OUTPUT_DIR:-$ROOT/runs/proofwriter_eval}" "$@"
