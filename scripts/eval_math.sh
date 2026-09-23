#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
: "${CHECKPOINT:?Set CHECKPOINT}"
: "${TEST_JSON:?Set TEST_JSON to the converted complete SVAMP or MAWPS test JSONL}"
: "${MODEL:?Set MODEL to the original frozen embedding model}"
python -m cond_drift_reason.train_candidate_attraction \
  --eval-only --resume "$CHECKPOINT" --test-json "$TEST_JSON" \
  --teacher "$MODEL" --backbone "${BACKBONE:-$MODEL}" \
  --max-candidates "${MAX_CANDIDATES:-1}" \
  --query-len "${QUERY_LEN:-128}" --resp-len "${RESP_LEN:-48}" \
  --eval-queries "${EVAL_QUERIES:-2147483647}" \
  --eval-repeats "${REPEATS:-5}" --eval-seed "${SEED:-0}" \
  --fail-on-target-truncation --ckpt-dir "${OUTPUT_DIR:-$ROOT/runs/math_eval}" "$@"
