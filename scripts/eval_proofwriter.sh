#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EXTRA=()
[[ -n "${MODEL:-}" ]] && EXTRA+=(--teacher "$MODEL")
[[ -n "${BACKBONE:-}" ]] && EXTRA+=(--backbone "$BACKBONE")
python -m tasks.reasoning.eval --ckpt "${CHECKPOINT:-proofwriter.pt}" \
  --test-json "${TEST_JSON:-data/proofwriter_all_test.jsonl}" "${EXTRA[@]}" \
  --which "${WHICH:-ema}" --max-candidates "${MAX_CANDIDATES:-8}" \
  --n "${N:-0}" --repeats "${REPEATS:-5}" --seed "${SEED:-0}" "$@"

