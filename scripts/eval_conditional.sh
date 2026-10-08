#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATASET=${DATASET:-owt}
VARIANT=${VARIANT:-P}
case "$VARIANT" in P|S) ;; *) echo "VARIANT must be P or S" >&2; exit 2 ;; esac
case "$DATASET" in
  lm1b)
    DATA_ARGS=(--test-jsonl-path "${TEST_JSON:-data/lm1b/test.jsonl}")
    FILE="conditional_64_${VARIANT}.pt"
    ;;
  owt)
    DATA_ARGS=(--owt-dir "${OWT_DIR:-data/openwebtext2}")
    FILE="conditional_512_${VARIANT}.pt"
    ;;
  *) echo "DATASET must be lm1b or owt" >&2; exit 2 ;;
esac
EXTRA=()
[[ -n "${MODEL:-}" ]] && EXTRA+=(--teacher "$MODEL")
[[ "$VARIANT" = P && -z "${MODEL:-}" ]] && EXTRA+=(--teacher gpt2)
python -m tasks.continuation.eval \
  --ckpt "${CHECKPOINT:-$FILE}" \
  --dataset "$DATASET" "${DATA_ARGS[@]}" "${EXTRA[@]}" \
  --which "${WHICH:-ema}" --n "${N:-0}" --seed "${SEED:-0}" \
  --eval-ppl-model "${PPL_MODEL:-gpt2-large}" "$@"

