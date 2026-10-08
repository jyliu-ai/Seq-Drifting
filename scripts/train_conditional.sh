#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DATASET=${DATASET:-owt}
case "$DATASET" in
  lm1b)
    : "${TRAIN_JSON:?Set TRAIN_JSON to prepared LM1B training prefix/continuation JSONL}"
    : "${TEST_JSON:?Set TEST_JSON to prepared LM1B held-out prefix/continuation JSONL}"
    DATA_ARGS=(--jsonl-path "$TRAIN_JSON" --test-jsonl-path "$TEST_JSON")
    RESP_LEN=${RESP_LEN:-64}
    QPS=${QPS:-128}
    ;;
  owt)
    : "${OWT_DIR:?Set OWT_DIR to the original OpenWebText2 JSONL-Zstandard shards}"
    DATA_ARGS=(--owt-dir "$OWT_DIR" --owt-docs "${OWT_DOCS:-200000}")
    RESP_LEN=${RESP_LEN:-512}
    QPS=${QPS:-8}
    ;;
  *) echo "DATASET must be lm1b or owt" >&2; exit 2 ;;
esac
torchrun --standalone --nproc_per_node="${NPROC:-4}" -m tasks.continuation.train \
  --dataset "$DATASET" "${DATA_ARGS[@]}" --teacher "${TEACHER:-gpt2}" \
  --query-len "${QUERY_LEN:-64}" --resp-len "$RESP_LEN" \
  --uncond-warmup-steps "${UNCOND_WARMUP:-200000}" \
  --queries-per-step "$QPS" --k-samples "${KSAMP:-4}" \
  --n-pos "${N_POS:-64}" --prob-thresh 0.01 --support thresh --no-repeat-ngram 4 \
  --gold-weight 0 --teacher-weight 1 --attract-temp "${ATTR_TEMP:-0.3}" \
  --repel "${REPEL:-5}" --repel-intra "${REPEL_INTRA:-100}" --repel-whole-batch \
  --sphere-step geodesic --sphere-geo-max 1.5708 \
  --d-model 768 --nhead 12 --ffn-dim 3072 --num-layers 12 --noise-dim 128 \
  --lr "${LR:-2e-4}" --steps "${STEPS:-300000}" \
  --ckpt-dir "${CKPT_DIR:-runs/conditional_$DATASET}" "$@"
