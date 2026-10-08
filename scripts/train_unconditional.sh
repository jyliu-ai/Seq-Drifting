#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
LENGTH=${LENGTH:-128}
VARIANT=${VARIANT:-P}
case "$VARIANT" in
  P) TEACHER=${TEACHER:-gpt2} ;;
  S) : "${TEACHER:?Set TEACHER to the frozen GPT-2 trained from scratch on the corresponding corpus}" ;;
  *) echo "VARIANT must be P or S" >&2; exit 2 ;;
esac
if [[ "$LENGTH" = 128 ]]; then
  # This is the conditional architecture's last warm-up checkpoint.
  torchrun --standalone --nproc_per_node="${NPROC:-4}" -m tasks.continuation.train \
    --warmup-only --teacher "$TEACHER" --query-len "${QUERY_LEN:-64}" --resp-len 128 \
    --queries-per-step "${QPS:-128}" --k-samples "${KSAMP:-4}" \
    --uncond-warmup-steps "${STEPS:-200000}" --steps "${STEPS:-200000}" \
    --n-pos 64 --prob-thresh 0.01 --support thresh --no-repeat-ngram 4 \
    --gold-weight 0 --teacher-weight 1 --attract-temp "${ATTR_TEMP:-0.3}" \
    --repel "${REPEL:-5}" --repel-intra "${REPEL_INTRA:-100}" --repel-whole-batch \
    --sphere-step geodesic --sphere-geo-max 1.5708 \
    --d-model 768 --nhead 12 --ffn-dim 3072 --num-layers 12 --noise-dim 128 \
    --lr "${LR:-2e-4}" --ckpt-dir "${CKPT_DIR:-runs/unconditional_128_$VARIANT}" "$@"
elif [[ "$LENGTH" = 1024 ]]; then
  torchrun --standalone --nproc_per_node="${NPROC:-4}" -m tasks.unconditional.train \
    --seq-len "${SEQ_LEN:-1023}" --gen-per-step "${GEN_PER_STEP:-32}" \
    --noise-dim 128 --num-layers 12 --lr "${LR:-2e-4}" --steps "${STEPS:-200000}" \
    --temp 1.0 --ema-decay 0.999 --ngram-min 3 --ngram-max 3 \
    --positive-source gpt2 --gpt2-start-step 0 --gpt2-method repair \
    --gpt2-teacher "$TEACHER" --gpt2-repair-target teacher \
    --gpt2-n-pos 64 --gpt2-prob-thresh 0.01 --gpt2-temp "${ATTR_TEMP:-0.3}" \
    --gpt2-repel "${REPEL:-5}" --gpt2-repel-perpos --gpt2-repel-intra "${REPEL_INTRA:-100}" \
    --gpt2-support thresh --gpt2-no-repeat-ngram 4 \
    --sphere-norm --sphere-step geodesic --sphere-geo-max 1.5708 --skip-bank \
    --tokenizer-name "${TOKENIZER:-gpt2}" --embed-model "${EMBED_MODEL:-gpt2}" \
    --ckpt-dir "${CKPT_DIR:-runs/unconditional_1024_$VARIANT}" --bf16 "$@"
else
  echo "LENGTH must be 128 or 1024" >&2; exit 2
fi

