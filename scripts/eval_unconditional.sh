#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LENGTH=${LENGTH:-128}
VARIANT=${VARIANT:-P}
case "$VARIANT" in P|S) ;; *) echo "VARIANT must be P or S" >&2; exit 2 ;; esac
case "$LENGTH" in
  128) ARCH=warmup ;;
  1024) ARCH=unconditional ;;
  *) echo "LENGTH must be 128 or 1024" >&2; exit 2 ;;
esac
EXTRA=()
[[ -n "${MODEL:-}" ]] && EXTRA+=(--teacher "$MODEL")
[[ -n "${TOKENIZER:-}" ]] && EXTRA+=(--tokenizer-name "$TOKENIZER")
[[ -n "${EMBED_MODEL:-}" ]] && EXTRA+=(--embed-model "$EMBED_MODEL")
# P releases use public GPT-2; S keeps the original model from cfg unless overridden.
if [[ "$VARIANT" = P && -z "${MODEL:-}" && -z "${EMBED_MODEL:-}" ]]; then
  EXTRA+=(--teacher gpt2)
fi
python -m tasks.unconditional.eval \
  --ckpt "${CHECKPOINT:-unconditional_${LENGTH}_${VARIANT}.pt}" --architecture "$ARCH" \
  --which "${WHICH:-ema}" --n-samples "${N_SAMPLES:-1024}" \
  --chunk "${BATCH:-32}" --seed "${SEED:-0}" "${EXTRA[@]}" "$@"

