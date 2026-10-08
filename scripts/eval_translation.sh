#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EXTRA=()
[[ -n "${TEACHER:-}" ]] && EXTRA+=(--teacher "$TEACHER")
python -m tasks.seq2seq.eval --ckpt "${CHECKPOINT:-WMT.pt}" --dataset wmt14_de_en \
  --data-dir "${DATA_DIR:-data/wmt14_de_en}" --split test \
  --which "${WHICH:-ema}" --n "${N:-0}" --seed "${SEED:-0}" "${EXTRA[@]}" "$@"

