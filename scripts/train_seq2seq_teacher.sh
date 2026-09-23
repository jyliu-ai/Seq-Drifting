#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

HERE="$(cd "$(dirname "$0")" && pwd)"
PKG_ROOT="$(cd "$HERE/.." && pwd)"
cd "$PKG_ROOT"

GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
NPROC=${NPROC:-4}
export CUDA_VISIBLE_DEVICES="$GPUS"

DATASET=${DATASET:-wmt14_de_en}
DATA_DIR=${DATA_DIR:?Set DATA_DIR to the prepared dataset directory}
BASE_MODEL=${BASE_MODEL:-gpt2}
OUTPUT_DIR=${OUTPUT_DIR:-"$PKG_ROOT/teachers/${DATASET}_gpt2"}
TARGET_LEN=${TARGET_LEN:-64}
EPOCHS=${EPOCHS:-3}
MAX_STEPS=${MAX_STEPS:-0}
LR=${LR:-5e-5}
GRAD_ACCUM=${GRAD_ACCUM:-1}

case "$DATASET" in
  wmt14_de_en)
    COND_LEN=${COND_LEN:-64}
    BATCH=${BATCH:-16}
    GRAD_CKPT=${GRAD_CKPT:-0}
    ;;
  xsum)
    COND_LEN=${COND_LEN:-1024}
    BATCH=${BATCH:-2}
    GRAD_CKPT=${GRAD_CKPT:-1}
    ;;
  *) echo "unknown DATASET=$DATASET" >&2; exit 2 ;;
esac

EXTRA=()
[ "$GRAD_CKPT" = "1" ] && EXTRA+=(--gradient-checkpointing)
[ "${BF16:-1}" = "0" ] && EXTRA+=(--no-bf16)
[ "${MAX_TRAIN:-0}" != "0" ] && EXTRA+=(--max-train "$MAX_TRAIN")
[ "${MAX_EVAL:-1000}" != "1000" ] && EXTRA+=(--max-eval "$MAX_EVAL")
[ "${RESUME_FROM:-}" != "" ] && EXTRA+=(--resume-from "$RESUME_FROM")
[ "${SAVE_EVERY:-1000}" != "1000" ] && EXTRA+=(--save-every "$SAVE_EVERY")

echo "[teacher] dataset=$DATASET base=$BASE_MODEL GPUs=$GPUS output=$OUTPUT_DIR"
torchrun --standalone --nproc_per_node="$NPROC" -m cond_drift_seq2seq.train_teacher \
    --dataset "$DATASET" --data-dir "$DATA_DIR" --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_DIR" --condition-len "$COND_LEN" --target-len "$TARGET_LEN" \
    --epochs "$EPOCHS" --max-steps "$MAX_STEPS" --per-device-batch "$BATCH" \
    --grad-accum-steps "$GRAD_ACCUM" --lr "$LR" "${EXTRA[@]}" "$@"
