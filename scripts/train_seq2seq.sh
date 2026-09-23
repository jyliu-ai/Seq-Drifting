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
case "$DATASET" in
  wmt14_de_en)
    COND_LEN=${COND_LEN:-64}
    QPS=${QPS:-8}
    EVAL_QUERIES=${EVAL_QUERIES:-5000}
    EVAL_BS=${EVAL_BS:-16}
    GRAD_CKPT=${GRAD_CKPT:-0}
    ;;
  xsum)
    COND_LEN=${COND_LEN:-1024}
    QPS=${QPS:-1}
    EVAL_QUERIES=${EVAL_QUERIES:-100}
    EVAL_BS=${EVAL_BS:-2}
    GRAD_CKPT=${GRAD_CKPT:-1}
    ;;
  *) echo "unknown DATASET=$DATASET" >&2; exit 2 ;;
esac

DATA_DIR=${DATA_DIR:?Set DATA_DIR to the prepared dataset directory}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
EVAL_SPLIT=${EVAL_SPLIT:-validation}
TARGET_LEN=${TARGET_LEN:-64}
TEACHER=${TEACHER:?Set TEACHER to the task-finetuned GPT-2 model}

# ELF uses a global batch of 512 and 100 epochs for these tasks. On four GPUs this
# is reproduced with gradient accumulation; use EPOCHS/STEPS to shorten pilot runs.
GLOBAL_BATCH=${GLOBAL_BATCH:-512}
GRAD_ACCUM=${GRAD_ACCUM:-0}
EPOCHS=${EPOCHS:-100}
STEPS=${STEPS:-0}
KSAMP=${KSAMP:-1}
N_TRAIN=${N_TRAIN:-0}
N_EVAL=${N_EVAL:-0}

# Ground-truth-only warmup, then the task-fine-tuned GPT-2 teacher is ramped in.
GOLD_WEIGHT=${GOLD_WEIGHT:-3}
TEACHER_W=${TEACHER_W:-1}
GOLD_WARMUP=${GOLD_WARMUP:-10000}
GOLD_WARMUP_RAMP=${GOLD_WARMUP_RAMP:-10000}
GOLD_DECAY_STEPS=${GOLD_DECAY_STEPS:-0}
GOLD_MIN_RATIO=${GOLD_MIN_RATIO:-1.0}
REPEL=${REPEL:-0}
REPEL_INTRA=${REPEL_INTRA:-0}
POS_TEACHER_DECAY=${POS_TEACHER_DECAY:-1.0}
POS_GOLD_BOOST=${POS_GOLD_BOOST:-1.0}
EOS_REPEL=${EOS_REPEL:-0.0}
EOS_REPEL_TAIL=${EOS_REPEL_TAIL:-0}
# Stop training a response where the teacher first calls the student's own token
# implausible. none | prefix (nothing past the flag) | first (only the flagged position)
# | teacher (keep gold everywhere, drop only the teacher term past the flag).
# Requires TEACHER_W>0: the flags come from the teacher forward, which is skipped entirely
# when the teacher term is off.
DIV_MASK=${DIV_MASK:-none}
N_POS=${N_POS:-32}
NO_REPEAT_NGRAM=${NO_REPEAT_NGRAM:-0}
NO_REPEAT_WINDOW=${NO_REPEAT_WINDOW:-0}
LS_INIT=${LS_INIT:-0.1}

# Discriminative CE over cosine logits. 0 = off (drift only, the original objective).
CE_WEIGHT=${CE_WEIGHT:-0}
CE_TAU=${CE_TAU:-0.07}
CE_RAMP=${CE_RAMP:-0}

LR=${LR:-1e-4}
# lr schedule: cosine decays from LR to LR*LR_MIN_RATIO over LR_DECAY_STEPS (0 = full
# run). This is the primary fix for runs that overshoot and never settle: without decay
# the model keeps moving at full LR through every checkpoint, so SWA and EMA are
# averaging points spread across a wide trajectory rather than clustering near an optimum.
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
LR_WARMUP=${LR_WARMUP:-1000}
LR_MIN_RATIO=${LR_MIN_RATIO:-0.1}
LR_DECAY_STEPS=${LR_DECAY_STEPS:-0}
# EMA decay: 0.9999 (~7500-step horizon) made every resumed run look like it peaked
# ~8k steps in and then decayed -- the peak was the average of old and new weights.
# 0.999 (~1k-step horizon) keeps eval close to the live model.
EMA_DECAY=${EMA_DECAY:-0.999}
EVAL_EVERY=${EVAL_EVERY:-2000}
SAVE_EVERY=${SAVE_EVERY:-2000}
TAG=${TAG:-}
CKPT_DIR=${CKPT_DIR:-"runs/${DATASET}_q${QPS}k${KSAMP}_gb${GLOBAL_BATCH}_g${GOLD_WEIGHT}_tw${TEACHER_W}${TAG:+_$TAG}"}

EXTRA=()
[ "$GRAD_ACCUM" != "0" ] && EXTRA+=(--grad-accum-steps "$GRAD_ACCUM")
[ "$GRAD_CKPT" = "1" ] && EXTRA+=(--gradient-checkpointing)
[ "${MASK_ILLEGAL:-1}" = "0" ] && EXTRA+=(--no-mask-illegal)
[ -n "${RESUME:-}" ] && EXTRA+=(--resume "$RESUME")
[ "$N_TRAIN" != "0" ] && EXTRA+=(--n-train "$N_TRAIN")
[ "$N_EVAL" != "0" ] && EXTRA+=(--n-eval "$N_EVAL")

echo "[run] dataset=$DATASET GPUs=$GPUS NPROC=$NPROC cond=$COND_LEN target=$TARGET_LEN"
echo "[run] micro/rank=$QPS global_batch=$GLOBAL_BATCH epochs=$EPOCHS steps=$STEPS -> $CKPT_DIR"

torchrun --standalone --nproc_per_node="$NPROC" -m cond_drift_seq2seq.train \
    --ls-init "$LS_INIT" \
    --dataset "$DATASET" --data-dir "$DATA_DIR" \
    --train-split "$TRAIN_SPLIT" --eval-split "$EVAL_SPLIT" \
    --condition-len "$COND_LEN" --target-len "$TARGET_LEN" \
    --teacher "$TEACHER" --queries-per-step "$QPS" --k-samples "$KSAMP" \
    --global-batch-size "$GLOBAL_BATCH" --epochs "$EPOCHS" --steps "$STEPS" \
    --gold-weight "$GOLD_WEIGHT" --teacher-weight "$TEACHER_W" \
    --gold-warmup-steps "$GOLD_WARMUP" --gold-warmup-ramp "$GOLD_WARMUP_RAMP" \
    --gold-decay-steps "$GOLD_DECAY_STEPS" --gold-min-ratio "$GOLD_MIN_RATIO" \
    --ce-weight "$CE_WEIGHT" --ce-tau "$CE_TAU" --ce-ramp "$CE_RAMP" \
    --repel "$REPEL" --repel-intra "$REPEL_INTRA" --n-pos "$N_POS" \
    --pos-teacher-decay "$POS_TEACHER_DECAY" --pos-gold-boost "$POS_GOLD_BOOST" \
    --eos-repel "$EOS_REPEL" --eos-repel-tail "$EOS_REPEL_TAIL" \
    --div-mask "$DIV_MASK" \
    --no-repeat-ngram "$NO_REPEAT_NGRAM" --no-repeat-window "$NO_REPEAT_WINDOW" \
    --lr "$LR" --lr-schedule "$LR_SCHEDULE" --lr-warmup "$LR_WARMUP" \
    --lr-min-ratio "$LR_MIN_RATIO" --lr-decay-steps "$LR_DECAY_STEPS" \
    --ema-decay "$EMA_DECAY" \
    --eval-every "$EVAL_EVERY" --eval-queries "$EVAL_QUERIES" \
    --eval-batch-size "$EVAL_BS" --save-every "$SAVE_EVERY" \
    --ckpt-dir "$CKPT_DIR" "${EXTRA[@]}" "$@"
