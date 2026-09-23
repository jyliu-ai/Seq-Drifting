#!/bin/bash
set -e
# Conditional continuation on OpenWebText, GPT-2 teacher, 4-GPU DDP.
# Phase 1 (uncond warmup): prefix masked -> reproduce the WORKING unconditional drifting
#   (same settings as the gen_ppl~36 run) to reach the "coherent English" basin.
# Phase 2 (conditional): prefix conditioning turns on; learn to continue the prefix.
#
# Launch: bash run_cond_owt.sh   (uses all 4 visible GPUs)
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PKG_ROOT="$CODE_DIR"
cd "$PKG_ROOT"

NPROC=${NPROC:-2}                     # GPU 数 (= effective batch 放大倍数)

# ════════ 数据 / teacher ════════
DATASET=owt
OWT_DIR=${OWT_DIR:?Set OWT_DIR to the directory containing *.jsonl.zst}
OWT_DOCS=${OWT_DOCS:-200000}
TEACHER=${TEACHER:-gpt2}              # 流形+teacher+评测参照, 和无条件版/FLM 一致
QUERY_LEN=${QUERY_LEN:-64}
RESP_LEN=${RESP_LEN:-128}             # = 无条件版 seq_len

# ════════ warmup ════════
UNCOND_WARMUP=${UNCOND_WARMUP:-200000}   # 前 N 步无条件(mask 前缀), 复现无条件版

# ════════ 批 (每卡 Q*K; 总 = *NPROC). 无条件版每卡 gen_per_step=512 ════════
QPS=${QPS:-128}                       # 每卡 query 数
KSAMP=${KSAMP:-4}                     # 每 query 采样数 -> 每卡 512, ×4卡 = 2048 (= 无条件版)

# ════════ teacher 支撑集 (= 无条件版好配置) ════════
N_POS=${N_POS:-64}
PROB=${PROB:-0.01}
SUPPORT=${SUPPORT:-thresh}
NO_REPEAT=${NO_REPEAT:-4}

# ════════ 吸引/斥力 (= 无条件版好配置: repel5 intra100 perpos1 repel_abs0) ════════
GOLD_WEIGHT=${GOLD_WEIGHT:-0}         # 纯 teacher (续写不用 gold); off-policy 消融: GOLD_WEIGHT=1 TEACHER_W=0
TEACHER_W=${TEACHER_W:-1}
ATTR_TEMP=${ATTR_TEMP:-0.3}
REPEL=${REPEL:-5}
REPEL_INTRA=${REPEL_INTRA:-100}
REPEL_BLOCK=${REPEL_BLOCK:-0}         # 全批斥力 (= 无条件版, 也实测比分块稳)
SPHERE_STEP=geodesic
SPHERE_GEO_MAX=1.5708

# ════════ 生成器 (= 无条件版: d768 h12 ffn3072 L12 noise128) ════════
D_MODEL=${D_MODEL:-768}
NHEAD=${NHEAD:-12}
FFN_DIM=${FFN_DIM:-3072}
N_LAYER=${N_LAYER:-12}
NOISE_DIM=${NOISE_DIM:-128}
LR=${LR:-2e-4}
STEPS=${STEPS:-400000}

REPEL_BLK_FLAG=""; [ "${REPEL_BLOCK}" = "0" ] && REPEL_BLK_FLAG="--repel-whole-batch"
TAG=${TAG:-}
CKPT_DIR=${CKPT_DIR:-"runs/condowt_gpt2_np${NPROC}_q${QPS}k${KSAMP}_r${RESP_LEN}_np${N_POS}_rep${REPEL}i${REPEL_INTRA}t${ATTR_TEMP}_warm${UNCOND_WARMUP}${TAG:+_$TAG}"}

echo "[run] NPROC=${NPROC} eff_batch=$((QPS*KSAMP*NPROC)) warmup=${UNCOND_WARMUP} repel=${REPEL} intra=${REPEL_INTRA} -> ${CKPT_DIR}"

torchrun --standalone --nproc_per_node=${NPROC} -m cond_drift.train \
    --dataset ${DATASET} --owt-dir "${OWT_DIR}" --owt-docs ${OWT_DOCS} --teacher "${TEACHER}" \
    --query-len ${QUERY_LEN} --resp-len ${RESP_LEN} \
    --uncond-warmup-steps ${UNCOND_WARMUP} \
    --queries-per-step ${QPS} --k-samples ${KSAMP} \
    --n-pos ${N_POS} --prob-thresh ${PROB} --support ${SUPPORT} --no-repeat-ngram ${NO_REPEAT} \
    --gold-weight ${GOLD_WEIGHT} --teacher-weight ${TEACHER_W} \
    --attract-temp ${ATTR_TEMP} --repel ${REPEL} --repel-intra ${REPEL_INTRA} ${REPEL_BLK_FLAG} \
    --sphere-step ${SPHERE_STEP} --sphere-geo-max ${SPHERE_GEO_MAX} \
    --d-model ${D_MODEL} --nhead ${NHEAD} --ffn-dim ${FFN_DIM} --num-layers ${N_LAYER} \
    --noise-dim ${NOISE_DIM} --lr ${LR} --steps ${STEPS} \
    --ckpt-dir "${CKPT_DIR}" "$@"
