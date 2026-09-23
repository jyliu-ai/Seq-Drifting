#!/bin/bash
set -e
# ── Linux (普通单机, 非 SLURM) 版 ──
# 与集群版的区别: 去掉 module load / SLURM 专用项; conda 激活按本机来。

# 代理: 自己机器一般不用; 需要再取消注释并改成你的代理
# export https_proxy=http://USER:PASS@HOST:PORT
# export http_proxy=http://USER:PASS@HOST:PORT
export PYTHONUNBUFFERED=1

# 激活 conda 环境 (按你的安装路径改; 或注释掉本段, 跑前自己 `conda activate py310`)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv

# 选卡 (可选): 例如只用 0,1 两张卡
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# 以"放本脚本的目录"为 Python 包, cd 到父目录, 用目录名做模块名。
# The script resolves its own repository root and does not require SLURM.
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PKG_NAME=uncond_drift
cd "$CODE_DIR"
echo "[run] CODE_DIR=$CODE_DIR  PKG_NAME=$PKG_NAME  CWD=$PWD"

# ── sanity check 超参 ──
# ════════════════ 环境 ════════════════
NPROC=${NPROC:-4}                    # GPU 数 (可 env 覆盖)

# ════════════════ 数据 ════════════════
DATASET=${DATASET:-openwebtext}       # openwebtext | wikitext-2 | wikitext-103 | ptb
SEQ_LEN=${SEQ_LEN:-1023}             # 内容长度(不含 BOS/EOS)(可 env 覆盖)
N_TRAIN=${N_TRAIN:-100000}

# ════════════════ 生成器 / 优化 ════════════════
GEN_PER_STEP=${GEN_PER_STEP:-32}    # 每卡 batch;len128 用 512 或 1024 (可 env 覆盖)
NOISE_DIM=128
N_LAYER=12
LR=${LR:-2e-4}
STEPS=${STEPS:-200000}
TEMP=1.0                             # z 采样温度
EMA_DECAY=0.999
EVAL_EVERY=${EVAL_EVERY:-1000}       # 每多少步 eval 一次 (扫参可调稀,如 10000)(可 env 覆盖)
EVAL_SAMPLES=${EVAL_SAMPLES:-1024}   # 每次 eval 生成多少条 (可 env 覆盖)

# ════════════════ 初始化 ════════════════
INIT_FROM=""                         # 球面模式从头训练(L2 的 3-gram checkpoint 与球面几何不兼容)。旧: .../pre_gpt2_step149999.pt

# ════════════════ n-gram curriculum ════════════════
NGRAM_MIN=3                          # 从 checkpoint 起 → 3
NGRAM_MAX=3
NGRAM_GROW_STEPS=""                  # 100000,150000 | 从 checkpoint 起 → ""

# ════════════════ 正样本来源 ════════════════
POS_SOURCE=gpt2                      # dataset | gpt2

# ──────── (A) 数据集 drift (gpt2 切换前 / positive-source=dataset) ────────
LOSS_MODE=drift                      # drift | match
SELECT_MODE=token_2gram_match        # token_2gram_match | repr_min2gram
TOKEN_MATCH_START=0
N_POS=32                             # 每条 gen top-k,再并集
N_NEG=0
ATTR_TEMP=0.1                        # affinity 温度
SINKHORN=10                          # Sinkhorn 迭代
ANNEAL_STEPS=1000000                 # alpha 退火 (repr 选择用)
MATCH_N_REAL=4096                    # 只 loss-mode=match 用

# ──────── (B) GPT-2 蒸馏 (positive-source=gpt2) ────────
GPT2_START=0                         # 此步后切 GPT-2;从 checkpoint 起 → 0 | 150000
GPT2_METHOD=repair                   # repair | continuation
GPT2_TEACHER=${GPT2_TEACHER:-gpt2}   # teacher 模型：hub 名 "gpt2" 或本地 HF 目录路径 (可 env 覆盖)
GPT2_POS=${GPT2_POS:-64}             # 支撑集大小 k：每位 top-k 合理 token (可 env 覆盖)
GPT2_PROB=0.01                       # [repair] prob<=此值算不合理
GPT2_TEMP=${GPT2_TEMP:-0.3}          # [repair] attraction 温度 τ (调小→更硬 mode-seeking)(可 env 覆盖)
GPT2_REPEL=${GPT2_REPEL:-5}          # [repair] 整条 gen-gen 排斥权重 λ (0=纯吸引)(可 env 覆盖)
GPT2_TOPK=5000                       # [continuation] top-k 合理集
GPT2_REPAIR_TARGET=teacher           # [repair] self(旧:修不合理+snap自己) | teacher(A2/reverse-KL:每位置朝最近GPT-2支撑token)
GPT2_REPEL_ABS=0                     # [repair] 1=绝对尺度排斥(B,退化点也抗坍) | 0=旧相对吸引尺度
GPT2_REPEL_PERPOS=1                  # 1=逐位置排斥 | 0=整条(旧)
GPT2_REPEL_INTRA=${GPT2_REPEL_INTRA:-100}   # 句内 token 排斥权重 (可 env 覆盖)
REPEL_EXTRA=${REPEL_EXTRA:-0}         # 每步额外生成的 no-grad 样本数,只进排斥不进 loss (0=关)(可 env 覆盖)
REPEL_QUEUE=${REPEL_QUEUE:-0}         # MoCo 队列长度,每步只刷新 REPEL_EXTRA 个 (0=只用当步新采的)(可 env 覆盖)
REPEL_CHUNK=${REPEL_CHUNK:-0}         # 额外样本每次生成多少条 (显存峰值);0=同 GEN_PER_STEP (可 env 覆盖)
GRAD_ACCUM=${GRAD_ACCUM:-1}           # 梯度累积步数;梯度 batch = GRAD_ACCUM x GEN_PER_STEP x NPROC (可 env 覆盖)
REPEL_REUSE=${REPEL_REUSE:-0}         # 1=排斥集直接复用梯度 batch (零额外前向,滞后<=1 步)(可 env 覆盖)
GPT2_SUPPORT=thresh         # thresh | nucleus
GPT2_NUCLEUS_P=0.99
NO_REPEAT=4                          # no-repeat-ngram: k>=2 禁止补全已出现的 k-gram(断句内重复);0=关
NO_REPEAT_WINDOW=0                   # 只回看这么多 token,0=整段前缀
GPT2_TEMP_LIST="" # 空="" 则用单个 GPT2_TEMP
SKIP_BANK=1
SKIP_BANK_FLAG=""; [ "${SKIP_BANK}" = "1" ] && SKIP_BANK_FLAG="--skip-bank"
# 诊断(b): teacher support 用真实前缀而非模型自生成前缀 (可 env 覆盖)
TEACHER_REAL_PREFIX=${TEACHER_REAL_PREFIX:-0}
TRP_FLAG=""; [ "${TEACHER_REAL_PREFIX}" = "1" ] && TRP_FLAG="--teacher-real-prefix"
REUSE_FLAG=""; [ "${REPEL_REUSE}" = "1" ] && REUSE_FLAG="--repel-reuse"
PERPOS_FLAG=""; [ "${GPT2_REPEL_PERPOS}" = "1" ] && PERPOS_FLAG="--gpt2-repel-perpos"

# repel-abs 是 store_true 开关: 1 才加 flag
REPEL_ABS_FLAG=""; [ "${GPT2_REPEL_ABS}" = "1" ] && REPEL_ABS_FLAG="--gpt2-repel-abs"

# ──────── 球面归一化 (sphere) ────────
SPHERE_NORM=1                        # 1=wte + 生成器输出都归一化到单位球(角度/余弦几何,排斥不再撑大范数)
SPHERE_STEP=geodesic                 # proj(弦+投影) | retract(切向+投影) | geodesic(测地线 exp)
SPHERE_GEO_MAX=1.5708                # 仅 geodesic:每步最大弧长 (rad, ~pi/2)
SPHERE_FLAG=""; [ "${SPHERE_NORM}" = "1" ] && SPHERE_FLAG="--sphere-norm"

# ════════════════ 输出 ════════════════
CKPT_DIR=${CKPT_DIR:-"runs/unconditional_${DATASET}_seq${SEQ_LEN}_n${N_TRAIN}_g${GEN_PER_STEP}_L${N_LAYER}_lr${LR}_np${NPROC}_${POS_SOURCE}_${GPT2_METHOD}_${GPT2_REPAIR_TARGET}_${GPT2_REPEL}_${GPT2_REPEL_PERPOS}_${GPT2_REPEL_INTRA}_sph${SPHERE_NORM}${SPHERE_STEP}_${GPT2_SUPPORT}_nr${NO_REPEAT}_trp${TEACHER_REAL_PREFIX}_rx${REPEL_EXTRA}q${REPEL_QUEUE}_ga${GRAD_ACCUM}ru${REPEL_REUSE}"}   # default can be overridden

torchrun --standalone --nproc_per_node=${NPROC} -m "${PKG_NAME}.train" \
    --dataset ${DATASET} --seq-len ${SEQ_LEN} --n-train ${N_TRAIN} \
    --gen-per-step ${GEN_PER_STEP} --noise-dim ${NOISE_DIM} --num-layers ${N_LAYER} \
    --lr ${LR} --steps ${STEPS} --temp ${TEMP} --ema-decay ${EMA_DECAY} \
    --eval-every ${EVAL_EVERY} --eval-samples ${EVAL_SAMPLES} \
    --init-from "${INIT_FROM}" \
    --ngram-min ${NGRAM_MIN} --ngram-max ${NGRAM_MAX} --ngram-grow-steps "${NGRAM_GROW_STEPS}" \
    --positive-source ${POS_SOURCE} \
    --loss-mode ${LOSS_MODE} --select-mode ${SELECT_MODE} --token-match-start ${TOKEN_MATCH_START} \
    --n-pos ${N_POS} --n-neg ${N_NEG} --attract-temp ${ATTR_TEMP} --sinkhorn-iters ${SINKHORN} \
    --select-anneal-steps ${ANNEAL_STEPS} --match-n-real ${MATCH_N_REAL} --per-position-force \
    --gpt2-start-step ${GPT2_START} --gpt2-method ${GPT2_METHOD} --gpt2-teacher "${GPT2_TEACHER}" \
    --gpt2-support ${GPT2_SUPPORT} --gpt2-nucleus-p ${GPT2_NUCLEUS_P} \
    --gpt2-temp-list "${GPT2_TEMP_LIST}" \
    --gpt2-n-pos ${GPT2_POS} --gpt2-prob-thresh ${GPT2_PROB} --gpt2-temp ${GPT2_TEMP} \
    --gpt2-repel ${GPT2_REPEL} --gpt2-topk ${GPT2_TOPK} \
    --gpt2-repel-intra ${GPT2_REPEL_INTRA} \
    --repel-extra ${REPEL_EXTRA} --repel-queue ${REPEL_QUEUE} --repel-chunk ${REPEL_CHUNK} \
    --grad-accum ${GRAD_ACCUM} ${REUSE_FLAG} \
    --gpt2-repair-target ${GPT2_REPAIR_TARGET} ${REPEL_ABS_FLAG} ${PERPOS_FLAG} \
    ${SPHERE_FLAG} --sphere-step ${SPHERE_STEP} --sphere-geo-max ${SPHERE_GEO_MAX} ${SKIP_BANK_FLAG} ${TRP_FLAG} \
    --gpt2-no-repeat-ngram ${NO_REPEAT} --gpt2-no-repeat-window ${NO_REPEAT_WINDOW} \
    --tokenizer-name "${TOKENIZER:-gpt2}" --embed-model "${EMBED_MODEL:-gpt2}" \
    --data-path "${DATA_PATH:-}" --ckpt-dir "${CKPT_DIR}" --bf16 "$@"
