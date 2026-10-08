# PAPER_TITLE_PLACEHOLDER

[Paper](ARXIV_URL_PLACEHOLDER) · [Checkpoints](https://huggingface.co/jyliuAI/Seq-Drifting)

## Setup

Python 3.10+, PyTorch with CUDA, Linux/Bash. Run commands from the repository root.

```bash
python -m pip install -r requirements.txt
```

Checkpoints download automatically when `CHECKPOINT` is a release filename. To download all weights into the Hugging Face cache:

```bash
bash scripts/download_checkpoints.sh
```

Training uses four GPUs; testing uses one GPU and EMA weights. Set `NPROC`, `CUDA_VISIBLE_DEVICES`, or `WHICH=model` to override.

## 1. Unconditional generation

No dataset is required for training or testing. The 128-token weights use the conditional generator at the end of unconditional warmup, with its constant visible prefix. The 1024 release uses the original unconditional generator; sequence length is read from the checkpoint.

### Length 128

```bash
# Train
LENGTH=128 TEACHER=gpt2 bash scripts/train_unconditional.sh

# Test
LENGTH=128 VARIANT=P bash scripts/eval_unconditional.sh
LENGTH=128 VARIANT=S bash scripts/eval_unconditional.sh
```

### Length 1024

```bash
# Train
LENGTH=1024 TEACHER=gpt2 bash scripts/train_unconditional.sh

# Test
LENGTH=1024 VARIANT=P bash scripts/eval_unconditional.sh
LENGTH=1024 VARIANT=S bash scripts/eval_unconditional.sh
```

Tests select `unconditional_<length>_<variant>.pt`. For S weights, use the original frozen support model; if its saved path has moved, set `MODEL=/path/to/support-model` and, when needed, `TOKENIZER=/path/to/support-model`. Training accepts `TEACHER` for the support model.

## 2. Conditional generation

LM1B JSONL rows: `{"query": "prefix", "response": "continuation"}`. OpenWebText2 uses the original `*.jsonl.zst` shards. Training uses 200k warmup steps followed by 100k conditional steps.

### LM1B: 64 → 64

```bash
# Train
DATASET=lm1b TRAIN_JSON=data/lm1b/train.jsonl TEST_JSON=data/lm1b/test.jsonl \
bash scripts/train_conditional.sh

# Test
DATASET=lm1b TEST_JSON=data/lm1b/test.jsonl VARIANT=P \
bash scripts/eval_conditional.sh

DATASET=lm1b TEST_JSON=data/lm1b/test.jsonl VARIANT=S \
bash scripts/eval_conditional.sh
```

Selects `conditional_64_P.pt` or `conditional_64_S.pt`.

### OpenWebText2: 64 → 512

```bash
# Train
DATASET=owt OWT_DIR=data/openwebtext2 bash scripts/train_conditional.sh

# Test
DATASET=owt OWT_DIR=data/openwebtext2 CHECKPOINT=conditional_512_P.pt \
bash scripts/eval_conditional.sh
```

For a custom support model, set `TEACHER` during training and `MODEL` during testing.

## 3. Translation: WMT14 De → En

```bash
# Prepare data
python -m common.prepare_seq2seq --dataset wmt14_de_en --output-dir data/wmt14_de_en

# Train support model
DATASET=wmt14_de_en DATA_DIR=data/wmt14_de_en OUTPUT_DIR=runs/teacher-wmt \
bash scripts/train_seq2seq_teacher.sh

# Train generator
DATA_DIR=data/wmt14_de_en TEACHER=runs/teacher-wmt/best \
bash scripts/train_translation.sh

# Test
DATA_DIR=data/wmt14_de_en TEACHER=/path/to/original-wmt-support-model CHECKPOINT=WMT.pt \
bash scripts/eval_translation.sh
```

## 4. Summarization: XSum

```bash
# Prepare data
python -m common.prepare_seq2seq --dataset xsum --output-dir data/xsum

# Train support model
DATASET=xsum DATA_DIR=data/xsum OUTPUT_DIR=runs/teacher-xsum \
bash scripts/train_seq2seq_teacher.sh

# Train generator
DATA_DIR=data/xsum TEACHER=runs/teacher-xsum/best \
bash scripts/train_summarization.sh

# Test
DATA_DIR=data/xsum TEACHER=/path/to/original-xsum-support-model CHECKPOINT=XSum.pt \
bash scripts/eval_summarization.sh
```

WMT/XSum tests need the same frozen support model used to train the released generator. Omit `TEACHER` if the checkpoint's saved path is available.

## 5. Mathematical reasoning: GSM8K-Aug → SVAMP / ASDiv

Use prepared JSONL rows such as `{"question": "...", "answer": "... #### 5"}` (`generated` is also accepted) and the checkpoint's original Qwen model.

```bash
# Train
TRAIN_JSON=data/gsm8k_aug_equation_train.jsonl EVAL_JSON=data/gsm8k_aug_equation_test.jsonl \
MODEL=Qwen/Qwen2.5-0.5B STEPS=120000 bash scripts/train_math.sh

# Test: SVAMP
MODEL=Qwen/Qwen2.5-0.5B BACKBONE=Qwen/Qwen2.5-0.5B \
TEST_JSON=data/svamp_test.jsonl CHECKPOINT=math_reasoning.pt \
bash scripts/eval_math.sh

# Test: ASDiv
MODEL=Qwen/Qwen2.5-0.5B BACKBONE=Qwen/Qwen2.5-0.5B \
TEST_JSON=data/asdiv_test.jsonl CHECKPOINT=math_reasoning.pt \
bash scripts/eval_math.sh
```

## 6. Logical reasoning: ProofWriter

```bash
# Train
TRAIN_JSON=data/proofwriter_all_train.jsonl EVAL_JSON=data/proofwriter_all_dev.jsonl \
MODEL=Qwen/Qwen2.5-0.5B bash scripts/train_proofwriter.sh

# Test: local checkpoint (no ProofWriter weights are included in the release)
TEST_JSON=data/proofwriter_all_test.jsonl CHECKPOINT=runs/proofwriter/step_60000.pt \
MODEL=Qwen/Qwen2.5-0.5B BACKBONE=Qwen/Qwen2.5-0.5B bash scripts/eval_proofwriter.sh
```

Math/ProofWriter use five seeded runs. Set `MODEL` to the original frozen model and `BACKBONE` if its backbone path differs.

## Code layout

```text
models/     Shared generator architectures and embedding models
common/     Configs, losses, metrics, checkpoint loading, and utilities
tasks/      Unconditional, continuation, seq2seq, and reasoning train/eval code
scripts/    Per-task training and testing commands
```
