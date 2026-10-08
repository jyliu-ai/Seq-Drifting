# One-Step Text Generation by Seq-Drifting

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

**P** uses public pretrained GPT-2 (`TEACHER=gpt2`). **S** uses a frozen GPT-2 trained from scratch on the corresponding corpus (`TEACHER=/path/to/scratch-gpt2`). `TEACHER` accepts a Hugging Face model ID or a local directory containing the model and tokenizer; it is not a Seq-Drifting `.pt` file.

The 128-token weights use the conditional generator at the end of unconditional warmup, with its constant visible prefix. The 1024 release uses the original unconditional generator; sequence length is read from the checkpoint.

### Length 128

```bash
# Train P
LENGTH=128 VARIANT=P TEACHER=gpt2 bash scripts/train_unconditional.sh

# Train S: use the GPT-2 trained from scratch on LM1B
LENGTH=128 VARIANT=S TEACHER=/path/to/lm1b-scratch-gpt2 \
bash scripts/train_unconditional.sh

# Test
LENGTH=128 VARIANT=P bash scripts/eval_unconditional.sh
LENGTH=128 VARIANT=S MODEL=/path/to/lm1b-scratch-gpt2 bash scripts/eval_unconditional.sh
```

### Length 1024

```bash
# Train P
LENGTH=1024 VARIANT=P TEACHER=gpt2 bash scripts/train_unconditional.sh

# Train S: use the GPT-2 trained from scratch on OpenWebText
LENGTH=1024 VARIANT=S TEACHER=/path/to/owt-scratch-gpt2 \
bash scripts/train_unconditional.sh

# Test
LENGTH=1024 VARIANT=P bash scripts/eval_unconditional.sh
LENGTH=1024 VARIANT=S bash scripts/eval_unconditional.sh
```

Tests select `unconditional_<length>_<variant>.pt`. For length 128, `MODEL` relocates the frozen model used for token embeddings. For length 1024, the embedding model is separate from `TEACHER`: training defaults to `EMBED_MODEL=gpt2 TOKENIZER=gpt2`, and testing reads these values from the checkpoint. Override `EMBED_MODEL` and `TOKENIZER` only with the original models if their saved paths have moved.

## 2. Conditional generation

LM1B JSONL rows: `{"query": "prefix", "response": "continuation"}`. OpenWebText2 uses the original `*.jsonl.zst` shards. Training uses 200k warmup steps followed by 100k conditional steps.

P/S have the same meaning as above. Set `TEACHER` for training and `MODEL` for testing to the same frozen GPT-2; use the LM1B support model for LM1B and the OpenWebText support model for OpenWebText2.

### LM1B: 64 → 64

```bash
# Train P
DATASET=lm1b VARIANT=P TEACHER=gpt2 \
TRAIN_JSON=data/lm1b/train.jsonl TEST_JSON=data/lm1b/test.jsonl \
bash scripts/train_conditional.sh

# Train S
DATASET=lm1b VARIANT=S TEACHER=/path/to/lm1b-scratch-gpt2 \
TRAIN_JSON=data/lm1b/train.jsonl TEST_JSON=data/lm1b/test.jsonl \
bash scripts/train_conditional.sh

# Test
DATASET=lm1b TEST_JSON=data/lm1b/test.jsonl VARIANT=P \
bash scripts/eval_conditional.sh

DATASET=lm1b TEST_JSON=data/lm1b/test.jsonl VARIANT=S MODEL=/path/to/lm1b-scratch-gpt2 \
bash scripts/eval_conditional.sh
```

Selects `conditional_64_P.pt` or `conditional_64_S.pt`.

### OpenWebText2: 64 → 512

```bash
# Train P
DATASET=owt VARIANT=P TEACHER=gpt2 OWT_DIR=data/openwebtext2 bash scripts/train_conditional.sh

# Train S
DATASET=owt VARIANT=S TEACHER=/path/to/owt-scratch-gpt2 OWT_DIR=data/openwebtext2 \
bash scripts/train_conditional.sh

# Test
DATASET=owt OWT_DIR=data/openwebtext2 CHECKPOINT=conditional_512_P.pt \
bash scripts/eval_conditional.sh
```

The OpenWebText2 example evaluates `conditional_512_P.pt`. To evaluate an S checkpoint, set `VARIANT=S MODEL=/path/to/owt-scratch-gpt2 CHECKPOINT=/path/to/conditional_512_S.pt`.

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

# Test
TEST_JSON=data/proofwriter_all_test.jsonl CHECKPOINT=proofwriter.pt \
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
