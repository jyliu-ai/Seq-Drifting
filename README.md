# Seq-Drifting

Anonymous source package for the Seq-Drifting main-model experiments:
unconditional generation, conditional continuation, translation, summarization,
mathematical reasoning, and ProofWriter.

All commands are run from this directory. Install the dependencies and use an
already configured Python/CUDA environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Set `CUDA_VISIBLE_DEVICES` and replace the example paths with paths on the target
machine. The scripts do not activate Conda.

## Unconditional generation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPT2_TEACHER=/models/gpt2 \
TOKENIZER=/models/gpt2 EMBED_MODEL=/models/gpt2 \
CKPT_DIR=/runs/unconditional bash scripts/train_unconditional.sh
```

Evaluate a checkpoint:

```bash
python -m uncond_drift.eval_ckpt \
  --ckpt /runs/unconditional/step_200000.pt \
  --tokenizer-name /models/gpt2 --embed-model /models/gpt2 --which ema
```

## Conditional continuation

`OWT_DIR` should contain the prepared OpenWebText2 JSONL-Zstandard shards.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 \
OWT_DIR=/data/openwebtext2 TEACHER=/models/gpt2 \
CKPT_DIR=/runs/conditional bash scripts/train_conditional.sh
```

## Translation

Use prepared WMT14 German-English data and a task-finetuned teacher:

```bash
DATASET=wmt14_de_en DATA_DIR=/data/wmt14_de_en BASE_MODEL=/models/gpt2 \
OUTPUT_DIR=/runs/teacher-wmt bash scripts/train_seq2seq_teacher.sh

DATASET=wmt14_de_en DATA_DIR=/data/wmt14_de_en \
TEACHER=/runs/teacher-wmt/best CKPT_DIR=/runs/wmt14-de-en \
EVAL_SPLIT=test bash scripts/train_translation.sh
```

Evaluate on the test split:

```bash
python -m cond_drift_seq2seq.eval_ckpt \
  --ckpt /runs/wmt14-de-en/best.pt \
  --data-dir /data/wmt14_de_en --teacher /runs/teacher-wmt/best --split test
```

## Summarization

Use prepared XSum data and a task-finetuned teacher:

```bash
DATASET=xsum DATA_DIR=/data/xsum BASE_MODEL=/models/gpt2 \
OUTPUT_DIR=/runs/teacher-xsum bash scripts/train_seq2seq_teacher.sh

DATASET=xsum DATA_DIR=/data/xsum \
TEACHER=/runs/teacher-xsum/best CKPT_DIR=/runs/xsum \
EVAL_SPLIT=test bash scripts/train_summarization.sh
```

Evaluate on the test split:

```bash
python -m cond_drift_seq2seq.eval_ckpt \
  --ckpt /runs/xsum/best.pt \
  --data-dir /data/xsum --teacher /runs/teacher-xsum/best --split test
```

## Mathematical reasoning

Use the prepared GSM8K-Aug equation JSONL for training, and the prepared SVAMP
or MAWPS JSONL directly for evaluation.

Train:

```bash
TRAIN_JSON=/data/gsm8k_aug_equation_train.jsonl \
EVAL_JSON=/data/gsm8k_aug_equation_test.jsonl \
MODEL=/models/qwen STEPS=120000 CKPT_DIR=/runs/math \
bash scripts/train_math.sh
```

Evaluate on SVAMP:

```bash
CHECKPOINT=/runs/math/step_120000.pt MODEL=/models/qwen \
TEST_JSON=/data/svamp_test.jsonl bash scripts/eval_math.sh
```

Evaluate on MAWPS:

```bash
CHECKPOINT=/runs/math/step_120000.pt MODEL=/models/qwen \
TEST_JSON=/data/mawps_test.jsonl bash scripts/eval_math.sh
```

## ProofWriter

Use the prepared combined-depth JSONL files directly. The example evaluates on
the real test split.

Train:

```bash
TRAIN_JSON=/data/proofwriter_all_train.jsonl \
EVAL_JSON=/data/proofwriter_all_test.jsonl MODEL=/models/qwen \
CKPT_DIR=/runs/proofwriter bash scripts/train_proofwriter.sh
```

Evaluate with five stochastic repeats:

```bash
CHECKPOINT=/runs/proofwriter/step_60000.pt \
TEST_JSON=/data/proofwriter_all_test.jsonl MODEL=/models/qwen \
bash scripts/eval_proofwriter.sh
```
