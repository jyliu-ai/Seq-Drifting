---
language:
  - en
  - de
pipeline_tag: text-generation
library_name: pytorch
tags:
  - seq-drifting
  - one-step-generation
  - non-autoregressive
  - text-generation
  - translation
  - summarization
  - reasoning
---

# One-Step Text Generation by Seq-Drifting

[Paper](ARXIV_URL_PLACEHOLDER) · [Code](https://github.com/jyliu-ai/Seq-Drifting)

Seq-Drifting generates a text sequence in one generator forward pass, then decodes continuous token representations into discrete tokens. This repository contains checkpoints for unconditional generation, text continuation, translation, summarization, and reasoning.

## Checkpoints

| Task | Checkpoint | Setting |
| --- | --- | --- |
| Unconditional generation | `unconditional_128_P.pt` | 128 tokens, P |
| Unconditional generation | `unconditional_128_S.pt` | 128 tokens, S |
| Unconditional generation | `unconditional_1024_P.pt` | Long sequence, P |
| Unconditional generation | `unconditional_1024_S.pt` | Long sequence, S |
| LM1B continuation | `conditional_64_P.pt` | 64-token prefix → 64 tokens, P |
| LM1B continuation | `conditional_64_S.pt` | 64-token prefix → 64 tokens, S |
| OpenWebText2 continuation | `conditional_512_P.pt` | 64-token prefix → 512 tokens, P |
| WMT14 De → En | `WMT.pt` | Translation |
| XSum | `XSum.pt` | Summarization |
| SVAMP / ASDiv | `math_reasoning.pt` | Mathematical reasoning |
| ProofWriter | `proofwriter.pt` | Logical reasoning, true/false |

**P** uses public pretrained GPT-2 for support-set construction. **S** uses GPT-2 trained from scratch on the corresponding corpus, then frozen. The 128-token unconditional checkpoints use the conditional architecture at the end of unconditional warmup, with a constant visible prefix. The 1024 checkpoints use the original unconditional architecture; the exact generation length is stored in the checkpoint configuration.

## Usage

Use the custom PyTorch loaders and task scripts in the code repository.

```bash
git clone https://github.com/jyliu-ai/Seq-Drifting.git
cd Seq-Drifting
python -m pip install -r requirements.txt
```

Each test script automatically downloads its checkpoint from this repository. Testing defaults to one GPU and EMA weights.

### Unconditional generation

```bash
LENGTH=128 VARIANT=P bash scripts/eval_unconditional.sh
LENGTH=128 VARIANT=S MODEL=/path/to/lm1b-scratch-gpt2 bash scripts/eval_unconditional.sh

LENGTH=1024 VARIANT=P bash scripts/eval_unconditional.sh
LENGTH=1024 VARIANT=S bash scripts/eval_unconditional.sh
```

### Conditional generation

```bash
DATASET=lm1b VARIANT=P TEST_JSON=data/lm1b/test.jsonl bash scripts/eval_conditional.sh

DATASET=lm1b VARIANT=S MODEL=/path/to/lm1b-scratch-gpt2 \
TEST_JSON=data/lm1b/test.jsonl bash scripts/eval_conditional.sh

DATASET=owt OWT_DIR=data/openwebtext2 CHECKPOINT=conditional_512_P.pt \
bash scripts/eval_conditional.sh
```

### Translation and summarization

```bash
python -m common.prepare_seq2seq --dataset wmt14_de_en --output-dir data/wmt14_de_en
DATA_DIR=data/wmt14_de_en TEACHER=/path/to/original-wmt-support-model CHECKPOINT=WMT.pt \
bash scripts/eval_translation.sh

python -m common.prepare_seq2seq --dataset xsum --output-dir data/xsum
DATA_DIR=data/xsum TEACHER=/path/to/original-xsum-support-model CHECKPOINT=XSum.pt \
bash scripts/eval_summarization.sh
```

### Reasoning

```bash
MODEL=Qwen/Qwen2.5-0.5B BACKBONE=Qwen/Qwen2.5-0.5B \
TEST_JSON=data/svamp_test.jsonl CHECKPOINT=math_reasoning.pt bash scripts/eval_math.sh

MODEL=Qwen/Qwen2.5-0.5B BACKBONE=Qwen/Qwen2.5-0.5B \
TEST_JSON=data/asdiv_test.jsonl CHECKPOINT=math_reasoning.pt bash scripts/eval_math.sh

MODEL=Qwen/Qwen2.5-0.5B BACKBONE=Qwen/Qwen2.5-0.5B \
TEST_JSON=data/proofwriter_all_test.jsonl CHECKPOINT=proofwriter.pt bash scripts/eval_proofwriter.sh
```

LM1B uses prepared `query`/`response` JSONL. Reasoning uses prepared `question`/`answer` JSONL, with final answers marked by `####` (`generated` can replace `answer`). OpenWebText2 uses `*.jsonl.zst` shards.

## Model paths

`MODEL`, `TEACHER`, `EMBED_MODEL`, and `BACKBONE` accept a model ID or a local Hugging Face model directory. `CHECKPOINT` selects the Seq-Drifting `.pt` file.

- For 128-token S and conditional S evaluation, `MODEL` must point to the original frozen scratch-trained GPT-2 used for the embedding table.
- The original 1024 implementation stores the embedding model separately from its training support model. Evaluation reads `embed_model` and `tokenizer_name` from the checkpoint; use `EMBED_MODEL` and `TOKENIZER` to relocate them if needed.
- WMT/XSum require the original task-specific frozen model for its embedding table. Use `TEACHER` to relocate it, or omit the override if its saved path is available.
- Reasoning uses the checkpoint's original embedding model and backbone. Override `MODEL` and `BACKBONE` when their saved paths have moved.

Training commands and P/S settings are listed in the [code README](https://github.com/jyliu-ai/Seq-Drifting#readme). Evaluation reports fluency/diversity for open-ended generation, BLEU for translation, ROUGE for summarization, and answer accuracy for reasoning. Experimental results are in the paper.
