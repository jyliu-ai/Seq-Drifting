"""(query, response) datasets for conditional drifting.

Two supervision sources are prepared:
  * the QUERY  -- conditioning for the generator and context for the Qwen teacher;
  * the GOLD RESPONSE (full solution text, tokenised to resp_len) -- the positive
    the generation is pulled toward when cfg.gold_weight > 0. Pure teacher
    self-distillation has no notion of a CORRECT answer, so on a task scored by
    accuracy the gold response is the signal that actually carries correctness.

Queries are wrapped with the chat template (add_generation_prompt=True) and
LEFT-padded to query_len, so the last query token is always real (the assistant cue)
and the response logits line up at position query_len-1. Gold responses are
RIGHT-padded to resp_len with a mask (no gold force past the real tokens).
"""
import re
from dataclasses import dataclass
from typing import List, Optional

import torch


def _extract_gsm8k_answer(ans: str) -> str:
    m = re.search(r"####\s*(.+)", ans)
    if not m:
        return ans.strip()
    return m.group(1).strip().replace(",", "").replace("$", "")


def _extract_boxed(sol: str) -> str:
    i = sol.rfind(r"\boxed")
    if i < 0:
        return sol.strip()
    j = sol.find("{", i)
    if j < 0:
        return sol.strip()
    depth = 0
    for k in range(j, len(sol)):
        if sol[k] == "{":
            depth += 1
        elif sol[k] == "}":
            depth -= 1
            if depth == 0:
                return sol[j + 1:k].strip()
    return sol.strip()


def _load_gsm8k_local(path: str, split: str):
    """Offline GSM8K from a nested json {"GSM8K@i": {ori_question, ori_answer, ...}}.
    Deterministic 95/5 split by index (i%20==0 -> test), no network."""
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    qs, finals, texts = [], [], []
    for i, (_, v) in enumerate(sorted(data.items())):
        q = v.get("ori_question")
        if q is None:
            continue
        is_test = (i % 20 == 0)
        if (split == "test") != is_test:
            continue
        ans = v.get("ori_answer", "")
        qs.append(q)
        finals.append(_extract_gsm8k_answer(ans))
        texts.append(ans)                       # full solution incl. the '#### x' line
    return qs, finals, texts


def _load_gsm8k_eq(path: str, split: str, test_path: str = ""):
    """Offline GSM8K from a FLAT jsonl (one {"question","answer"} per line). The answer may be the
    full natural-language solution or the equation-only variant; either way accuracy is read from
    the '#### final' line and the whole answer is kept as the gold text (strip_calc_annot removes
    the inline '<<...>>' markup at build time).

    If test_path is given, the two files are the explicit train / test splits (each read whole);
    otherwise a single file is split deterministically by line index (i%10==0 -> test)."""
    import json
    from pathlib import Path

    if not path:
        raise ValueError("gsm8k_eq requires --gsm8k-json for the training JSONL")
    if not Path(path).is_file():
        raise FileNotFoundError(f"gsm8k_eq training JSONL not found: {path}")

    explicit_split = bool(test_path)
    if explicit_split and not Path(test_path).is_file():
        raise FileNotFoundError(f"gsm8k_eq test JSONL not found: {test_path}")

    # With --test-json, the two files are independent explicit splits.  The
    # fallback line-index split is only allowed when no test file is supplied.
    src = path if (not explicit_split or split == "train") else test_path
    qs, finals, texts = [], [], []
    with open(src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not explicit_split:                  # one file -> deterministic 90/10 split
                if (split == "test") != (i % 10 == 0):
                    continue
            # Metamath-generated records store the model target under
            # `generated`; keep compatibility with standard GSM8K JSONL.
            ans = str(row.get("generated", row.get("answer", "")))
            if not ans:
                raise KeyError("generated/answer")
            qs.append(str(row["question"]))
            finals.append(_extract_gsm8k_answer(ans))
            texts.append(ans)                       # full solution incl. the '#### x' line
    if not qs:
        raise ValueError(f"gsm8k_eq {split} split is empty: {src}")
    return qs, finals, texts


# ── OpenWebText2 continuation (prefix -> continuation, PURE TEACHER) ─────────────
# The generator is conditioned on a real text PREFIX (the query) and the Qwen teacher
# scores [prefix ++ generated continuation]; the corpus continuation is kept only as an
# eval reference (gold_weight=0 -> it is never a training target). So the loader only has
# to hand back realistic prefixes.
_OWT_CACHE = {}


def _iter_owt_docs(owt_dir, max_docs):
    """Yield document texts from the *.jsonl.zst shards (offline, streaming)."""
    import glob
    import io
    import json
    import os
    import zstandard
    shards = sorted(glob.glob(os.path.join(owt_dir, "*.jsonl.zst")))
    if not shards:
        raise FileNotFoundError(f"no *.jsonl.zst under {owt_dir!r}")
    n = 0
    for shard in shards:
        with open(shard, "rb") as fh:
            with zstandard.ZstdDecompressor().stream_reader(fh) as reader:
                for line in io.TextIOWrapper(reader, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        txt = json.loads(line).get("text", "")
                    except Exception:
                        continue
                    if txt:
                        yield txt
                        n += 1
                        if n >= max_docs:
                            return


def _owt_examples(cfg, tokenizer):
    """Tokenise docs ONCE (cached) -> list of (prefix_ids[:Lq], cont_ref_ids[:Lr]).

    PURE TEACHER uses NO corpus ground-truth, so training only needs the PREFIX: we slide
    NON-overlapping Lq-token windows over the whole document (stride = Lq), and every
    Lq-chunk becomes a prefix -> ~Lr/Lq times more prefixes per doc than a Lq+Lr stride,
    and the whole doc is used. The next up-to-Lr tokens are kept only as an EVAL reference
    (never a training target; may be short/empty at a doc end). Lr (the GENERATION length)
    is an independent hyperparameter, decoupled from the data window. owt_docs caps EXAMPLES."""
    key = (cfg.owt_dir, cfg.owt_docs, cfg.query_len, cfg.resp_len)
    if key in _OWT_CACHE:
        return _OWT_CACHE[key]
    Lq, Lr = cfg.query_len, cfg.resp_len
    ex = []
    for txt in _iter_owt_docs(cfg.owt_dir, cfg.owt_docs):
        toks = tokenizer(txt, add_special_tokens=False)["input_ids"]
        for start in range(0, len(toks) - Lq + 1, Lq):             # non-overlapping PREFIXES
            prefix = toks[start:start + Lq]
            cont_ref = toks[start + Lq:start + Lq + Lr]            # reference only (eval display)
            ex.append((prefix, cont_ref))
            if len(ex) >= cfg.owt_docs:
                break
        if len(ex) >= cfg.owt_docs:
            break
    _OWT_CACHE[key] = ex
    return ex


def build_owt_dataset(cfg, tokenizer, split: str) -> "CondDataset":
    ex = _owt_examples(cfg, tokenizer)
    Lq, Lr = cfg.query_len, cfg.resp_len
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    q_rows, q_msk, r_rows, r_msk, golds = [], [], [], [], []
    for i, (prefix, cont) in enumerate(ex):
        is_eval = (i % 50 == 0)                                     # deterministic 2% held-out
        if (split == "test") != is_eval:
            continue
        q_rows.append(list(prefix))                                # long doc -> exactly Lq, no pad
        q_msk.append([True] * Lq)
        rpad = Lr - len(cont)
        r_rows.append(list(cont) + [pad_id] * rpad)                # reference continuation
        r_msk.append([True] * len(cont) + [False] * rpad)
        golds.append(tokenizer.decode(cont, skip_special_tokens=True))

    N = len(q_rows)
    print(f"[data:{split}] OWT2 continuation N={N} Lq={Lq} Lr={Lr} (prefix->continuation, "
          f"pure-teacher; corpus continuation kept as eval reference only)")
    return CondDataset(
        query_ids=torch.tensor(q_rows, dtype=torch.long),
        query_mask=torch.tensor(q_msk, dtype=torch.bool),
        resp_ids=torch.tensor(r_rows, dtype=torch.long),
        resp_mask=torch.tensor(r_msk, dtype=torch.bool),
        gold=golds, pad_id=pad_id)


def _load_pairs(cfg, split: str):
    """-> (questions, gold_final (for accuracy), gold_text (full response))."""
    name = cfg.dataset_name
    if name == "gsm8k_local":
        return _load_gsm8k_local(cfg.local_json, split)
    if name == "gsm8k_eq":
        return _load_gsm8k_eq(cfg.local_json, split, getattr(cfg, "test_json", ""))
    if name == "jsonl":
        import json
        qs, texts = [], []
        with open(cfg.jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qs.append(str(row[cfg.jsonl_query_key]))
                texts.append(str(row.get(cfg.jsonl_response_key, "")))
        return qs, list(texts), texts

    from datasets import load_dataset
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        ans = list(ds["answer"])
        return list(ds["question"]), [_extract_gsm8k_answer(a) for a in ans], ans
    if name == "math":
        sp = "train" if split == "train" else "test"
        ds = load_dataset("hendrycks/competition_math", split=sp)
        sol = list(ds["solution"])
        return list(ds["problem"]), [_extract_boxed(s) for s in sol], sol
    if name == "svamp":
        # SVAMP: Body + Question -> numeric Answer (no chain-of-thought, so the gold "response"
        # is just the answer). Local svamp_path (a JSON list of {Body,Question,Answer}) is used
        # offline; otherwise the ChilleD/SVAMP hub set (train 700 / test 300).
        if cfg.svamp_path:
            import json
            with open(cfg.svamp_path, encoding="utf-8") as f:
                rows = json.load(f)
            has_split = False                              # local dump: deterministic i%10 split
        else:
            sp = "train" if split == "train" else "test"
            ds = load_dataset("ChilleD/SVAMP", split=sp)
            rows = [{"Body": b, "Question": q, "Answer": a}
                    for b, q, a in zip(ds["Body"], ds["Question"], ds["Answer"])]
            has_split = True
        qs, finals = [], []
        for i, row in enumerate(rows):
            if not has_split and ((split == "test") != (i % 10 == 0)):
                continue
            body, q = str(row.get("Body", "")).strip(), str(row.get("Question", "")).strip()
            ans = str(row.get("Answer", "")).strip()
            qs.append(f"{body} {q}".strip()); finals.append(ans)
        return qs, finals, list(finals)
    if name == "gpqa":
        # GPQA: eval-only 4-choice benchmark. Local gpqa_path (a gpqa_*.csv with columns
        # Question / Correct Answer / Incorrect Answer 1..3) is used offline; otherwise the
        # Idavidrein/gpqa hub set (gpqa_main, single 'train' split). Each item is formatted as a
        # shuffled multiple choice; gold = the correct letter, gold text = 'X. <answer>'. The
        # same set is returned for both splits (there is no train/test split to speak of).
        if cfg.gpqa_path:
            import csv
            with open(cfg.gpqa_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        else:
            rows = list(load_dataset("Idavidrein/gpqa", "gpqa_main", split="train"))
        import random
        letters = "ABCD"
        rng = random.Random(0)
        qs, finals, texts = [], [], []
        for row in rows:
            correct = str(row["Correct Answer"]).strip()
            opts = [correct,
                    str(row["Incorrect Answer 1"]).strip(),
                    str(row["Incorrect Answer 2"]).strip(),
                    str(row["Incorrect Answer 3"]).strip()]
            rng.shuffle(opts)
            body = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))
            q = f"{str(row['Question']).strip()}\n{body}\nAnswer:"
            cl = letters[opts.index(correct)]
            qs.append(q); finals.append(cl); texts.append(f"{cl}. {correct}")
        return qs, finals, texts
    raise ValueError(f"unknown dataset_name {name!r}")


@dataclass
class CondDataset:
    query_ids: torch.Tensor      # (N, query_len) long, LEFT-padded
    query_mask: torch.Tensor     # (N, query_len) bool
    resp_ids: torch.Tensor       # (N, resp_len)  long, gold response, RIGHT-padded
    resp_mask: torch.Tensor      # (N, resp_len)  bool (True = real gold token)
    gold: List[str]              # gold FINAL answers (accuracy eval)
    pad_id: int

    def __len__(self):
        return self.query_ids.shape[0]

    def sample(self, n: int, generator: Optional[torch.Generator] = None):
        """Random n rows -> (query ids, query mask, gold resp ids, gold resp mask, idx)."""
        idx = torch.randint(0, len(self), (n,), generator=generator)
        return (self.query_ids[idx], self.query_mask[idx],
                self.resp_ids[idx], self.resp_mask[idx], idx)


def build_dataset(cfg, tokenizer, split: str) -> CondDataset:
    if cfg.dataset_name == "owt":                          # continuation task (prefix -> cont)
        return build_owt_dataset(cfg, tokenizer, split)
    questions, finals, texts = _load_pairs(cfg, split)
    if cfg.n_train and split == "train" and len(questions) > cfg.n_train:
        questions, finals, texts = (questions[:cfg.n_train], finals[:cfg.n_train],
                                    texts[:cfg.n_train])

    Lq, Lr = cfg.query_len, cfg.resp_len
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = pad_id

    q_rows, q_msk, r_rows, r_msk = [], [], [], []
    q_trunc = r_trunc = 0                                   # how many got cut by Lq / Lr
    q_lens, r_lens = [], []
    strip_calc = getattr(cfg, "strip_calc_annot", False)
    for q, t in zip(questions, texts):
        if strip_calc:                                     # drop inline '<<a-b=c>>' calc markup
            t = re.sub(r"<<[^>]*>>", "", t)
        if cfg.use_chat_template:
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    add_generation_prompt=True, tokenize=False, enable_thinking=False)
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    add_generation_prompt=True, tokenize=False)
        else:                                              # base continuation model: plain cue
            tmpl = getattr(cfg, "base_prompt_template", "") or "{q}"
            prompt = tmpl.format(q=q)
        toks = tokenizer(prompt, add_special_tokens=not cfg.use_chat_template)["input_ids"]
        q_lens.append(len(toks)); q_trunc += len(toks) > Lq
        toks = toks[-Lq:]                                  # LEFT-truncate (keep assistant cue)
        pad = Lq - len(toks)
        q_rows.append([pad_id] * pad + toks)               # LEFT-pad
        q_msk.append([False] * pad + [True] * len(toks))

        # gold response from position 0, then ONE end-of-text token as the delimiter (it IS
        # supervised -- the model has to learn where to stop), then pad. The pad tail carries
        # no target at all: resp_mask is False there and the loss ignores those positions.
        # Over-long solutions are simply truncated.
        rfull = tokenizer(t, add_special_tokens=False)["input_ids"]
        r_lens.append(len(rfull) + 1); r_trunc += len(rfull) + 1 > Lr    # +1 for the EOS
        rt = rfull[:Lr - 1] + [eos_id]
        rpad = Lr - len(rt)
        r_rows.append(rt + [pad_id] * rpad)                # RIGHT-pad
        r_msk.append([True] * len(rt) + [False] * rpad)    # True covers the gold + the EOS

    def pctl(xs, p):
        xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]
    N = len(q_rows)
    print(f"[data:{split}] N={N}  query tok: max={max(q_lens)} p90={pctl(q_lens,0.9)} "
          f"-> Lq={Lq} truncated {q_trunc}/{N} ({q_trunc/N:.0%})")
    print(f"[data:{split}] response tok: max={max(r_lens)} p90={pctl(r_lens,0.9)} "
          f"-> Lr={Lr} truncated {r_trunc}/{N} ({r_trunc/N:.0%})")

    return CondDataset(
        query_ids=torch.tensor(q_rows, dtype=torch.long),
        query_mask=torch.tensor(q_msk, dtype=torch.bool),
        resp_ids=torch.tensor(r_rows, dtype=torch.long),
        resp_mask=torch.tensor(r_msk, dtype=torch.bool),
        gold=finals, pad_id=pad_id)
