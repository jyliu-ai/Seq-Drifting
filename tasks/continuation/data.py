"""Prefix/continuation datasets for LM1B and OpenWebText2."""
from dataclasses import dataclass
from typing import List, Optional

import torch

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
    if name in {"jsonl", "lm1b"}:
        import json
        path = (getattr(cfg, "test_jsonl_path", "") if split == "test" else cfg.jsonl_path)
        if not path:
            if name == "lm1b":
                raise ValueError(f"LM1B {split} requires an explicit prepared JSONL file")
            path = cfg.jsonl_path
        qs, texts = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qs.append(str(row[cfg.jsonl_query_key]))
                texts.append(str(row.get(cfg.jsonl_response_key, "")))
        return qs, list(texts), texts

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
    for q, t in zip(questions, texts):
        if cfg.use_chat_template:
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    add_generation_prompt=True, tokenize=False, enable_thinking=False)
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    add_generation_prompt=True, tokenize=False)
        else:
            prompt = q
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
