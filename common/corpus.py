"""Tiny token-chunk loader for the text drifting sanity check.

This loader implements the token packing used by these experiments:
load a dataset, GPT-2 tokenise with the MDLM packing protocol, and return the
first ``n_train`` chunks as a single ``(N, seq_len)`` LongTensor. The whole set
fits in the memory bank, so no DataLoader / sampler machinery is required.

Packing protocol:
  1. Tokenise each doc with ``add_special_tokens=False``; append ONE EOS per doc
     (a content-level doc separator).
  2. Concatenate every doc's token stream into one flat array.
  3. Chunk into blocks of ``seq_len`` CONTENT tokens (drop the tail).

NOTE: ``seq_len`` is the CONTENT length -- the tokens the generator actually
models. There is NO fixed [BOS]/[EOS] wrapping in the stored rows (so training
never sees them and they cannot create cheap same-position boundary matches).
The wrapping ``[BOS] + content + [EOS]`` is added only at eval time (e.g. so the
GPT-2 perplexity scorer gets a BOS prefix). For GPT-2 bos==eos==50256.

For openwebtext we *stream* documents and stop as soon as we have enough chunks,
so the full ~12GB corpus is never downloaded for a 1000-row run.
"""
from typing import Tuple

import numpy as np
import torch
from transformers import GPT2TokenizerFast


# dataset name -> (hf_name, hf_config, streamable, filter_headings)
_DATASET_MAP = {
    "openwebtext":  ("openwebtext",   None,                  True,  False),
    "wikitext-2":   ("wikitext",      "wikitext-2-raw-v1",   False, True),
    "wikitext-103": ("wikitext",      "wikitext-103-raw-v1", False, True),
    "ptb":          ("ptb_text_only", "penn_treebank",       False, False),
}


def get_tokenizer(name: str = "gpt2") -> GPT2TokenizerFast:
    tok = GPT2TokenizerFast.from_pretrained(name)
    tok.pad_token = tok.eos_token
    tok.model_max_length = int(1e30)   # silence the >1024 length warning
    return tok


def _keep(text: str, filter_headings: bool) -> bool:
    t = text.strip()
    if not t:
        return False
    if filter_headings and t.startswith("=") and t.endswith("="):
        return False
    return True


def _iter_texts(dataset_name: str, n_chunks: int, content_size: int,
                filter_headings: bool, verbose: bool):
    """Yield raw document strings, streaming when possible. We only need roughly
    ``n_chunks * content_size`` content tokens, so streaming stops early."""
    from datasets import load_dataset
    hf_name, hf_config, streamable, _ = _DATASET_MAP[dataset_name]

    if streamable:
        ds = load_dataset(hf_name, split="train", streaming=True,
                          trust_remote_code=True)
        for row in ds:
            yield row["text"]
    else:
        ds = load_dataset(hf_name, hf_config, split="train",
                         trust_remote_code=True)
        key = "sentence" if "sentence" in ds.column_names else "text"
        for t in ds[key]:
            yield t


def load_token_chunks(cfg, verbose: bool = True
                      ) -> Tuple[torch.Tensor, GPT2TokenizerFast]:
    """Return ``(tokens (N, seq_len) long, tokenizer)`` with ``N <= cfg.n_train``."""
    if cfg.dataset_name not in _DATASET_MAP:
        raise ValueError(f"Unknown dataset '{cfg.dataset_name}'. "
                         f"Choose from {list(_DATASET_MAP)}")
    _, _, _, filter_headings = _DATASET_MAP[cfg.dataset_name]
    tok = get_tokenizer(cfg.tokenizer_name)
    eos_id = tok.eos_token_id                       # per-doc separator (content-level)
    content_size = cfg.seq_len                      # seq_len = CONTENT length (no BOS/EOS wrap)
    assert content_size >= 2, "seq_len (content length) must be >= 2"

    if verbose:
        print(f"Loading {cfg.dataset_name}: need {cfg.n_train} chunks "
              f"of {content_size} content tokens (no BOS/EOS wrap) ...")

    flat: list[int] = []
    need = cfg.n_train * content_size
    if getattr(cfg, "data_path", ""):
        from common.local_data import iter_texts
        texts = iter_texts(cfg.data_path)
    else:
        texts = _iter_texts(cfg.dataset_name, cfg.n_train, content_size,
                            filter_headings, verbose)
    for text in texts:
        if not _keep(text, filter_headings):
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        flat.extend(ids)
        flat.append(eos_id)
        if len(flat) >= need:
            break

    n_chunks = min(cfg.n_train, len(flat) // content_size)
    if n_chunks == 0:
        raise RuntimeError("Not enough tokens to form a single chunk.")
    flat = np.asarray(flat[: n_chunks * content_size], dtype=np.int64)
    chunks = flat.reshape(n_chunks, content_size)          # (N, seq_len) CONTENT only

    if verbose:
        print(f"  Built {n_chunks} content chunks. Example decode:")
        print("   ", repr(tok.decode(chunks[0].tolist())))
    return torch.from_numpy(chunks), tok
