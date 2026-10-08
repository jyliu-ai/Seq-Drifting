"""Generation metrics for the text drifting sanity check.

  - gen_ppl : exp(mean NLL) of GPT-2 Large on the decoded token sequences (lower better)
  - entropy : GPT-2 Large mean per-token distribution entropy on those sequences
  - self_bleu : Texygen-style Self-BLEU-4 among the samples (lower = more diverse; needs nltk)
  - gen_time_per_sample / gen_time_total : wall-clock time of the one-step generation
  - distinct_seqs : # of distinct decoded sequences among the samples (diversity)
  - distinct_reals_hit : # of distinct real rows the batch's nearest 2-gram lands on
  - exact_match : generations whose decoded tokens equal a real row exactly

GPT-2 Large is the EVAL model only (a separate scorer of naturalness); it is not
part of the generator's representation. It is loaded lazily + cached.
"""
import math
import time
from functools import lru_cache
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from common.losses import min_2gram_dist


@lru_cache(maxsize=1)
def _get_gpt2_large(device_str: str):
    from transformers import GPT2LMHeadModel
    print("[eval] Loading GPT-2 Large for gen_ppl/entropy ...")
    dev = torch.device(device_str)
    # fp16 on GPU halves the (B, T, V) logits footprint and speeds up the matmul;
    # the per-chunk reduction below upcasts to fp32 so the scores stay stable.
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    model = GPT2LMHeadModel.from_pretrained("gpt2-large", torch_dtype=dtype).to(dev)
    model.eval()
    print(f"[eval] GPT-2 Large loaded ({dtype}).")
    return model


@torch.no_grad()
def gpt2_large_ppl_entropy(token_ids: torch.Tensor, device: torch.device,
                           batch_size: int = 32, t_chunk: int = 128) -> Tuple[float, float]:
    """gen_ppl + mean per-token distribution entropy under GPT-2 Large.

    Peak memory is bounded to (batch_size, t_chunk, V): a full (B, T, V) fp32
    log_softmax plus its exp is what spikes eval memory at long T, so both the NLL
    and the entropy are reduced over time in chunks and only the small chunk is
    upcast to fp32.
    """
    gpt2 = _get_gpt2_large(str(device))
    total_nll = total_ent = 0.0
    total_toks = 0
    for s in range(0, token_ids.shape[0], batch_size):
        ids = token_ids[s : s + batch_size].to(device)
        logits = gpt2(input_ids=ids[:, :-1]).logits            # (b, T-1, V) fp16 on cuda
        labels = ids[:, 1:]
        Lm1 = labels.shape[1]
        for t in range(0, Lm1, t_chunk):
            lg = logits[:, t : t + t_chunk].float()            # (b, tc, V) fp32, small
            lb = labels[:, t : t + t_chunk]
            total_nll += F.cross_entropy(lg.reshape(-1, lg.size(-1)),
                                         lb.reshape(-1), reduction="sum").item()
            log_probs = F.log_softmax(lg, dim=-1)
            total_ent += (-(log_probs.exp() * log_probs).sum(-1)).sum().item()
            total_toks += lb.numel()
        del logits
    return math.exp(total_nll / max(total_toks, 1)), total_ent / max(total_toks, 1)


@torch.no_grad()
def nearest_real_report(gen_ng, bank, tokenizer, top: int = 10, chunk: int = 256):
    """For each generation in the CURRENT batch, find its nearest real row by the
    drift's position-aligned n-gram distance, then summarise how concentrated the
    mapping is (are many gens being pulled to the same row?).

    Returns (distinct_reals, top1_frac, topK_frac, lines) where lines describe
    the most-hit reals "real#i <- c gens : <text>".
    """
    N, M, D2 = bank.real_ng.shape
    real_flat = bank.real_ng.reshape(N, M * D2)              # position-aligned
    B = gen_ng.shape[0]
    nn = torch.empty(B, dtype=torch.long, device=gen_ng.device)
    for s in range(0, B, chunk):
        g = gen_ng[s:s + chunk]
        dd = torch.cdist(g.reshape(g.shape[0], M * D2), real_flat)  # (b, N) pos-aligned L2
        nn[s:s + chunk] = dd.argmin(dim=1)
    counts = torch.bincount(nn, minlength=N)
    distinct = int((counts > 0).sum())
    topc, topi = counts.topk(min(top, N))
    lines = []
    for c, i in zip(topc.tolist(), topi.tolist()):
        if c == 0:
            continue
        txt = tokenizer.decode(bank.tokens[i].tolist(), skip_special_tokens=True)
        lines.append(f"    real#{i} <- {c} gens : {txt!r}")
    return distinct, topc[0].item() / B, topc.sum().item() / B, lines


@torch.no_grad()
def matched_positives_report(gen_ng, gen_tokens, bank, tokenizer, n_pos: int,
                             select_mode: str = "repr_min2gram",
                             n_show: int = 3, show_pos: int = 6):
    """For the first n_show generations, show what the ACTIVE selection picks
    (n = bank.n, the current curriculum window).

    repr_min2gram: each generation's top-n_pos reals by min-n-gram distance.
    token_2gram_match: the reals each generation EXACTLY matches at a same-position
        n-gram (and how many; "none -> fallback" if zero). Shows whether any real
        shares an exact same-position n consecutive tokens with the generation.
    """
    lines = []
    n, n_ng = bank.n, bank.n_ng
    if select_mode == "token_2gram_match":
        gc = gen_tokens.to(torch.int64).to(bank.tokens.device)
        n_with_match = 0
        for g in range(min(n_show, gen_tokens.shape[0])):
            mt = torch.zeros(bank.N, dtype=torch.bool, device=bank.tokens.device)
            for i in range(n_ng):                              # any-position n-gram match
                mm = torch.ones(bank.N, dtype=torch.bool, device=bank.tokens.device)
                for k in range(n):
                    mm &= gc[g, i + k] == bank.tokens[:, i + k]
                mt |= mm
            idxs = mt.nonzero().flatten()
            n_with_match += int(idxs.numel() > 0)
            gtxt = tokenizer.decode(gen_tokens[g].tolist(), skip_special_tokens=True)
            lines.append(f"  gen#{g} {gtxt!r}  -> {idxs.numel()} exact same-pos {n}-gram matches:")
            if idxs.numel() == 0:
                lines.append("      (none -> fallback to repr min-n-gram nearest)")
            for r in idxs[:show_pos].tolist():
                rtxt = tokenizer.decode(bank.tokens[r].tolist(), skip_special_tokens=True)
                lines.append(f"      real#{r} {rtxt!r}")
        return n_with_match, lines

    dmin = min_2gram_dist(gen_ng.to(bank.real_ng.device), bank.real_ng)  # (G, N)
    k = min(n_pos, bank.N)
    top_d, top_i = (-dmin).topk(k, dim=-1)                     # smallest dist
    top_d = -top_d
    for g in range(min(n_show, gen_ng.shape[0])):
        gtxt = tokenizer.decode(gen_tokens[g].tolist(), skip_special_tokens=True)
        lines.append(f"  gen#{g} {gtxt!r}  -> top min-{n}gram matches:")
        for r, d in list(zip(top_i[g].tolist(), top_d[g].tolist()))[:show_pos]:
            rtxt = tokenizer.decode(bank.tokens[r].tolist(), skip_special_tokens=True)
            lines.append(f"      real#{r} (d={d:.3f}) {rtxt!r}")
    distinct_union = int(torch.unique(top_i).numel())
    return distinct_union, lines


_PUNKT_READY = False


def _ensure_punkt():
    """Texygen tokenizes with nltk.word_tokenize, which needs the punkt data."""
    global _PUNKT_READY
    if _PUNKT_READY:
        return
    import nltk
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    _PUNKT_READY = True


def self_bleu(token_ids: torch.Tensor, eos_id, tokenizer, n_gram: int = 4,
              sample_size: int = 500, seed: int = 0, mode: str = "pairwise",
              n_refs: int = 50):
    """Self-BLEU on WORD tokens (nltk.word_tokenize), two protocols:

    mode="pairwise" (default): each sample is scored by BLEU-n against `n_refs` other
        samples taken ONE AT A TIME (single-reference), then averaged. Two independent
        fluent texts share few n-grams, so this lands in the usual low range and is
        what discriminates diversity (diverse -> low, collapsed -> high). This matches
        the range the diffusion baselines report.

    mode="multiref": Texygen's protocol -- each sample vs ALL other samples as one
        reference set. On long sequences the union of references covers almost every
        common n-gram, so BLEU precision saturates near 1.0 for ANY fluent generator
        and stops measuring diversity. Kept for reference, not recommended here.

    Words, not BPE ids: subword n-grams overlap far more, inflating the score. Returns
    None if nltk is unavailable."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
    except Exception:
        print("[eval] nltk not installed; skipping Self-BLEU (pip install nltk).")
        return None
    import random
    _ensure_punkt()

    seqs = []
    for row in token_ids.tolist():
        if eos_id is not None and eos_id in row:
            row = row[:row.index(eos_id)]
        words = nltk.word_tokenize(tokenizer.decode(row, skip_special_tokens=True))
        if words:
            seqs.append(words)
    n = len(seqs)
    if n < 2:
        return 0.0
    rng = random.Random(seed)
    idx = list(range(n))
    if sample_size and sample_size < n:
        idx = rng.sample(idx, sample_size)
    pool = [seqs[j] for j in idx]                              # working set
    P = len(pool)
    weights = tuple(1.0 / n_gram for _ in range(n_gram))
    smooth = SmoothingFunction().method1
    scores = []
    if mode == "multiref":
        for k in range(P):
            refs = pool[:k] + pool[k + 1:]                    # all OTHER samples at once
            scores.append(sentence_bleu(refs, pool[k], weights=weights,
                                        smoothing_function=smooth))
    else:                                                     # pairwise (single-reference mean)
        for k in range(P):
            others = list(range(k)) + list(range(k + 1, P))
            js = others if len(others) <= n_refs else rng.sample(others, n_refs)
            s = sum(sentence_bleu([pool[j]], pool[k], weights=weights,
                                  smoothing_function=smooth) for j in js)
            scores.append(s / max(len(js), 1))
    return float(sum(scores) / max(len(scores), 1))


@torch.no_grad()
def run_eval(raw_model, embedder, bank, cfg, device, tokenizer,
             n_samples: int, n_show: int = 10, compute_ppl: bool = True
             ) -> Tuple[Dict[str, float], List[str]]:
    """Sample continuous embeddings, decode to tokens via cosine-NN, score."""
    was_training = raw_model.training
    raw_model.eval()
    z = raw_model.sample_z(n_samples, cfg.noise_dim, cfg.temp, device)
    # Generate in chunks: the whole n_samples in one forward materialises a
    # (n_samples, H, T, T) attention matrix (46+ GiB at T=1024, n_samples=1000).
    # Chunking keeps the peak at (gen_chunk, H, T, T) while emb stays full.
    gen_chunk = int(getattr(cfg, "eval_gen_chunk", 64))
    is_cuda = torch.device(device).type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    emb = torch.cat([raw_model(z[i:i + gen_chunk]) for i in range(0, n_samples, gen_chunk)],
                    dim=0)                                     # (B, T, D)
    if is_cuda:
        torch.cuda.synchronize(device)
    gen_time = time.perf_counter() - t0                        # one-step generation of n_samples
    tokens = embedder.decode(emb)                              # (B, T)
    gen_ng = embedder.to_ngrams(emb, bank.n)                   # (B, T-n+1, n*D)

    distinct_seqs = torch.unique(tokens, dim=0).shape[0]
    has_bank = getattr(bank, "real_ng", None) is not None      # skip_bank -> no real n-grams
    if has_bank:
        distinct = bank.select_positives(gen_ng, n_pos=1).numel()
        real = bank.tokens
        exact = int((tokens[:, None, :] == real[None, :, :]).all(-1).any(-1).sum().item())
    else:
        distinct = 0                                           # distinct_reals_hit / exact skipped
        exact = 0

    metrics = {
        "distinct_seqs": distinct_seqs,
        "distinct_reals_hit": distinct,
        "exact_match": exact,
        "n_samples": n_samples,
        "gen_time_total": gen_time,                            # sec to generate all n_samples
        "gen_time_per_sample": gen_time / max(n_samples, 1),   # sec/sample (Time column)
    }
    sb = self_bleu(tokens, getattr(tokenizer, "eos_token_id", None), tokenizer,
                   n_gram=int(getattr(cfg, "self_bleu_ngram", 4)),
                   sample_size=int(getattr(cfg, "self_bleu_samples", 500)))
    if sb is not None:
        metrics["self_bleu"] = sb
    if compute_ppl:
        # seq_len is the CONTENT length; wrap [BOS] + content + [EOS] so GPT-2 gets
        # a BOS prefix (and scores the first content token / the ending) for ppl.
        B = tokens.shape[0]
        bos = torch.full((B, 1), tokenizer.bos_token_id, dtype=tokens.dtype, device=tokens.device)
        eos = torch.full((B, 1), tokenizer.eos_token_id, dtype=tokens.dtype, device=tokens.device)
        wrapped = torch.cat([bos, tokens, eos], dim=1)
        ppl, ent = gpt2_large_ppl_entropy(wrapped, device)
        metrics["gen_ppl"] = ppl
        metrics["entropy"] = ent

    shown = [tokenizer.decode(row.tolist(), skip_special_tokens=True)
             for row in tokens[:n_show]]
    if was_training:
        raw_model.train()
    return metrics, shown



@lru_cache(maxsize=1)
def _get_eval_lm(name: str, device_str: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[eval] loading reference LM {name} for gen-ppl ...")
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(name).to(torch.device(device_str)).eval()
    return m, tok


@torch.no_grad()
def generative_ppl(texts: List[str], device, eval_model: str = "gpt2-large",
                   batch_size: int = 16, max_len: int = 256) -> float:
    """FLM-style GENERATIVE PERPLEXITY: exp(mean NLL) of the generated text under a reference
    LM (gpt2-large), corpus-level (= torchmetrics Perplexity in FLM). Lower = more fluent."""
    texts = [t for t in texts if t.strip()]
    if not texts:
        return float("nan")
    m, tok = _get_eval_lm(eval_model, str(device))
    tot_nll = 0.0
    tot_tok = 0
    for s in range(0, len(texts), batch_size):
        enc = tok(texts[s:s + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
        ids = enc.input_ids.to(device)
        am = enc.attention_mask.to(device)
        logits = m(input_ids=ids, attention_mask=am).logits
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)      # predict token t+1
        nll = -lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        mask = am[:, 1:].bool()
        tot_nll += nll[mask].sum().item()
        tot_tok += int(mask.sum())
    return math.exp(tot_nll / max(tot_tok, 1))


def token_stats(id_lists: List[List[int]]):
    """FLM-style diversity on the GENERATED token ids (no reference LM):
      entropy  = mean over samples of the per-sample token-distribution entropy
                 (-sum p log p over the unique tokens IN that sample); in nats, <= log(len).
      uniq_tok = unique tokens / total tokens across all samples.
    This is what FLM's record_entropy / record_unique_tokens report -- a DIVERSITY measure,
    NOT the reference-LM predictive entropy (which was the old, non-comparable definition)."""
    import math as _m
    ents, all_ids = [], []
    for ids in id_lists:
        if not ids:
            continue
        all_ids.extend(ids)
        counts: Dict[int, int] = {}
        for t in ids:
            counts[t] = counts.get(t, 0) + 1
        n = len(ids)
        ents.append(-sum((c / n) * _m.log(c / n) for c in counts.values()))
    entropy = sum(ents) / max(len(ents), 1)
    uniq = len(set(all_ids)) / max(len(all_ids), 1)
    return entropy, uniq
