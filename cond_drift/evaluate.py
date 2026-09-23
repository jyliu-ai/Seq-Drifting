"""Conditional eval: generate a response per held-out query, decode, and score.

For the continuation task the key metric is FLM-style GENERATIVE PERPLEXITY: the fluency
of the generated text under a reference LM (gpt2-large), plus per-token entropy and a
corpus unique-token ratio -- the same metric set FLM/FMLM report."""
import math
import re
from functools import lru_cache
from typing import Dict, List, Tuple

import torch


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


def _last_number(text: str) -> str:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else ""


def _extract_boxed(text: str) -> str:
    i = text.rfind(r"\boxed")
    if i < 0:
        return _last_number(text)
    j = text.find("{", i)
    if j < 0:
        return _last_number(text)
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k].strip()
    return _last_number(text)


def extract_pred(text: str, dataset_name: str) -> str:
    if dataset_name == "math":
        return _extract_boxed(text)
    return _last_number(text)               # gsm8k / generic numeric


def _norm(s: str) -> str:
    return s.strip().rstrip(".").replace(" ", "").replace("$", "")


@torch.no_grad()
def run_eval(gen, embedder, dataset, cfg, tokenizer, device,
             n_queries: int, n_show: int = 6) -> Tuple[Dict[str, float], List[str]]:
    was_training = gen.training
    gen.eval()
    n = min(n_queries, len(dataset))
    q_ids = dataset.query_ids[:n].to(device)
    q_msk = dataset.query_mask[:n].to(device)
    gold = dataset.gold[:n]

    texts, preds, id_lists = [], [], []
    B = 32
    for s in range(0, n, B):
        qi, qm = q_ids[s:s + B], q_msk[s:s + B]
        qe = embedder.query_embeds(qi)
        z = gen.sample_z(qi.shape[0], cfg.noise_dim, cfg.temp, device)
        emb = gen(qe, qm, z)
        if cfg.sphere_norm:
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        toks = embedder.decode(emb)                        # (b, Lr)
        eos_id = tokenizer.eos_token_id
        for row in toks:
            ids = row.tolist()
            if eos_id is not None and eos_id in ids:       # the model learns to emit EOS; the
                ids = ids[:ids.index(eos_id)]              # positions after it are unsupervised
            id_lists.append(ids)                           # generated token ids (for diversity)
            t = tokenizer.decode(ids, skip_special_tokens=True)   # -> so cut the response there
            texts.append(t)
            preds.append(extract_pred(t, cfg.dataset_name))

    metrics: Dict[str, float] = {"n_queries": float(n)}
    if cfg.eval_accuracy:
        correct = sum(int(_norm(p) == _norm(g) and p != "") for p, g in zip(preds, gold))
        metrics["accuracy"] = correct / max(n, 1)
    # DIVERSITY on the generated token ids (FLM record_entropy / record_unique_tokens): no LM
    ent, uniq = token_stats(id_lists)
    metrics["entropy"] = ent
    metrics["uniq_tok"] = uniq
    if getattr(cfg, "eval_gen_ppl", True):          # FLUENCY: gen-ppl under gpt2-large
        try:
            metrics["gen_ppl"] = generative_ppl(
                texts, device, getattr(cfg, "eval_ppl_model", "gpt2-large"))
        except Exception as ex:                     # e.g. reference LM not cached offline
            print(f"  [eval] gen_ppl skipped ({type(ex).__name__}: {ex})")
    # crude within-response repetition rate (fraction of responses with a repeated 4-gram)
    def has_rep(row):
        w = row.split()
        grams = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
        return len(grams) != len(set(grams))
    metrics["repeat_rate"] = sum(has_rep(t) for t in texts) / max(len(texts), 1)

    # query-collapse: across DIFFERENT queries, how many DISTINCT responses? 1 = the model
    # ignores the query entirely (the 'every query starts with The total number of' failure).
    metrics["q_distinct"] = len({t for t in texts}) / max(len(texts), 1)
    # how much of the response is shared across queries: mean prefix agreement on the first
    # 8 decoded tokens (1.0 = every query emits the identical opening).
    heads = [" ".join(t.split()[:8]) for t in texts]
    metrics["same_head"] = max(heads.count(h) for h in set(heads)) / max(len(heads), 1)

    # z-collapse probe: SAME query, k_samples different z -> how many DISTINCT responses?
    # 1.0 = the generator ignores z entirely (deterministic query->answer map, the collapse
    # the gold attraction pulls toward); k_samples = z fully alive.
    if cfg.k_samples > 1 and n > 0:
        n_probe = min(4, n)
        qi = q_ids[:n_probe].repeat_interleave(cfg.k_samples, dim=0)
        qm = q_msk[:n_probe].repeat_interleave(cfg.k_samples, dim=0)
        qe = embedder.query_embeds(qi)
        z = gen.sample_z(qi.shape[0], cfg.noise_dim, cfg.temp, device)
        e = gen(qe, qm, z)
        if cfg.sphere_norm:
            e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        tk = embedder.decode(e).reshape(n_probe, cfg.k_samples, -1)
        per_q = [len({tuple(r.tolist()) for r in tk[i]}) for i in range(n_probe)]
        metrics["z_distinct"] = sum(per_q) / len(per_q)      # out of k_samples

    shown = []
    for i in range(min(n_show, n)):
        qtxt = tokenizer.decode(q_ids[i][q_msk[i]].tolist(), skip_special_tokens=True)
        shown.append(f"Q({len(q_msk[i].nonzero())}tok): {qtxt}\n"
                     f"   A(gold={gold[i]!r} pred={preds[i]!r}): {texts[i]!r}")
    if was_training:
        gen.train()
    return metrics, shown
