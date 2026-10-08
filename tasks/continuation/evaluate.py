"""Evaluate generated continuations on fluency and diversity."""
from typing import Dict, List, Tuple

from common.metrics import generative_ppl, token_stats
import torch








@torch.no_grad()
def run_eval(gen, embedder, dataset, cfg, tokenizer, device,
             n_queries: int, n_show: int = 6) -> Tuple[Dict[str, float], List[str]]:
    was_training = gen.training
    gen.eval()
    n = min(n_queries, len(dataset))
    q_ids = dataset.query_ids[:n].to(device)
    q_msk = dataset.query_mask[:n].to(device)
    gold = dataset.gold[:n]

    texts, id_lists = [], []
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

    metrics: Dict[str, float] = {"n_queries": float(n)}
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
                     f"   Reference: {gold[i]!r}\n   Generated: {texts[i]!r}")
    if was_training:
        gen.train()
    return metrics, shown
