"""Conditional causal-LM teacher: score [query ++ response prefix] and return, for every
RESPONSE position, the support set (top-n_pos plausible next tokens) + a validity
mask + the implausibility of the gen's own token.

Same support-set / no-repeat logic as the unconditional gpt2 teacher, but:
  * the context is the real (left-padded) query, not a single BOS;
  * logits are read at the response-predicting positions [Lq-1 .. Lq+Lr-2];
  * the LM head is applied in ROW CHUNKS (never materialise
    the full (G, Lr, V) logits) and only top-k + logsumexp are kept.
"""
import torch
from functools import lru_cache

from models.embeddings import load_teacher
from common.config import Seq2SeqDriftConfig


_CROP_NOTICES = set()


def _max_positions(model):
    for name in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(model.config, name, None)
        if value is not None:
            return int(value)
    return None


def fit_teacher_context(query_ids, query_mask, resp_len, model, pad_id):
    """Keep the first source tokens that fit before the response in a causal LM.

    The student still sees the complete condition. This only bounds the teacher input,
    e.g. XSum uses 1024 source slots while GPT-2 can score at most 960 + 64 tokens.
    Inputs are expected to be left-padded with a contiguous valid suffix.
    """
    max_positions = _max_positions(model)
    if max_positions is None or query_ids.shape[1] + resp_len <= max_positions:
        return query_ids, query_mask
    max_query = max_positions - resp_len
    if max_query <= 0:
        raise ValueError(
            f"teacher context {max_positions} is not longer than response length {resp_len}")

    batch, source_width = query_ids.shape
    lengths = query_mask.long().sum(dim=1)
    take = lengths.clamp(max=max_query)
    dst = torch.arange(max_query, device=query_ids.device).unsqueeze(0).expand(batch, -1)
    dst_start = max_query - take
    keep = dst >= dst_start.unsqueeze(1)
    src_start = source_width - lengths
    src_idx = src_start.unsqueeze(1) + dst - dst_start.unsqueeze(1)
    src_idx = src_idx.clamp(min=0, max=source_width - 1)
    gathered = query_ids.gather(1, src_idx)
    cropped_ids = torch.where(keep, gathered, torch.full_like(gathered, pad_id))
    cropped_mask = keep.bool()

    notice = (source_width, max_query, resp_len, max_positions)
    if notice not in _CROP_NOTICES:
        print(f"[teacher] condition cropped {source_width}->{max_query} tokens so "
              f"condition+response={max_positions} fits the teacher context")
        _CROP_NOTICES.add(notice)
    return cropped_ids, cropped_mask


def _no_repeat_banned(resp_tokens, k, W, device):
    """resp_tokens (G, Lr) -> (M, 2) long pairs (flat_row, token) to forbid, where
    flat_row = g*Lr + pos: token completes a k-gram already present in response g's
    prefix. Empty (0,2) if none."""
    G, Lr = resp_tokens.shape
    offs = torch.arange(-(k - 1), 0, device=device)
    pairs = []
    for pp in range(k - 1, Lr):
        ctx = resp_tokens[:, pp - k + 1:pp]                    # (G, k-1)
        lo = max(k - 1, pp - W)
        if pp - lo <= 0:
            continue
        J = torch.arange(lo, pp, device=device)               # (Nj,)
        base = J.view(-1, 1) + offs.view(1, -1)               # (Nj, k-1)
        ctx_earlier = resp_tokens[:, base]                    # (G, Nj, k-1)
        match = (ctx_earlier == ctx.unsqueeze(1)).all(dim=-1) # (G, Nj)
        if match.any():
            gm = match.nonzero(as_tuple=False)
            gs, ms = gm[:, 0], gm[:, 1]
            banned = resp_tokens[gs, J[ms]]                   # (Mm,)
            rowg = gs * Lr + pp
            pairs.append(torch.stack([rowg, banned], dim=1))
    if pairs:
        return torch.cat(pairs, dim=0)
    return torch.zeros((0, 2), dtype=torch.long, device=device)


@torch.no_grad()
def build_repairs_cond(query_ids, query_mask, resp_tokens, cfg, device,
                       forbid_mask=None, row_chunk: int = 4096):
    """query_ids (Q, Lq), query_mask (Q, Lq) bool, resp_tokens (G, Lr) with G=Q*K
    (block order: query q owns rows [q*K:(q+1)*K]). Returns:
        support (G, Lr, n_pos) long, valid (G, Lr, n_pos) bool, implausible (G, Lr) bool.
    """
    model, tokenizer = load_teacher(
        cfg.teacher_model, str(device), "bf16" if cfg.use_bf16 else "fp32")
    Q, Lq = query_ids.shape
    G, Lr = resp_tokens.shape
    K = G // Q
    n_pos, thresh = cfg.n_pos, cfg.prob_thresh
    query_ids = query_ids.to(device).long()
    query_mask = query_mask.to(device).bool()
    resp_tokens = resp_tokens.to(device).long()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if isinstance(cfg, Seq2SeqDriftConfig):
        query_ids, query_mask = fit_teacher_context(
            query_ids, query_mask, Lr, model, pad_id)
    Lq = query_ids.shape[1]

    qids_g = query_ids.repeat_interleave(K, dim=0)            # (G, Lq)
    qmsk_g = query_mask.repeat_interleave(K, dim=0)           # (G, Lq)
    input_ids = torch.cat([qids_g, resp_tokens], dim=1)       # (G, Lq+Lr)
    attn = torch.cat([qmsk_g, torch.ones(G, Lr, dtype=torch.bool, device=device)], dim=1)
    pos_ids = (attn.long().cumsum(dim=1) - 1).clamp_min(0)    # correct RoPE ids under left-pad

    # model-agnostic: base_model is the transformer (GPT-2 -> .transformer, Qwen -> .model),
    # get_output_embeddings() is the LM head. Works for a GPT-2 teacher (768-dim, 50k vocab)
    # as well as Qwen, so the manifold + reference LM can match GPT-2-based baselines.
    hidden = model.base_model(input_ids=input_ids, attention_mask=attn,
                              position_ids=pos_ids).last_hidden_state   # (G, Lq+Lr, H)
    hidden_resp = hidden[:, Lq - 1:Lq + Lr - 1, :].contiguous()     # (G, Lr, H) predicts resp[r]
    Hd = hidden_resp.shape[-1]
    rows = hidden_resp.reshape(G * Lr, Hd)                    # (R, H)
    R = rows.shape[0]

    lm_head = model.get_output_embeddings()                  # nn.Linear (V, H)
    lm_w = lm_head.weight                                     # (V, H)
    lm_b = getattr(lm_head, "bias", None)
    V = lm_w.shape[0]
    own_flat = resp_tokens.reshape(R)                         # (R,) gen's own next token

    banned = None
    if cfg.no_repeat_ngram >= 2 and Lr > cfg.no_repeat_ngram:
        W = cfg.no_repeat_window if cfg.no_repeat_window > 0 else Lr
        banned = _no_repeat_banned(resp_tokens, int(cfg.no_repeat_ngram), W, device)  # (M,2)

    sup = torch.zeros(R, n_pos, dtype=torch.long, device=device)
    sup_p = torch.zeros(R, n_pos, dtype=torch.float, device=device)
    gen_p = torch.zeros(R, dtype=torch.float, device=device)
    for s in range(0, R, row_chunk):
        e = min(s + row_chunk, R)
        lg = torch.nn.functional.linear(rows[s:e].float(), lm_w.float(),
                                        None if lm_b is None else lm_b.float())  # (c, V)
        if forbid_mask is not None:
            lg = lg.masked_fill(forbid_mask.to(device).bool().unsqueeze(0), float("-inf"))
        if banned is not None and banned.numel():
            m = (banned[:, 0] >= s) & (banned[:, 0] < e)
            if m.any():
                lg[banned[m, 0] - s, banned[m, 1]] = float("-inf")
        lse = torch.logsumexp(lg, dim=-1, keepdim=True)       # (c,1) uses the true full distribution
        topv, topi = lg.topk(n_pos, dim=-1)                   # (c, n_pos)
        sup[s:e] = topi
        sup_p[s:e] = torch.exp(topv - lse)
        gen_lg = lg.gather(1, own_flat[s:e].unsqueeze(1)).squeeze(1)
        gen_p[s:e] = torch.exp(gen_lg - lse.squeeze(1))

    support = sup.reshape(G, Lr, n_pos)
    supp_prob = sup_p.reshape(G, Lr, n_pos)
    implausible = (gen_p <= thresh).reshape(G, Lr)
    if cfg.support == "nucleus":
        before = supp_prob.cumsum(dim=-1) - supp_prob
        valid = before < cfg.nucleus_p
        valid[..., 0] = True
    else:
        valid = supp_prob > thresh
    return support, valid, implausible



@lru_cache(maxsize=2)
def _get_teacher(model_name: str, device_str: str):
    from transformers import GPT2LMHeadModel
    print(f"[gpt2-teacher] loading {model_name} ...")
    dev = torch.device(device_str)
    # fp16 on GPU: the teacher is only used to rank top-k tokens and threshold at
    # prob_thresh, so half precision is ample, and it halves both the (G, T, V) logits
    # /softmax footprint and the lm_head matmul cost that runs every training step.
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    m = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=dtype).to(dev)
    m.eval()
    print(f"[gpt2-teacher] loaded ({dtype}).")
    return m


@torch.no_grad()
def build_positives(gen_tokens: torch.Tensor, n_pos: int, topk: int,
                    bos_id: int, device, model_name: str = "gpt2"):
    """gen_tokens (G, T) -> (pos_tokens (G*n_pos, T), p_star (G,)).  [continuation method]

    pos_tokens: plausible prefix + sampled top-k token at p* + GPT-2 continuation.
    p_star: per-generation first-divergence position (gen token not in top-k).
    """
    gpt2 = _get_teacher(model_name, str(device))
    gen_tokens = gen_tokens.to(device).to(torch.long)
    G, T = gen_tokens.shape
    bos = torch.full((G, 1), bos_id, dtype=torch.long, device=device)

    inp = torch.cat([bos, gen_tokens], dim=1)                 # (G, T+1)
    logits = gpt2(input_ids=inp).logits[:, :T, :]             # (G, T, V)
    topk_idx = logits.topk(topk, dim=-1).indices              # (G, T, topk)
    plausible = (topk_idx == gen_tokens.unsqueeze(-1)).any(dim=-1)     # (G, T)

    not_plaus = ~plausible
    has_bad = not_plaus.any(dim=1)
    first_bad = not_plaus.float().argmax(dim=1)
    p_star = torch.where(has_bad, first_bad,
                         torch.full((G,), T - 1, device=device, dtype=torch.long))  # (G,)

    rows = torch.arange(G, device=device)
    logits_p = logits[rows, p_star]                           # (G, V)
    tv, ti = logits_p.topk(topk, dim=-1)                      # (G, topk)
    samp = torch.multinomial(torch.softmax(tv, dim=-1), n_pos, replacement=True)
    sampled = torch.gather(ti, 1, samp)                       # (G, n_pos)

    R = G * n_pos
    canvas = gen_tokens.repeat_interleave(n_pos, dim=0).clone()        # (R, T)
    pstar_r = p_star.repeat_interleave(n_pos)                          # (R,)
    canvas[torch.arange(R, device=device), pstar_r] = sampled.reshape(R)
    bos_r = torch.full((R, 1), bos_id, dtype=torch.long, device=device)

    transformer, lm_head = gpt2.transformer, gpt2.lm_head
    min_ps = int(pstar_r.min().item())
    for s in range(min_ps + 1, T):                            # fill positions > p* (last-token logits only)
        inp_s = torch.cat([bos_r, canvas[:, :s]], dim=1)      # (R, s+1)
        h_last = transformer(input_ids=inp_s).last_hidden_state[:, -1, :]
        lg = lm_head(h_last)
        tvs, tis = lg.topk(topk, dim=-1)
        nxt = torch.gather(tis, 1, torch.multinomial(torch.softmax(tvs, dim=-1), 1)).squeeze(-1)
        canvas[:, s] = torch.where(s > pstar_r, nxt, canvas[:, s])

    return canvas, p_star


@torch.no_grad()
def build_repairs(gen_emb: torch.Tensor, gen_tokens: torch.Tensor, n_pos: int,
                  prob_thresh: float, bos_id: int, device, model_name: str = "gpt2",
                  illegal_mask: torch.Tensor = None, target_mode: str = "self",
                  support: str = "thresh", nucleus_p: float = 0.99,
                  no_repeat_ngram: int = 0, no_repeat_window: int = 0):
    """(gen_emb (G,T,D), gen_tokens (G,T)) -> (cand_tokens (G,T,n_pos), valid (G,T,n_pos),
    implausible (G,T)).  One teacher-forced pass; positions checked independently.

    implausible[g, p] = GPT-2 prob of gen_tokens[g, p] (given the gen prefix) <= prob_thresh.

    cand_tokens / valid depend on target_mode:
      "self":    among the PLAUSIBLE tokens (prob > thresh, illegal excluded, OWN token
                 excluded) at position p, the n_pos NEAREST to gen_emb[g, p] (wte distance).
      "teacher": the GPT-2 SUPPORT SET = the n_pos most PROBABLE tokens (illegal excluded;
                 own token kept iff it is itself probable); valid = those with prob > thresh.
                 The mode-seeking distance affinity over this set is applied in the loss.
    valid masks padded / sub-threshold candidates (zero affinity in the loss).
    """
    gpt2 = _get_teacher(model_name, str(device))
    gen_tokens = gen_tokens.to(device).to(torch.long)
    gen_emb = gen_emb.to(device).float()
    G, T = gen_tokens.shape
    D = gen_emb.shape[-1]
    bos = torch.full((G, 1), bos_id, dtype=torch.long, device=device)

    # input is [BOS, g_0 .. g_{T-2}] (length T): position p predicts g_p. Dropping the
    # last gen token (whose logit was sliced off anyway) makes this fit a teacher whose
    # context is exactly T -- e.g. a block_size=128 LM1B teacher at seq_len 128 -- while
    # giving logits identical to the old [BOS, g_0..g_{T-1}] form for positions 0..T-1.
    inp = torch.cat([bos, gen_tokens[:, :-1]], dim=1)         # (G, T)
    logits = gpt2(input_ids=inp).logits                       # (G, T, V): logits[:, p] -> content[p]
    probs = torch.softmax(logits, dim=-1)                     # (G, T, V)
    gen_prob = probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)   # (G, T)
    implausible = gen_prob <= prob_thresh                     # (G, T)

    if target_mode == "teacher":
        # support set = top-n_pos by probability (illegal excluded; own token kept iff probable)
        p = probs
        if illegal_mask is not None:
            p = p.masked_fill(illegal_mask.to(device).bool().view(1, 1, -1), 0.0)
        if no_repeat_ngram >= 2 and T > no_repeat_ngram:
            # zero the probability of any token that would complete a k-gram already present
            # in this generation's own prefix, so the support set can't offer the repeat token
            # as a target (breaks the repetition feedback loop). Bans the k-gram, not the token.
            p = p.clone()
            k = int(no_repeat_ngram)
            W = no_repeat_window if no_repeat_window > 0 else T
            offs = torch.arange(-(k - 1), 0, device=device)               # (k-1,) context offsets
            for pp in range(k - 1, T):
                ctx = gen_tokens[:, pp - k + 1:pp]                        # (G, k-1) context at pp
                lo = max(k - 1, pp - W)
                if pp - lo <= 0:
                    continue
                J = torch.arange(lo, pp, device=device)                   # (Nj,) earlier positions
                base = J.view(-1, 1) + offs.view(1, -1)                   # (Nj, k-1) earlier ctx idx
                ctx_earlier = gen_tokens[:, base]                        # (G, Nj, k-1)
                match = (ctx_earlier == ctx.unsqueeze(1)).all(dim=-1)     # (G, Nj) same (k-1)-context
                if match.any():
                    gm = match.nonzero(as_tuple=False)                   # (M, 2): (g, m)
                    gs, ms = gm[:, 0], gm[:, 1]
                    banned = gen_tokens[gs, J[ms]]                       # (M,) token that followed
                    p[gs, pp, banned] = 0.0
        topp, topi = p.topk(n_pos, dim=-1)                    # (G, T, n_pos) by prob (descending)
        if support == "nucleus":                              # top-p: keep prefix reaching nucleus_p
            before = topp.cumsum(dim=-1) - topp               # cumulative prob BEFORE each token
            valid = before < nucleus_p                        # smallest prefix that reaches nucleus_p
            valid[..., 0] = True                              # always keep the argmax (non-empty)
        else:                                                 # fixed absolute floor
            valid = topp > prob_thresh
        return topi, valid, implausible

    plaus = probs > prob_thresh                               # (G, T, V) plausible set
    plaus.scatter_(-1, probs.argmax(dim=-1, keepdim=True), True)        # keep argmax (non-empty)

    # squared distance from each gen embedding to EVERY token's wte; restrict to plausible;
    # take the n_pos nearest plausible tokens
    wte = gpt2.transformer.wte.weight.float()                 # (V, D) (same gpt2 wte as embedder)
    df = gen_emb.reshape(G * T, D)                            # (G*T, D)
    dist2 = ((df * df).sum(-1, keepdim=True) + (wte * wte).sum(-1).unsqueeze(0)
             - 2.0 * (df @ wte.t()))                          # (G*T, V) squared distance
    dist2 = dist2.masked_fill(~plaus.reshape(G * T, -1), float("inf"))
    if illegal_mask is not None:                                  # never repair toward junk tokens
        dist2 = dist2.masked_fill(illegal_mask.to(dist2.device).bool().unsqueeze(0), float("inf"))
    dist2.scatter_(-1, gen_tokens.reshape(-1, 1), float("inf"))   # never repair toward own token
    topd, topi = dist2.topk(n_pos, dim=-1, largest=False)     # nearest plausible (G*T, n_pos)
    valid = torch.isfinite(topd)                             # genuine candidate (vs padding)
    repair = torch.where(valid, topi, topi[:, :1].expand_as(topi))   # pad with nearest (masked later)
    return repair.reshape(G, T, n_pos), valid.reshape(G, T, n_pos), implausible
