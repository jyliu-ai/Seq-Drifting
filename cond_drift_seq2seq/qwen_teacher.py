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

from .qwen_features import load_qwen


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
    model, tokenizer = load_qwen(
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
