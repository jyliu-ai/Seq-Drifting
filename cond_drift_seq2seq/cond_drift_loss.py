"""Conditional drift loss = the unconditional teacher-mode repair loss, with the
gen-gen repulsion restricted to samples of the SAME query (block-diagonal).

Attraction (teacher mode), intra-sequence repulsion, and the sphere/geodesic goal
step are identical to the unconditional repair loss. The only change is that
cross-query samples are never pushed apart -- different queries SHOULD have different
responses, so repelling them adds noise. group_size = k_samples; the batch is laid
out [q0 x K, q1 x K, ...].
"""
from typing import Dict, Tuple

import torch

_NEG = -1e4          # logit floor for tokens the decoder may never emit. With tau~0.07 real
                     # logits stay within +-15, so this is effectively -inf for the softmax
                     # while staying finite (a NaN here would silently poison the whole run).


def cosine_ce_loss(gen_emb, target_ids, wte_unit, loss_mask=None, tau: float = 0.07,
                   forbid_mask=None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Cross-entropy over cosine logits (emb . wte / tau), target = the gold token id.

    Why this exists alongside the drift term: the drift loss is an MSE onto ONE gold
    embedding per position, so wherever the true token is multi-modal its minimiser is
    the conditional MEAN of the candidate embeddings. In GPT-2's anisotropic input space
    that mean sits beside the vocabulary centroid, and cosine-NN decodes the centroid to
    ' the'. CE's minimiser is the conditional DISTRIBUTION, so competing modes are not
    averaged into a single point.

    Only SUPERVISED positions with a LEGAL gold token are gathered before the vocabulary
    matmul: that keeps the padded tail and the unlearnable forbidden-gold positions out of
    the loss, and roughly halves the (n_pos, V) logits. `ce_skip` reports the fraction of
    supervised positions dropped for a forbidden gold.
    """
    G, T, H = gen_emb.shape
    e = torch.nn.functional.normalize(gen_emb.float(), dim=-1).reshape(G * T, H)
    tgt = target_ids.reshape(G * T).to(e.device)
    sel = (loss_mask.reshape(G * T).bool().to(e.device) if loss_mask is not None
           else torch.ones(G * T, dtype=torch.bool, device=e.device))
    n_sup = int(sel.sum())
    if forbid_mask is not None:
        # A gold token that is itself forbidden is UNLEARNABLE under the mask: its logit is
        # pinned at _NEG, so the position contributes ~1e4 to the loss and a gradient with no
        # attracting target (masked_fill blocks the gold term, leaving only repulsion). WMT14's
        # English targets hit this on ~0.6% of positions (BPE byte fragments of accented
        # characters), which was enough to dominate the mean CE. Drop those positions.
        sel = sel & ~forbid_mask.to(e.device)[tgt]
    if not bool(sel.any()):
        zero = gen_emb.sum() * 0.0
        return zero, {"ce": 0.0, "ce_acc": 0.0, "ce_skip": 1.0}
    skip = 1.0 - float(int(sel.sum())) / max(n_sup, 1)
    e, tgt = e[sel], tgt[sel]

    # Matmul in the vocabulary's dtype (bf16), softmax in fp32: casting the (V, H) matrix
    # up to fp32 on every microbatch costs more than it buys, while a 50k-way softmax in
    # bf16 does lose accuracy. This is the usual LM-head recipe.
    w = wte_unit
    logits = (e.to(w.dtype) @ w.t()).float() / tau                 # (n_sup, V)
    if forbid_mask is not None:
        # Suppress the tokens the decoder may never emit, so probability mass is not spent
        # on them. Safe to use a finite floor here: positions whose GOLD is forbidden were
        # already dropped above, so no target is ever sitting at _NEG.
        logits = logits.masked_fill(forbid_mask.to(logits.device).unsqueeze(0), _NEG)
    ce = torch.nn.functional.cross_entropy(logits, tgt)
    with torch.no_grad():
        acc = float((logits.argmax(dim=-1) == tgt).float().mean())
    return ce, {"ce": float(ce.detach()), "ce_acc": acc, "ce_skip": skip}


def cond_repair_loss(gen_emb, repair_wte, repair_valid, implausible, group_size: int,
                     temp: float = 0.3, repel: float = 5.0, repel_perpos: bool = True,
                     repel_intra: float = 5.0, repel_abs: bool = False, abs_scale: float = 1.0,
                     free_prefix: int = 0, sphere: bool = True, sphere_step: str = "geodesic",
                     sphere_geo_max: float = 1.5708, temp_list=(),
                     gold_wte=None, gold_mask=None, gold_weight: float = 0.0,
                     loss_mask=None, teacher_weight: float = 1.0, repel_block: bool = True,
                     div_mask: str = "none",
                     pos_teacher_decay: float = 1.0, pos_gold_boost: float = 1.0,
                     eos_wte=None, eos_repel: float = 0.0, eos_repel_tail: int = 0,
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """gen_emb (G, T, H) grad; repair_wte (G, T, n_pos, H); repair_valid (G, T, n_pos).
    G = Q * group_size (block order). Teacher-mode attraction toward the Qwen support
    set; every position pulled (no own-token snap)."""
    G, T, H = gen_emb.shape
    old = gen_emb.detach().float()

    # --- divergence gate ------------------------------------------------------------
    # `implausible` flags positions where the teacher gives the student's OWN token
    # p <= prob_thresh. Because the teacher scores position r conditioned on the
    # student's tokens 0..r-1, the first flag is where the signal stops describing the
    # reference and starts describing a coherent continuation of the student's drift.
    #   prefix / first -- drop ALL force past the flag (or keep only the flag itself)
    #   teacher        -- drop only the TEACHER term there, keep the gold pull: gold is a
    #                     clean function of (query, r), while the teacher term is
    #                     conditioned on a prefix the student cannot observe at inference.
    div_keep = None
    div_first = None
    if div_mask != "none" and implausible is not None and implausible.numel():
        sup = (loss_mask.bool() if loss_mask is not None
               else torch.ones(G, T, dtype=torch.bool, device=old.device))
        bad = implausible.bool().to(old.device) & sup
        has = bad.any(dim=1)
        # float().argmax() takes the FIRST maximum, so this is the first flagged index.
        # Rows with no flag get T, which keeps the whole response under "<= first".
        div_first = torch.where(
            has, bad.float().argmax(dim=1),
            torch.full((G,), T, dtype=torch.long, device=old.device))
        ar = torch.arange(T, device=old.device).unsqueeze(0)
        if div_mask == "first":
            # Only the flagged position itself. A row the teacher never flags has nothing
            # to localise to, so it keeps its full supervision instead of dropping out.
            keep = torch.where(has.unsqueeze(1), ar == div_first.unsqueeze(1), sup)
        else:
            keep = ar <= div_first.unsqueeze(1)
        div_keep = (keep & sup).float()
    with torch.no_grad():
        # --- attraction: mode-seeking toward the support set (per-position std temp) ---
        # NOTE memory: rw is (G, T, n_pos, H) and H (~2560 for Qwen) is large, so the distance
        # and centroid are computed WITHOUT materialising any (G, T, n_pos, H) temporary:
        # dist^2 = |old|^2 + |rw|^2 - 2 old.rw (einsum), centroid = einsum(aff, rw).
        rw = repair_wte.float()                                        # (G, T, n_pos, H)
        rw2 = (rw * rw).sum(dim=-1)                                    # (G, T, n_pos)
        old2 = (old * old).sum(dim=-1, keepdim=True)                  # (G, T, 1)
        cross = torch.einsum("gth,gtph->gtp", old, rw)               # (G, T, n_pos)
        dist = torch.sqrt((old2 + rw2 - 2.0 * cross).clamp_min(0.0) + 1e-12)   # (G, T, n_pos)
        scale = dist[repair_valid].mean() if repair_valid.any() else dist.new_tensor(1.0)
        vmask = repair_valid.float()
        cnt = vmask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_r = (dist * vmask).sum(dim=-1, keepdim=True) / cnt
        tscale = (((dist - mean_r) * vmask) ** 2).sum(dim=-1, keepdim=True).div(cnt).sqrt()
        tscale = tscale.clamp_min(1e-6)                                # (G, T, 1) per-position spread
        temps = list(temp_list) if len(temp_list) > 0 else [temp]
        aff = None
        for tp in temps:
            lg = (-dist / (tscale * tp)).masked_fill(~repair_valid, float("-inf"))
            a = torch.nan_to_num(torch.softmax(lg, dim=-1))
            aff = a if aff is None else aff + a
        aff = aff / len(temps)
        centroid = torch.einsum("gtp,gtph->gth", aff, rw)             # (G, T, H) no big temporary
        has_cand = repair_valid.any(dim=-1).unsqueeze(-1)
        target = torch.where(has_cand, centroid, old)                 # no candidate -> stay
        # teacher_weight scales (or disables) the teacher-support attraction. On an
        # accuracy-scored task the teacher support (built on the model's OWN garbage prefix)
        # only points at "locally plausible" tokens, not the correct ones, and it FIGHTS the
        # gold pull -- with gold_weight 5 + teacher + repel the equilibrium sat ~39deg off the
        # gold and decoded to junk. teacher_weight=0 => pure gold supervision (fixed point IS
        # the gold response).
        attract = teacher_weight * (target - old)
        # Per-position teacher decay: weight teacher contribution linearly from
        # 1.0 at position 0 to pos_teacher_decay at position T-1. This reduces
        # the teacher's influence at the tail where its support is conditioned on
        # the student's own (potentially wrong) prefix and thus less trustworthy.
        if pos_teacher_decay != 1.0:
            pos = torch.linspace(1.0, pos_teacher_decay, T,
                                 device=old.device).view(1, T, 1)
            attract = attract * pos
        if div_mask == "teacher" and div_keep is not None:
            # Past the first flagged token the support set is conditioned on the student's
            # own drift, so it pulls toward a coherent continuation of the wrong sentence.
            # Gated HERE, before the gold pull is added, so gold keeps acting everywhere.
            attract = attract * div_keep.unsqueeze(-1)

        # --- (0) GOLD attraction: position-aligned pull toward the real response ---
        # Teacher support only says "plausible here"; it carries no notion of the CORRECT
        # answer, so on an accuracy-scored task it can never reach the right tokens. The
        # gold response is the positive that carries correctness. Applied only where the
        # gold has real tokens (mask), so padded tail positions keep teacher-only force.
        gold_frac = 0.0
        if gold_weight > 0 and gold_wte is not None:
            gpull = gold_wte.float() - old                            # (G, T, H)
            if gold_mask is not None:
                gpull = gpull * gold_mask.unsqueeze(-1).to(gpull.dtype)
            # Per-position gold boost: weight gold contribution linearly from
            # 1.0 at position 0 to pos_gold_boost at position T-1. Compensates
            # for the teacher becoming less reliable in the tail.
            if pos_gold_boost != 1.0:
                pos_g = torch.linspace(1.0, pos_gold_boost, T,
                                       device=old.device).view(1, T, 1)
                gpull = gpull * pos_g
            attract = attract + gold_weight * gpull
            gold_frac = float(torch.sqrt(torch.clamp((gold_weight * gpull) ** 2, min=0).mean()))

        a_rms = torch.sqrt(torch.clamp((attract ** 2).mean(), min=1e-8))

        # --- (0b) EOS repulsion: push content positions away from the EOS embedding ----
        # For the last eos_repel_tail positions before the gold EOS, add a force that
        # pushes the output embedding away from the EOS token's embedding. Restricting
        # to the tail (rather than all content positions) avoids disrupting early
        # positions that are learning correctly — only the positions near the EOS
        # boundary need this nudge.
        if eos_repel > 0 and eos_wte is not None and gold_mask is not None:
            eos_pos_idx = gold_mask.long().sum(dim=1, keepdim=True) - 1  # (G, 1) 0-indexed EOS
            pos_idx = torch.arange(T, device=old.device).unsqueeze(0)    # (1, T)
            # tail window: [eos_pos - eos_repel_tail, eos_pos); if eos_repel_tail==0 use all
            if eos_repel_tail > 0:
                tail_start = (eos_pos_idx - eos_repel_tail).clamp_min(0)
                pre_eos_mask = (pos_idx >= tail_start) & (pos_idx < eos_pos_idx)
            else:
                pre_eos_mask = pos_idx < eos_pos_idx                     # all content positions
            eos_emb = eos_wte.float().view(1, 1, -1)                     # (1, 1, H)
            eos_diff = old - eos_emb                                     # (G, T, H) away from EOS
            eos_diff = eos_diff * pre_eos_mask.unsqueeze(-1).to(eos_diff.dtype)
            er_rms = torch.sqrt(torch.clamp((eos_diff ** 2).mean(), min=1e-8))
            attract = attract + eos_repel * (a_rms / er_rms) * eos_diff
            # recompute a_rms after adding the EOS repulsion
            a_rms = torch.sqrt(torch.clamp((attract ** 2).mean(), min=1e-8))

        # --- (1) gen-gen repulsion ---
        # repel_block=True (default): BLOCK-DIAGONAL -- only the K samples of the SAME prefix
        #   repel each other (different prefixes should differ on their own).
        # repel_block=False: WHOLE-BATCH -- every sample repels every other, like the
        #   unconditional version (whose anti-collapse came from ALL 2048 samples repelling).
        #   Broader pressure; can catch "every prefix -> the same soup", which block cannot.
        min_group = group_size if repel_block else 2
        if repel > 0 and G > 1 and G >= min_group:
            eye = torch.eye(G, dtype=torch.bool, device=old.device)
            if repel_block:
                gid = (torch.arange(G, device=old.device) // group_size)
                offd = (gid[:, None] == gid[None, :]) & ~eye         # within-prefix, non-self
            else:
                offd = ~eye                                          # whole batch, non-self
            if repel_perpos:
                op = old.permute(1, 0, 2)                             # (T, G, H)
                dgg = torch.cdist(op, op)                             # (T, G, G)
                ob = offd.unsqueeze(0)                                # (1, G, G)
                srep = ((dgg * ob).sum(dim=(-1, -2))
                        / ob.sum(dim=(-1, -2)).clamp_min(1.0)).clamp_min(1e-6).view(T, 1, 1)
                logits = (-dgg / (srep * temp)).masked_fill(~ob, float("-inf"))
                aff_rep = torch.nan_to_num(torch.softmax(logits, dim=-1))
                rep = (op - aff_rep @ op).permute(1, 0, 2)            # (G, T, H)
            else:
                flat = old.reshape(G, -1)                            # (G, T*H)
                dgg = torch.cdist(flat, flat)                        # (G, G)
                srep = dgg[offd].mean().clamp_min(1e-6) if offd.any() else dgg.new_tensor(1.0)
                logits = (-dgg / (srep * temp)).masked_fill(~offd, float("-inf"))
                aff_rep = torch.nan_to_num(torch.softmax(logits, dim=1))
                rep = (flat - aff_rep @ flat).reshape(G, T, H)
            r_rms = torch.sqrt(torch.clamp((rep ** 2).mean(), min=1e-8))
            coef = (abs_scale / r_rms) if repel_abs else (a_rms / r_rms)
            attract = attract + repel * coef * rep

        # --- (2) intra-sequence repulsion (anti token-repeat), per response ---
        if repel_intra > 0 and T > 1:
            dpp = torch.cdist(old, old)                               # (G, T, T)
            eyeT = torch.eye(T, dtype=torch.bool, device=old.device).unsqueeze(0)
            if loss_mask is not None:
                # Mask out pad positions from both axes: a pad token neither attracts
                # nor repels any other token. This prevents the ~42 near-identical pad
                # embeddings from distorting the scale estimate and the softmax.
                cmask = loss_mask.bool().unsqueeze(2) & loss_mask.bool().unsqueeze(1)  # (G,T,T)
                valid_offt = ~eyeT & cmask
            else:
                valid_offt = ~eyeT
            sp = ((dpp * valid_offt).sum(dim=(-1, -2))
                  / valid_offt.sum(dim=(-1, -2)).clamp_min(1.0)).clamp_min(1e-6).view(G, 1, 1)
            logits = (-dpp / (sp * temp)).masked_fill(~valid_offt, float("-inf"))
            affp = torch.nan_to_num(torch.softmax(logits, dim=-1))
            repi = old - affp @ old
            if loss_mask is not None:                                # no force on the pad tail
                repi = repi * loss_mask.unsqueeze(-1).to(repi.dtype)
            ri_rms = torch.sqrt(torch.clamp((repi ** 2).mean(), min=1e-8))
            coef_i = (abs_scale / ri_rms) if repel_abs else (a_rms / ri_rms)
            attract = attract + repel_intra * coef_i * repi

        if free_prefix > 0:
            attract[:, :free_prefix, :] = 0.0
        if div_mask in ("prefix", "first") and div_keep is not None:
            # Drop ALL force past the flag -- gold and teacher alike. div_keep also becomes
            # the loss denominator below, so the dropped positions do not dilute the mean
            # into looking better while contributing nothing.
            attract = attract * div_keep.unsqueeze(-1)
        if loss_mask is not None:
            # positions past the gold's end-of-text token get NO target at all (not even the
            # teacher's): leaving them to the teacher conditioned on the model's own garbage
            # prefix is a self-referential loop, and it is what produced the junk tail.
            attract = attract * loss_mask.unsqueeze(-1).to(attract.dtype)

        # --- goal on the sphere (geodesic exp-map by default) ---
        if not sphere:
            goal = old + attract
        elif sphere_step == "proj":
            goal = old + attract
            goal = goal / goal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            ftan = attract - (attract * old).sum(dim=-1, keepdim=True) * old
            fn = ftan.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            if sphere_step == "geodesic":
                a = fn.clamp(max=sphere_geo_max)
                goal = torch.cos(a) * old + torch.sin(a) * (ftan / fn)
            else:                                                     # retract
                goal = old + ftan
                goal = goal / goal.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # Diagnostics only: measure the actual generator output against the aligned gold
    # embedding. These values are detached and never affect the drifting objective.
    gold_rmse = 0.0
    gold_l2 = 0.0
    gold_cos = 0.0
    if gold_wte is not None:
        pred = gen_emb.detach().float()
        gold = gold_wte.float()
        gold_diff2 = (pred - gold).pow(2).sum(dim=-1)                # (G, T)
        gold_cosines = torch.nn.functional.cosine_similarity(
            pred, gold, dim=-1, eps=1e-8)
        if gold_mask is None:
            denom = torch.tensor(float(G * T), device=pred.device).clamp_min(1.0)
            gold_rmse = float(torch.sqrt(gold_diff2.sum() / (denom * H)))
            gold_l2 = float(torch.sqrt(gold_diff2.sum() / denom))
            gold_cos = float(gold_cosines.sum() / denom)
        else:
            mask = gold_mask.to(device=pred.device, dtype=pred.dtype)
            denom = mask.sum().clamp_min(1.0)
            gold_rmse = float(torch.sqrt((gold_diff2 * mask).sum() / (denom * H)))
            gold_l2 = float(torch.sqrt((gold_diff2 * mask).sum() / denom))
            gold_cos = float((gold_cosines * mask).sum() / denom)

    diff2 = (gen_emb.float() - goal) ** 2                             # (G, T, H)
    # Denominator. Under prefix/first the dropped positions have goal == old, so they
    # contribute exactly 0 -- counting them would shrink the reported loss without any
    # gradient behind it, and the shrink would grow as the model improved. Averaging over
    # the positions that actually carry force keeps the number comparable across steps.
    denom_mask = None
    if div_mask in ("prefix", "first") and div_keep is not None:
        denom_mask = div_keep
    elif loss_mask is not None:
        denom_mask = loss_mask.to(diff2.dtype)
    if denom_mask is None:
        loss = diff2.sum(dim=-1).mean()      # mean squared L2 distance per position
    else:
        m = denom_mask.unsqueeze(-1).to(diff2.dtype)
        loss = (diff2 * m).sum() / m.sum().clamp_min(1.0)             # sum H, mean positions
    info = {"scale": float(scale),
            "gold_f": gold_frac,
            "gold_rmse": gold_rmse,
            "gold_l2": gold_l2,
            "gold_cos": gold_cos,
            "n_implaus": float(implausible.float().sum(dim=1).mean())}
    if div_first is not None:
        # Mean first-divergence position, and the fraction of supervised positions still
        # trained. div_pos is the useful progress signal: as the student improves the first
        # flag moves right, so this should climb even while the loss is flat.
        info["div_pos"] = float(div_first.float().mean())
        sup_n = (loss_mask.float().sum() if loss_mask is not None
                 else torch.tensor(float(G * T), device=old.device))
        info["div_keep"] = float(div_keep.sum() / sup_n.clamp_min(1.0))
    return loss, info
