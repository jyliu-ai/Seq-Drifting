"""Drift loss on 2-gram token embeddings, min-distance variant (per user spec).

Each sample is a set of M = T-1 two-gram embeddings (R^{2D}). The distance
between two samples is the MIN over all 2-gram pairs -- a generated sample is
"close" to a real one if just ONE of its 2-grams is close. The drift force pulls
the generated sample's matched (argmin) 2-gram toward the target's matched
2-gram, and repels generated samples from each other the same way.

The affinity / multi-R / symmetric-softmax / scaling structure mirrors
drifting-main/drift_loss.py; only the distance (min-over-2-grams) and the force
routing (onto the matched 2-gram) are text-specific.
"""
from typing import Dict, Optional, Tuple

import torch


def _affinity(logits: "torch.Tensor", sinkhorn_iters: int,
              active: "torch.Tensor" = None) -> "torch.Tensor":
    """Turn (G, K) logits = -dist/eps into an affinity matrix.

    sinkhorn_iters<=0: one symmetric step sqrt(softmax_row * softmax_col) --
        the original drift affinity (row=each generation, col=each target).
    sinkhorn_iters>0: iterate row/column normalisation until each generation's row
        sums to 1 AND each target column receives equal mass -- a balanced
        (Sinkhorn) coupling that caps how many gens land on the SAME target.
        ``active`` (G, K) bool flags the entries that are real matches (the result of
        the matching, token OR repr alike). The column quota is shared ONLY over
        columns matched by >=1 gen; masked/unmatched columns get zero mass -- so
        Sinkhorn never balances against the dead columns the per-position masking
        produces (it only spreads gens that landed on the same real).
    """
    if sinkhorn_iters <= 0:
        a = torch.softmax(logits, dim=1)
        at = torch.softmax(logits, dim=0)
        return torch.sqrt(torch.clamp(a * at, min=1e-6))
    Gn, Kn = logits.shape
    # per-row max-subtraction for numerical stability (absorbed by row scaling u)
    Kmat = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
    if active is not None:
        Kmat = Kmat * active.to(Kmat.dtype)         # drop unmatched entries entirely
        col_active = active.any(dim=0)              # (K,) column matched by >=1 gen
        n_active = col_active.sum().clamp_min(1)
        col_target = torch.where(col_active, Gn / n_active.to(logits.dtype),
                                 logits.new_zeros(()))            # (K,) dead cols -> 0
    else:
        col_target = logits.new_full((Kn,), float(Gn) / float(Kn))
    v = torch.ones(Kn, device=logits.device, dtype=logits.dtype)
    for _ in range(sinkhorn_iters):
        u = 1.0 / (Kmat @ v).clamp_min(1e-12)       # row sum -> 1
        v = col_target / (Kmat.t() @ u).clamp_min(1e-12)  # active col sum -> Gn/n_active
    return u[:, None] * Kmat * v[None, :]


def min_2gram_dist(a: torch.Tensor, b: torch.Tensor,
                   return_arg: bool = False):
    """Min 2-gram distance between every (a_i, b_j) sample pair.

    a: (A, M, D)  b: (B, M, D)  ->  D_min: (A, B); if return_arg also the
    matched 2-gram indices (pa: a's 2-gram, pb: b's 2-gram), each (A, B).
    """
    A, M, D = a.shape
    Bn = b.shape[0]
    dd = torch.cdist(a.reshape(A * M, D), b.reshape(Bn * M, D))   # (A*M, B*M)
    dd = dd.reshape(A, M, Bn, M).permute(0, 2, 1, 3).reshape(A, Bn, M * M)
    dmin, amin = dd.min(dim=-1)                                   # (A, B)
    if not return_arg:
        return dmin
    return dmin, amin // M, amin % M                              # pa, pb


@torch.no_grad()
def greedy_one_to_one(sim: torch.Tensor) -> torch.Tensor:
    """ning-style greedy 1-to-1 match. sim (G, P), ideally P>=G. Sort all pairs
    by similarity descending, assign each generation a UNIQUE real. Returns the
    matched real index per generation (G,), -1 if left unmatched (P<G)."""
    G, P = sim.shape
    idx = sim.flatten().argsort(descending=True)
    rows = torch.div(idx, P, rounding_mode="floor")
    cols = idx % P
    used_r = torch.zeros(G, dtype=torch.bool, device=sim.device)
    used_c = torch.zeros(P, dtype=torch.bool, device=sim.device)
    matched = torch.full((G,), -1, dtype=torch.long, device=sim.device)
    for _ in range(G):
        valid = (~used_r[rows]) & (~used_c[cols])
        nz = torch.nonzero(valid)
        if nz.numel() == 0:
            break
        p = nz[0, 0]
        r, c = rows[p], cols[p]
        matched[r] = c
        used_r[r] = True
        used_c[c] = True
    return matched


def match_loss_2gram(gen: torch.Tensor, real: torch.Tensor
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Greedy one-to-one matching attraction (ning MD_loss analogue, in embedding
    space). Each generation is matched to a UNIQUE real (smallest position-aligned
    distance) and pulled toward THAT one real: a clean single target (no soft
    averaging -> no off-manifold garbage), distinct targets by construction
    (-> coverage + anti-collapse, no Sinkhorn / no repulsion needed).

    gen  (G, M, D) generated 2-grams (grad).  real (P, M, D) candidates, P>=G.
    """
    G, M, D = gen.shape
    P = real.shape[0]
    gen = gen.to(torch.float32)
    real = real.to(torch.float32)
    gen_f = gen.reshape(G, -1)
    real_f = real.reshape(P, -1)
    with torch.no_grad():
        cost = torch.cdist(gen_f.detach(), real_f)          # (G, P)
        matched = greedy_one_to_one(-cost)                  # (G,) real idx per gen
    valid = matched >= 0
    tgt = real_f[matched.clamp_min(0)]                      # (G, M*D)
    diff = (gen_f - tgt)[valid]
    loss = (diff ** 2).mean()
    info = {"scale": float(cost.mean()), "matched": float(valid.sum().item())}
    return loss, info


def gpt2_repair_loss(gen_emb: torch.Tensor, repair_wte: torch.Tensor,
                     repair_valid: torch.Tensor, own_wte: torch.Tensor,
                     implausible: torch.Tensor,
                     temp: float = 1.0, repel: float = 1.0,
                     target_mode: str = "self", repel_abs: bool = False,
                     abs_scale: float = 1.0, repel_perpos: bool = False,
                     repel_intra: float = 0.0, free_prefix: int = 0,
                     sphere: bool = False, sphere_step: str = "proj",
                     sphere_geo_max: float = 1.5708, temp_list=(),
                     repel_extra: torch.Tensor = None
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Per-position GPT-2 token target + whole-sequence anti-collapse repulsion.

    gen_emb     (G, T, D)        generator output (grad)
    repair_wte  (G, T, n_pos, D) wte of the candidate tokens per position (see build_repairs)
    own_wte     (G, T, D)        wte of the gen's OWN decoded token per position
    implausible (G, T) bool      prob(gen token) <= thresh (used by "self"; diag for "teacher")
    repel_extra (K, T, D)        OPTIONAL extra no-grad generations. They widen the
                                 REPULSION neighbour set only -- never attracted, never
                                 in the loss, never given a gradient. Lets the repel
                                 field be estimated from G+K samples while the gradient
                                 batch stays G. None => identical to the original.

    ATTRACTION -> an ON-MANIFOLD token target at every position; both modes weight the
    candidates by the gen's own embedding DISTANCE (mode-seeking: pull toward the nearest
    candidate, not the candidate mean):
      "self":    implausible position -> distance-affinity over nearest PLAUSIBLE tokens;
                 plausible position   -> the gen's OWN token (snap, zero force). Degenerate
                 (a locally-plausible token is a zero-force fixed point -> soup).
      "teacher": EVERY position -> distance-affinity over the GPT-2 SUPPORT SET (A2 /
                 reverse-KL). No own-token snap, no implausible gate; force ~ how far the
                 gen is from what GPT-2 wants, so locally-plausible-but-not-preferred tokens
                 (',' '\\n') still get pulled away. Positions with no valid candidate stay.

    REPULSION (whole-sequence, weight ``repel``): each generation pushed from the others
    (softmax-weighted by whole-sequence distance). repel_abs=False scales it to the
    ATTRACTION RMS (vanishes when attraction ~0); repel_abs=True (B) scales it to the
    candidate-DISTANCE scale (an absolute unit that survives at the degenerate point).
    Goal is detached: gradient flows ONLY through gen_emb.
    """
    G, T, D = gen_emb.shape
    old = gen_emb.detach().float()
    with torch.no_grad():                                           # goal carries NO gradient
        # --- attraction: distance-affinity token target ---
        rw = repair_wte.float()                                     # (G, T, n_pos, D)
        dist = torch.sqrt(((old.unsqueeze(2) - rw) ** 2).sum(dim=-1) + 1e-12)  # (G, T, n_pos)
        if target_mode == "teacher":
            vm = repair_valid                                       # every supported candidate
        else:
            vm = repair_valid & implausible.unsqueeze(-1)         # candidates @ implausible only
        scale = dist[vm].mean() if vm.any() else dist.new_tensor(1.0)   # reported + repel_abs unit
        if target_mode == "teacher":
            # affinity temperature = per-position SPREAD (std) of the candidate distances, NOT the
            # global mean. Dividing by the mean (~17) washes the per-candidate distances to ~uniform
            # logits => target = support CENTROID = mean-seeking blur (the A1 failure: every position
            # sits at the mean of GPT-2's support set and decodes to word salad). Per-row std keeps
            # the softmax contrastive so the NEAREST candidate dominates => real mode-seeking (A2).
            vmask = repair_valid.float()
            cnt = vmask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean_r = (dist * vmask).sum(dim=-1, keepdim=True) / cnt
            tscale = (((dist - mean_r) * vmask) ** 2).sum(dim=-1, keepdim=True).div(cnt).sqrt()
            tscale = tscale.clamp_min(1e-6)                       # (G, T, 1) per-position spread
        else:
            tscale = scale.clamp_min(1e-6)                        # self mode: original (global mean)
        temps = list(temp_list) if len(temp_list) > 0 else [temp]   # multi-temp fusion (variant 3)
        aff = None
        for tp in temps:                                          # average softmax affinity over temps
            lg = (-dist / (tscale * tp)).masked_fill(~repair_valid, float("-inf"))
            a = torch.nan_to_num(torch.softmax(lg, dim=-1))
            aff = a if aff is None else aff + a
        aff = aff / len(temps)                                    # blend sharp + broad scales
        centroid = (aff.unsqueeze(-1) * rw).sum(dim=2)            # (G, T, D) nearest-mode target
        has_cand = repair_valid.any(dim=-1).unsqueeze(-1)
        if target_mode == "teacher":                              # pull every position; no candidate -> stay
            target = torch.where(has_cand, centroid, old)
        else:                                                      # repair implausible; else snap own
            do_repair = (implausible.unsqueeze(-1) & has_cand)
            target = torch.where(do_repair, centroid, own_wte.float())
        attract = target - old                                     # (G, T, D) raw restoring force
        a_rms = torch.sqrt(torch.clamp((attract ** 2).mean(), min=1e-8))   # base attraction scale

        # --- (1) gen-gen repulsion: push different gens apart (per-position OR whole-sequence) ---
        K = 0 if repel_extra is None else repel_extra.shape[0]
        if repel > 0 and G + K > 1:
            # Self-mask over the (G, G+K) neighbour set: only the G real rows can be their own
            # neighbour, the K extras never are (no real gen is among them). At K=0 this is
            # exactly torch.eye(G) and the whole block reduces to the pre-extras behaviour.
            eye = torch.zeros(G, G + K, dtype=torch.bool, device=old.device)
            eye[:, :G] = torch.eye(G, dtype=torch.bool, device=old.device)
            if repel_perpos:
                # PER-POSITION: gen-gen distance computed INDEPENDENTLY at each position, so
                # every position (incl. the prefix) is diversified -- not just the tail.
                op = old.permute(1, 0, 2)                          # (T, G, D)
                op_all = op if K == 0 else torch.cat(
                    [op, repel_extra.permute(1, 0, 2)], dim=1)     # (T, G+K, D)
                dgg = torch.cdist(op, op_all)                      # (T, G, G+K)
                eye3 = eye.unsqueeze(0)                            # (1, G, G+K)
                offd = ~eye3
                srep = ((dgg * offd).sum(dim=(-1, -2))
                        / offd.sum(dim=(-1, -2)).clamp_min(1.0)).clamp_min(1e-6).view(T, 1, 1)
                logits = (-dgg / (srep * temp)).masked_fill(eye3, float("-inf"))
                aff_rep = torch.softmax(logits, dim=-1)            # (T, G, G+K)
                rep = (op - aff_rep @ op_all).permute(1, 0, 2)     # (G, T, D)
            else:
                # WHOLE-SEQUENCE: gen-gen distance over the flattened sequence (tail-only div risk)
                flat = old.reshape(G, -1)                          # (G, T*D)
                flat_all = flat if K == 0 else torch.cat(
                    [flat, repel_extra.reshape(K, -1)], dim=0)     # (G+K, T*D)
                dgg = torch.cdist(flat, flat_all)                  # (G, G+K)
                srep = dgg[~eye].mean().clamp_min(1e-6)
                logits = (-dgg / (srep * temp)).masked_fill(eye, float("-inf"))   # closer -> repel more
                aff_rep = torch.softmax(logits, dim=1)
                rep = (flat - aff_rep @ flat_all).reshape(G, T, D) # push g from nearby-gen centroid
            r_rms = torch.sqrt(torch.clamp((rep ** 2).mean(), min=1e-8))
            if repel_abs:                                          # B: FIXED wte-manifold unit (does
                coef = abs_scale / r_rms                          # NOT grow with inflation -> no runaway)
            else:                                                  # original: ~attraction magnitude
                a_rms = torch.sqrt(torch.clamp((attract ** 2).mean(), min=1e-8))
                coef = a_rms / r_rms
            attract = attract + repel * coef * rep

        # --- (2) intra-sequence repulsion: push a gen's OWN positions apart (anti token-repeat) ---
        if repel_intra > 0 and T > 1:
            dpp = torch.cdist(old, old)                            # (G, T, T) position-position dist
            eyeT = torch.eye(T, dtype=torch.bool, device=old.device).unsqueeze(0)   # (1, T, T)
            offt = ~eyeT
            sp = ((dpp * offt).sum(dim=(-1, -2))
                  / offt.sum(dim=(-1, -2)).clamp_min(1.0)).clamp_min(1e-6).view(G, 1, 1)
            logits = (-dpp / (sp * temp)).masked_fill(eyeT, float("-inf"))
            affp = torch.softmax(logits, dim=-1)                  # (G, T, T) closer same-seq pos -> more
            repi = old - affp @ old                               # (G, T, D) push pos from repeated ones
            ri_rms = torch.sqrt(torch.clamp((repi ** 2).mean(), min=1e-8))
            coef_i = (abs_scale / ri_rms) if repel_abs else (a_rms / ri_rms)
            attract = attract + repel_intra * coef_i * repi

        if free_prefix > 0:                                       # leading positions kept as-is:
            attract[:, :free_prefix, :] = 0.0                     # zero force -> zero gradient there
        if not sphere:
            goal = old + attract
        elif sphere_step == "proj":                               # chord + project (radial distorts)
            goal = (old + attract)
            goal = goal / goal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        else:                                                     # tangent-project the force first
            ftan = attract - (attract * old).sum(dim=-1, keepdim=True) * old
            fn = ftan.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            if sphere_step == "geodesic":                         # exp-map along the great circle
                a = fn.clamp(max=sphere_geo_max)
                goal = torch.cos(a) * old + torch.sin(a) * (ftan / fn)
            else:                                                 # "retract": project the tangent step
                goal = old + ftan
                goal = goal / goal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    loss = ((gen_emb.float() - goal) ** 2).mean()
    info = {"scale": float(scale),
            "n_repair": float(implausible.float().sum(dim=1).mean()),
            "repel_n": float(G + K)}
    return loss, info


def drift_loss_2gram(
    gen: torch.Tensor,                  # (G, M, D) generated 2-grams (grad)
    pos: torch.Tensor,                  # (P, M, D) positive real 2-grams
    neg: Optional[torch.Tensor] = None, # (N, M, D) negative real 2-grams
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2),
    attract_temp: float = 1.0,          # affinity temperature; <1 sharpens (->one-hot)
    sinkhorn_iters: int = 0,            # >0: balanced (Sinkhorn) affinity, caps per-real intake
    per_position: bool = False,         # True: per-position force (only the matched 2-gram pulled)
    pos_match_pos: Optional[torch.Tensor] = None,  # (G,P) same-position match pos per (gen,positive);
                                        # if given, per-position drift with attraction routed there
    pos_own: Optional[torch.Tensor] = None,  # (G,P) bool: did gen g actually SELECT positive j?
                                        # per-position attraction only pulls a gen toward its OWN picks
    repel_intra: float = 0.0,           # intra-sequence repulsion weight: push each position away from
                                        # the OTHER positions of its own generation (anti within-sentence
                                        # token repetition). Same mechanism as the teacher path; without
                                        # it the off-policy dataset drift collapses every position onto
                                        # the dominant token ("the the the ..."). 0 = off.
    force_pos_mask: Optional[torch.Tensor] = None,  # (G,M) bool: keep force only at these positions
                                        # (others get zero force -> no gradient). Used by GPT-2 mode to
                                        # restrict force to [0..p*] while the affinity stays whole-sequence.
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Affinity from the POSITION-ALIGNED distance: flatten each sample's 2-gram
    sequence and take L2 (gen position p vs target position p, NO cross-position
    pairs). This reflects whether a generation actually REPRODUCES a target
    position-by-position, so it is discriminative and not biased toward
    central/common-token reals the way mean-over-all-pairs is. Consistent with
    the position-aligned force below.

    ``attract_temp`` is the affinity softmax temperature: the logits are
    ``-dist/(R*attract_temp)``, so attract_temp<1 makes the affinity sharper
    (closer to one-hot, each generation locks onto its single nearest target).
    Small temp preserves the initial diversity between generations instead of
    pulling them all toward the soft average (which homogenises them early).

    NOTE: min-2-gram (cross-position, "one matching 2-gram is enough") is still
    used, but only for candidate positive SELECTION in the memory bank.
    """
    G, M, D = gen.shape
    dtype = torch.float32
    gen = gen.to(dtype)
    pos = pos.to(dtype)
    old_gen = gen.detach()
    if neg is None:
        neg = old_gen[:0]
    neg = neg.to(dtype)
    targets = torch.cat([old_gen, neg, pos], dim=0)              # (K, M, D)
    K = targets.shape[0]
    C_g, C_n = G, neg.shape[0]
    split_idx = C_g + C_n
    info: Dict[str, float] = {}

    self_mask = torch.zeros(G, K, dtype=torch.bool, device=gen.device)
    diag = torch.arange(G, device=gen.device)
    self_mask[diag, diag] = True                                # gen vs itself

    MASK_VAL = 1e9                                               # masked (non-matching) target

    def coeffs_from_dist(dist, active=None):
        """(G,K) distance -> list of per-R coeff (G,K), with scale-norm + self-mask.
        Entries set to MASK_VAL (masked-out targets) are excluded from the scale and
        get ~0 affinity. ``active`` (G,K) bool (the match result) is passed to Sinkhorn
        so its column quota only counts the matched columns."""
        valid = (~self_mask) & (dist < MASK_VAL * 0.5)
        scale = dist[valid].mean() if valid.any() else dist.new_tensor(1.0)
        dn = dist / torch.clamp(scale, min=1e-3)
        dn = dn.masked_fill(self_mask, 100.0)
        out = []
        for R in R_list:
            aff = _affinity(-dn / (R * attract_temp), sinkhorn_iters, active=active)  # symmetric: V=0
            aff_neg, aff_pos = aff[:, :split_idx], aff[:, split_idx:]
            sum_pos = aff_pos.sum(dim=1, keepdim=True)
            sum_neg = aff_neg.sum(dim=1, keepdim=True)
            out.append(torch.cat([-aff_neg * sum_pos, aff_pos * sum_neg], dim=1))
        return float(scale), out

    with torch.no_grad():
        force = torch.zeros(G, M, D, device=gen.device, dtype=dtype)
        if per_position and pos_match_pos is not None:
            # per-position drift, position-aligned (i=j). At position p the ATTRACTION
            # pulls gen g only toward positives that (a) g SELECTED itself (pos_own, Q2 --
            # no class condition here, so other gens' picks are unrelated) AND (b) match
            # at p (pm==p). The match-result mask (gen cols + matched positives) is fed to
            # Sinkhorn so its column quota only counts genuinely matched columns.
            pm = pos_match_pos.to(gen.device)                  # (G, P_pos)
            own = (pos_own.to(gen.device) if pos_own is not None
                   else torch.ones_like(pm, dtype=torch.bool))
            gen_active = torch.ones(G, split_idx, dtype=torch.bool, device=gen.device)
            scales = 0.0
            for p in range(M):
                dist_p = torch.cdist(old_gen[:, p, :], targets[:, p, :])   # (G, K)
                matched_p = (pm == p) & own                     # (G, P_pos) own pick matched at p
                dpos = torch.where(matched_p, dist_p[:, split_idx:],
                                   torch.full_like(dist_p[:, split_idx:], MASK_VAL))
                dist_p = torch.cat([dist_p[:, :split_idx], dpos], dim=1)
                active_p = torch.cat([gen_active, matched_p], dim=1)   # match result -> Sinkhorn
                scale_p, coeffs = coeffs_from_dist(dist_p, active_p)
                scales += scale_p
                tgt_p, gen_p = targets[:, p, :], old_gen[:, p, :]
                for coeff in coeffs:
                    ff = coeff @ tgt_p - coeff.sum(dim=1, keepdim=True) * gen_p   # (G,D)
                    fnorm = torch.sqrt(torch.clamp((ff ** 2).mean(), min=1e-8))
                    force[:, p, :] = force[:, p, :] + ff / fnorm
            info["scale"] = scales / max(M, 1)
        elif per_position:
            # argmin routing: min-2-gram picks, per (gen,target), the SINGLE matched
            # 2-gram position pa (in gen) / pb (in target). Force pulls only gen's
            # position pa toward target's position pb; other positions get no force.
            dmin, pa, pb = min_2gram_dist(old_gen, targets, return_arg=True)  # (G,K)
            info["scale"], coeffs = coeffs_from_dist(dmin)
            ig = diag.view(G, 1).expand(G, K)
            iy = torch.arange(K, device=gen.device).view(1, K).expand(G, K)
            gen_m = old_gen[ig, pa]                             # (G,K,D) gen matched 2-gram
            tgt_m = targets[iy, pb]                             # (G,K,D) target matched 2-gram
            idx = pa.unsqueeze(-1).expand(G, K, D)
            for coeff in coeffs:
                delta = coeff.unsqueeze(-1) * (tgt_m - gen_m)   # (G,K,D)
                f = torch.zeros(G, M, D, device=gen.device, dtype=dtype)
                f.scatter_add_(1, idx, delta)                  # force only at position pa
                fnorm = torch.sqrt(torch.clamp((f ** 2).mean(), min=1e-8))
                force = force + f / fnorm
        else:
            dist = torch.cdist(old_gen.reshape(G, -1), targets.reshape(K, -1))   # (G,K)
            info["scale"], coeffs = coeffs_from_dist(dist)
            for coeff in coeffs:
                f = torch.einsum("gk,kmd->gmd", coeff, targets)
                f = f - coeff.sum(dim=1)[:, None, None] * old_gen
                fnorm = torch.sqrt(torch.clamp((f ** 2).mean(), min=1e-8))
                force = force + f / fnorm

        # intra-sequence repulsion: push each position away from the OTHER positions of its
        # own generation (anti within-sentence repetition). Ported from the teacher path
        # (gpt2_repair_loss); this is what stops the off-policy dataset drift from collapsing
        # every position onto the dominant token. The attraction force above is RMS ~1 per R,
        # so repel_intra is measured in those same units.
        if repel_intra > 0 and M > 1:
            a_rms = torch.sqrt(torch.clamp((force ** 2).mean(), min=1e-8))   # attraction magnitude
            dpp = torch.cdist(old_gen, old_gen)                 # (G, M, M) position-position dist
            eyeM = torch.eye(M, dtype=torch.bool, device=old_gen.device).unsqueeze(0)
            offm = ~eyeM
            sp = ((dpp * offm).sum(dim=(-1, -2))
                  / offm.sum(dim=(-1, -2)).clamp_min(1.0)).clamp_min(1e-6).view(G, 1, 1)
            affp = torch.softmax((-dpp / (sp * attract_temp)).masked_fill(eyeM, float("-inf")), dim=-1)
            repi = old_gen - affp @ old_gen                     # (G, M, D) push off repeated positions
            ri_rms = torch.sqrt(torch.clamp((repi ** 2).mean(), min=1e-8))
            # combine (intra sets the direction at large repel_intra) then rescale the total
            # force back to the attraction magnitude -- the euclidean analogue of the teacher's
            # geodesic-step cap, so a large repel_intra steers WITHOUT inflating the step / loss.
            combined = force + repel_intra * (a_rms / ri_rms) * repi
            c_rms = torch.sqrt(torch.clamp((combined ** 2).mean(), min=1e-8))
            force = combined * (a_rms / c_rms)

        if force_pos_mask is not None:                          # GPT-2 mode: force only on [0..p*]
            force = force * force_pos_mask.unsqueeze(-1).to(force.dtype)
        goal = old_gen + force                                  # detached target

    loss = ((gen - goal) ** 2).mean()
    return loss, info
