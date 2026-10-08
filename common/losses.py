"""Drift objectives, supervised embedding loss, and candidate attraction."""
import math
import torch
from typing import Dict, Tuple, Optional

_NEG = -1e4

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


def continuation_repair_loss(gen_emb, repair_wte, repair_valid, implausible, group_size: int,
                     temp: float = 0.3, repel: float = 5.0, repel_perpos: bool = True,
                     repel_intra: float = 5.0, repel_abs: bool = False, abs_scale: float = 1.0,
                     free_prefix: int = 0, sphere: bool = True, sphere_step: str = "geodesic",
                     sphere_geo_max: float = 1.5708, temp_list=(),
                     gold_wte=None, gold_mask=None, gold_weight: float = 0.0,
                     loss_mask=None, teacher_weight: float = 1.0, repel_block: bool = True
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """gen_emb (G, T, H) grad; repair_wte (G, T, n_pos, H); repair_valid (G, T, n_pos).
    G = Q * group_size (block order). Teacher-mode attraction toward the Qwen support
    set; every position pulled (no own-token snap)."""
    G, T, H = gen_emb.shape
    old = gen_emb.detach().float()
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
            attract = attract + gold_weight * gpull
            gold_frac = float(torch.sqrt(torch.clamp((gold_weight * gpull) ** 2, min=0).mean()))

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
            offt = ~eyeT
            sp = ((dpp * offt).sum(dim=(-1, -2))
                  / offt.sum(dim=(-1, -2)).clamp_min(1.0)).clamp_min(1e-6).view(G, 1, 1)
            logits = (-dpp / (sp * temp)).masked_fill(eyeT, float("-inf"))
            affp = torch.softmax(logits, dim=-1)
            repi = old - affp @ old
            if loss_mask is not None:                                # no force on the pad tail
                repi = repi * loss_mask.unsqueeze(-1).to(repi.dtype)
            ri_rms = torch.sqrt(torch.clamp((repi ** 2).mean(), min=1e-8))
            coef_i = (abs_scale / ri_rms) if repel_abs else (a_rms / ri_rms)
            attract = attract + repel_intra * coef_i * repi

        if free_prefix > 0:
            attract[:, :free_prefix, :] = 0.0
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

    diff2 = (gen_emb.float() - goal) ** 2                             # (G, T, H)
    if loss_mask is None:
        loss = diff2.mean()
    else:                                    # mean over SUPERVISED positions only -- the pad
        m = loss_mask.unsqueeze(-1).to(diff2.dtype)                   # tail must not dilute it
        loss = (diff2 * m).sum() / (m.sum() * diff2.shape[-1]).clamp_min(1.0)
    info = {"scale": float(scale),
            "gold_f": gold_frac,
            "n_implaus": float(implausible.float().sum(dim=1).mean())}
    return loss, info


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


def softmin_candidate_target(gen_emb, candidates, candidate_mask,
                             temperature=0.15, detach_query=True):
    """Build a differentiable target from candidates.

    gen_emb:       (B,T,H), current generated manifold embeddings.
    candidates:    (B,K,T,H), candidate embeddings, padded.
    candidate_mask: (B,K,T), valid candidate positions.
    Returns target (B,T,H), weights (B,K), seq_dist (B,K).

    Distances are computed over the candidate's valid positions.  The target
    is position-aligned; positions beyond a candidate's end are not included
    in the loss mask.
    """
    import torch
    g = gen_emb.detach().float() if detach_query else gen_emb.float()
    c = candidates.float()
    cm = candidate_mask.bool()
    # Per-position squared Euclidean distance, normalized by embedding width.
    d2 = ((g[:, None] - c) ** 2).mean(dim=-1)
    d2 = d2.masked_fill(~cm, 0.0)
    denom = cm.sum(dim=-1).clamp_min(1)
    seq_dist = d2.sum(dim=-1) / denom
    valid_group = cm.any(dim=-1)
    logits = -seq_dist / max(float(temperature), 1e-6)
    logits = logits.masked_fill(~valid_group, -1e4)
    weights = torch.softmax(logits, dim=-1)
    weights = torch.nan_to_num(weights)
    # A candidate can be valid at some positions and padded at later positions.
    # Renormalize independently at every position so padding embeddings do not
    # become part of the target or dilute the valid candidate mixture.
    valid_weights = weights[..., None] * cm.float()
    raw_denom = valid_weights.sum(dim=1)
    denom_pos = raw_denom.clamp_min(1e-6)
    target = (valid_weights[..., None] * c).sum(dim=1) / denom_pos[..., None]
    target_mask = raw_denom > 1e-6
    return target, target_mask, weights, seq_dist


def candidate_attraction_loss(gen_emb, candidates, candidate_mask,
                              weight=3.0, temperature=0.3,
                              repel_intra=0.0,
                              sphere=True, sphere_step="geodesic",
                              sphere_geo_max=1.5708):
    """Gold-style manifold loss using a query-specific candidate set."""
    import torch
    old = gen_emb.detach().float()
    target, mask, weights, seq_dist = softmin_candidate_target(
        gen_emb, candidates, candidate_mask, temperature=temperature)
    force = weight * (target - old)
    force = force * mask[..., None].to(force.dtype)

    # Repel nearby positions within one response. Cross-query repulsion would be
    # harmful here because each batch item is a different question and k_samples
    # is currently one.
    if repel_intra > 0 and old.shape[1] > 1:
        dpp = torch.cdist(old, old)
        eye = torch.eye(old.shape[1], dtype=torch.bool, device=old.device).unsqueeze(0)
        valid_pairs = mask.unsqueeze(2) & mask.unsqueeze(1) & ~eye
        pair_count = valid_pairs.sum(dim=-1, keepdim=True).clamp_min(1)
        scale = ((dpp * valid_pairs).sum(dim=-1, keepdim=True) / pair_count)
        scale = scale.clamp_min(1e-6)
        logits = (-dpp / (scale * max(float(temperature), 1e-6))).masked_fill(
            ~valid_pairs, float("-inf"))
        affinity = torch.nan_to_num(torch.softmax(logits, dim=-1))
        repel = old - affinity @ old
        repel = repel * mask[..., None].to(repel.dtype)
        force_rms = torch.sqrt(force.pow(2).mean().clamp_min(1e-8))
        repel_rms = torch.sqrt(repel.pow(2).mean().clamp_min(1e-8))
        force = force + repel_intra * (force_rms / repel_rms) * repel
    if sphere:
        tangent = force - (force * old).sum(dim=-1, keepdim=True) * old
        norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if sphere_step == "geodesic":
            angle = norm.clamp(max=sphere_geo_max)
            goal = torch.cos(angle) * old + torch.sin(angle) * tangent / norm
        else:
            goal = old + tangent
            goal = goal / goal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    else:
        goal = old + force
    diff = (gen_emb.float() - goal) ** 2
    loss = (diff * mask[..., None]).sum() / (
        mask.sum() * diff.shape[-1]).clamp_min(1.0)
    with torch.no_grad():
        pred = torch.nn.functional.normalize(gen_emb.float(), dim=-1)
        tar = torch.nn.functional.normalize(target.float(), dim=-1)
        cos = (pred * tar).sum(dim=-1)
        cos = (cos * mask).sum() / mask.sum().clamp_min(1)
    return loss, {
        "candidate_cos": float(cos),
        "candidate_dist": float(seq_dist.min(dim=-1).values.mean()),
        "candidate_entropy": float(
            -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        ),
        "candidate_mask_frac": float(mask.float().mean()),
        "repel_intra": float(repel_intra),
    }
