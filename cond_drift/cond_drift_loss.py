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


def cond_repair_loss(gen_emb, repair_wte, repair_valid, implausible, group_size: int,
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
