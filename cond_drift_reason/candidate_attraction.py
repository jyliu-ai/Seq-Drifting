"""Query-specific multi-solution attraction for conditional text drifting.

For each query, candidates are the real solutions belonging to that query.
The target is a sequence-level soft-min over candidates:
  w_k = softmax(-distance(generated, candidate_k) / temperature)
  target = sum_k w_k * candidate_k

Candidates should normally be grouped by query AND final answer.  Grouping
different final answers would make the attraction set internally contradictory.
"""
import math
import re
from collections import defaultdict

def final_answer(text: str) -> str:
    m = re.search(r"####\s*(.+)", text)
    if m:
        return m.group(1).strip().replace(",", "").replace("$", "")
    return text.strip()


def normalize_query(text: str) -> str:
    return " ".join(str(text).split())


def load_candidate_groups(path, same_final=True, max_candidates=16):
    """Return list of {question, solutions, answer} groups from JSONL.

    Supports MetaMath-style `generated` and standard `answer` fields.
    Groups with one solution are retained: they are ordinary gold attraction.
    """
    import json

    grouped = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            q = normalize_query(row["question"])
            sol = str(row.get("generated", row.get("answer", ""))).strip()
            if not sol:
                continue
            key = (q, final_answer(sol)) if same_final else (q, "")
            grouped[key].append(sol)

    out = []
    for (q, ans), sols in grouped.items():
        unique = list(dict.fromkeys(sols))
        if max_candidates > 0:
            unique = unique[:max_candidates]
        out.append({"question": q, "solutions": unique, "answer": ans})
    return out


def summarize_groups(groups):
    sizes = [len(g["solutions"]) for g in groups]
    multi = [x for x in sizes if x > 1]
    return {
        "groups": len(groups),
        "multi_groups": len(multi),
        "max_candidates": max(sizes, default=0),
        "mean_candidates": sum(sizes) / max(len(sizes), 1),
    }


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
