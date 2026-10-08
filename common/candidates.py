"""Prepared reasoning data grouping, tokenization, and scoring."""
import re
import json
import torch
from collections import defaultdict
from decimal import Decimal, InvalidOperation

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


LABEL_RE = re.compile(r"####\s*(true|false|unknown)\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"####\s*(-?\d[\d,]*\.?\d*)")


def extract_scored_answer(text):
    """Extract either a ProofWriter label or a numeric final answer."""
    text = str(text)
    match = LABEL_RE.search(text)
    if match:
        return match.group(1).lower()
    match = NUMBER_RE.search(text)
    if match:
        return match.group(1).replace(",", "")
    bare = text.strip().lower().replace(",", "").replace("$", "")
    return bare if bare in {"true", "false", "unknown"} else ""


def scored_answers_equal(pred, gold):
    if gold in {"true", "false", "unknown"}:
        return pred == gold
    try:
        return pred != "" and Decimal(pred) == Decimal(gold)
    except InvalidOperation:
        return False


class CandidateDataset:
    def __init__(self, groups, tokenizer, query_len, resp_len, max_candidates):
        self.groups = groups
        self.tok = tokenizer
        self.Lq, self.Lr = query_len, resp_len
        # max_candidates=0 means that load_candidate_groups retained every
        # candidate; otherwise use the explicit fixed cap.
        self.K = (max_candidates if max_candidates > 0 else
                  max((len(g["solutions"]) for g in groups), default=1))
        self.pad = tokenizer.pad_token_id
        if self.pad is None:
            self.pad = tokenizer.eos_token_id
        self.truncated_queries = 0
        self.truncated_targets = 0
        self.max_query_tokens = 0
        self.max_target_tokens = 0
        qids, qmask, cids, cmask, gold = [], [], [], [], []
        for g in groups:
            prompt = f"Question: {g['question']}\nAnswer:"
            query_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            self.max_query_tokens = max(self.max_query_tokens, len(query_tokens))
            self.truncated_queries += len(query_tokens) > self.Lq
            qi = query_tokens[-self.Lq:]
            qpad = [self.pad] * (self.Lq - len(qi)) + qi
            qids.append(qpad)
            qmask.append([False] * (self.Lq - len(qi)) + [True] * len(qi))
            rows, masks = [], []
            for sol in g["solutions"][:self.K]:
                solution_tokens = tokenizer(
                    sol, add_special_tokens=False)["input_ids"]
                self.max_target_tokens = max(
                    self.max_target_tokens, len(solution_tokens) + 1)
                self.truncated_targets += len(solution_tokens) + 1 > self.Lr
                ids = solution_tokens[:self.Lr - 1]
                ids = ids + [tokenizer.eos_token_id]
                ids = ids[:self.Lr]
                rows.append(ids + [self.pad] * (self.Lr - len(ids)))
                masks.append([True] * len(ids) + [False] * (self.Lr - len(ids)))
            while len(rows) < self.K:
                rows.append([self.pad] * self.Lr)
                masks.append([False] * self.Lr)
            cids.append(rows); cmask.append(masks); gold.append(g["answer"])
        self.query_ids = torch.tensor(qids, dtype=torch.long)
        self.query_mask = torch.tensor(qmask, dtype=torch.bool)
        self.candidate_ids = torch.tensor(cids, dtype=torch.long)
        self.candidate_mask = torch.tensor(cmask, dtype=torch.bool)
        self.gold = gold

    def sample(self, n, device):
        ix = torch.randint(0, len(self.groups), (n,))
        return (
            self.query_ids[ix].to(device),
            self.query_mask[ix].to(device),
            self.candidate_ids[ix].to(device),
            self.candidate_mask[ix].to(device),
        )


@torch.no_grad()
def eval_one_pass(gen, ds, embedder, tok, cfg, device, n):
    gen.eval()
    n = min(n, len(ds.groups))
    correct = 0
    formatted = 0
    candidate_token_acc_sum = 0.0
    candidate_sequence_correct = 0
    token_correct = token_total = 0
    sequence_correct = eos_correct = eos_total = 0
    shown = []
    for s in range(0, n, 32):
        end = min(s + 32, n)
        q = ds.query_ids[s:end].to(device)
        qm = ds.query_mask[s:end].to(device)
        qe = embedder.query_embeds(q)
        z = gen.sample_z(q.shape[0], cfg.noise_dim, cfg.temp, device)
        amp = bool(cfg.use_bf16 and device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            emb = gen(qe, qm, z).float()
        if cfg.sphere_norm:
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        ids = embedder.decode(emb)
        candidate_ids = ds.candidate_ids[s:end].to(device)
        candidate_mask = ds.candidate_mask[s:end].to(device)
        candidate_valid = candidate_mask.any(dim=-1)
        candidate_matches = ids[:, None, :].eq(candidate_ids)
        candidate_ratios = (candidate_matches & candidate_mask).sum(dim=-1) / (
            candidate_mask.sum(dim=-1).clamp_min(1))
        candidate_ratios = candidate_ratios.masked_fill(~candidate_valid, -1.0)
        candidate_token_acc_sum += float(candidate_ratios.max(dim=-1).values.sum())
        candidate_exact = (candidate_matches | ~candidate_mask).all(dim=-1)
        candidate_sequence_correct += int(
            (candidate_exact & candidate_valid).any(dim=-1).sum())
        if ds.K == 1:
            target_ids = ds.candidate_ids[s:end, 0].to(device)
            target_mask = ds.candidate_mask[s:end, 0].to(device)
            matches = ids.eq(target_ids)
            token_correct += int((matches & target_mask).sum())
            token_total += int(target_mask.sum())
            sequence_correct += int((matches | ~target_mask).all(dim=1).sum())
            eos_mask = target_mask & target_ids.eq(tok.eos_token_id)
            eos_correct += int((matches & eos_mask).sum())
            eos_total += int(eos_mask.sum())
        for i, row in enumerate(ids):
            row_ids = row.tolist()
            eos_id = tok.eos_token_id
            if eos_id is not None and eos_id in row_ids:
                row_ids = row_ids[:row_ids.index(eos_id)]
            text = tok.decode(row_ids, skip_special_tokens=True)
            pred = extract_scored_answer(text)
            gold = extract_scored_answer(f"#### {ds.gold[s + i]}")
            formatted += int(pred != "")
            correct += int(scored_answers_equal(pred, gold))
            if len(shown) < 6:
                shown.append(
                    f"Q: {ds.groups[s+i]['question'][:180]}\n"
                    f"gold={gold!r} pred={pred!r} output={text!r}")
    metrics = {
        "accuracy": correct / max(n, 1),
        "format_rate": formatted / max(n, 1),
        "candidate_token_acc": candidate_token_acc_sum / max(n, 1),
        "candidate_sequence_acc": candidate_sequence_correct / max(n, 1),
        "n_queries": float(n),
    }
    if ds.K == 1:
        metrics.update({
            "token_acc": token_correct / max(token_total, 1),
            "eos_acc": eos_correct / max(eos_total, 1),
            "sequence_acc": sequence_correct / max(n, 1),
        })
    return metrics, shown
