"""Train conditional drifting with query-specific multi-solution attraction.

Unlike teacher mode, this never asks a frozen LM to score the model's own
possibly-wrong prefix.  Each query has a candidate set made from real dataset
solutions.  The candidate closest to the current generated sequence receives
the largest attraction weight.

This file is intended to be copied beside the existing cond_drift_reason
package and run with `python -m cond_drift_reason.train_candidate_attraction`.
"""
import argparse
import copy
import os
import re
import time
from decimal import Decimal, InvalidOperation

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import CondDriftConfig
from .qwen_features import load_qwen, QwenEmbedder
from .cond_generator import QwenCondGenerator, CondDriftGenerator
from .candidate_attraction import (
    load_candidate_groups, summarize_groups, candidate_attraction_loss)
from seq_drifting_common.ema import ema_update


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-json")
    p.add_argument("--test-json")
    p.add_argument("--teacher", required=True,
                   help="Qwen model whose input embedding manifold is used")
    p.add_argument("--backbone")
    p.add_argument("--same-final", action="store_true", default=True)
    p.add_argument("--allow-different-final", action="store_false", dest="same_final")
    p.add_argument("--max-candidates", type=int, default=8,
                   help="fixed per-group cap; 0 keeps all candidates")
    p.add_argument("--query-len", type=int, default=128)
    p.add_argument("--resp-len", type=int, default=256)
    p.add_argument("--queries-per-step", type=int, default=16)
    p.add_argument("--steps", type=int, default=120000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--candidate-temp", type=float, default=0.3)
    p.add_argument("--candidate-weight", type=float, default=3.0)
    p.add_argument("--repel-intra", type=float, default=1.0,
                   help="within-response repulsion; 0 disables it")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-queries", type=int, default=200)
    p.add_argument("--eval-train-queries", type=int, default=0,
                   help="also evaluate this many training queries; 0 disables")
    p.add_argument("--eval-only", action="store_true",
                   help="load --resume, evaluate, and exit without training")
    p.add_argument("--eval-repeats", type=int, default=1,
                   help="number of seeded noise draws in --eval-only mode")
    p.add_argument("--eval-seed", type=int, default=0,
                   help="first noise seed in --eval-only mode")
    p.add_argument("--fail-on-target-truncation", action="store_true",
                   help="abort if any target plus EOS exceeds --resp-len")
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--resume")
    p.add_argument("--d-model", type=int)
    p.add_argument("--nhead", type=int)
    p.add_argument("--ffn-dim", type=int)
    p.add_argument("--num-layers", type=int)
    p.add_argument("--noise-dim", type=int)
    p.add_argument("--temp", type=float)
    p.add_argument("--no-bf16", dest="use_bf16", action="store_false", default=None)
    p.add_argument("--no-qwen-backbone", dest="qwen_backbone",
                   action="store_false", default=None)
    return p.parse_args()


def cfg_from_args(a):
    cfg = CondDriftConfig()
    vals = {
        "teacher_model": a.teacher, "backbone_model": a.backbone or "",
        "query_len": a.query_len, "resp_len": a.resp_len,
        "queries_per_step": a.queries_per_step, "steps": a.steps, "lr": a.lr,
        "checkpoint_dir": a.ckpt_dir, "qwen_backbone": (
            True if a.qwen_backbone is None else a.qwen_backbone),
    }
    for k, v in vals.items():
        if v is not None:
            setattr(cfg, k, v)
    cfg.teacher_weight = 0.0
    cfg.repel = 0.0
    cfg.repel_intra = 0.0
    cfg.gold_weight = 0.0
    cfg.eval_gen_ppl = False
    return cfg


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


def main():
    a = parse_args()
    if a.eval_only:
        if not a.resume:
            raise ValueError("--eval-only requires --resume")
        if not a.test_json:
            raise ValueError("--eval-only requires --test-json")
    elif not a.train_json:
        raise ValueError("training requires --train-json")
    cfg = cfg_from_args(a)
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_rank = rank == 0
    torch.manual_seed(cfg.seed + rank)
    if main_rank:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    _, tok = load_qwen(cfg.teacher_model, str(device),
                       "bf16" if cfg.use_bf16 else "fp32")
    train_groups = (load_candidate_groups(
        a.train_json, same_final=a.same_final, max_candidates=a.max_candidates)
        if a.train_json else [])
    test_groups = load_candidate_groups(
        a.test_json, same_final=a.same_final, max_candidates=a.max_candidates
    ) if a.test_json else train_groups
    if main_rank and train_groups:
        print(f"[candidates] train {summarize_groups(train_groups)}", flush=True)
    if main_rank and a.test_json:
        print(f"[candidates] test  {summarize_groups(test_groups)}", flush=True)
    embedder = QwenEmbedder(cfg, tok, device)
    cfg.embed_dim, cfg.vocab_size = embedder.H, embedder.V
    train_ds = (CandidateDataset(train_groups, tok, cfg.query_len, cfg.resp_len,
                                 a.max_candidates)
                if train_groups else None)
    test_ds = CandidateDataset(test_groups, tok, cfg.query_len, cfg.resp_len,
                               a.max_candidates)
    if main_rank and train_ds is not None:
        print(f"[tokens] train query_truncated={train_ds.truncated_queries} "
              f"target_truncated={train_ds.truncated_targets} "
              f"query_max={train_ds.max_query_tokens} "
              f"target_with_eos_max={train_ds.max_target_tokens}", flush=True)
    if main_rank:
        print(f"[tokens] test query_truncated={test_ds.truncated_queries} "
              f"target_truncated={test_ds.truncated_targets} "
              f"query_max={test_ds.max_query_tokens} "
              f"target_with_eos_max={test_ds.max_target_tokens}", flush=True)
    truncated_targets = test_ds.truncated_targets
    if train_ds is not None:
        truncated_targets += train_ds.truncated_targets
    if a.fail_on_target_truncation and truncated_targets:
        raise ValueError(
            f"{truncated_targets} targets exceed resp_len={cfg.resp_len}; "
            "rebuild/filter the data or use a compatible longer-response model")
    gen = (QwenCondGenerator(cfg, cfg.embed_dim) if cfg.qwen_backbone
           else CondDriftGenerator(cfg, cfg.embed_dim)).to(device)
    ema = copy.deepcopy(gen).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    gen_ddp = DDP(gen, device_ids=[local_rank]) if ddp else gen
    opt = torch.optim.AdamW(gen.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start = 0
    if a.resume:
        checkpoint_device = "cpu" if a.eval_only else device
        blob = torch.load(a.resume, map_location=checkpoint_device,
                          weights_only=False)
        gen.load_state_dict(blob["model"]); ema.load_state_dict(blob["ema"])
        if not a.eval_only and "optimizer" in blob:
            opt.load_state_dict(blob["optimizer"])
        for pg in opt.param_groups:
            pg["lr"] = cfg.lr
        start = blob.get("step", 0)
        if main_rank:
            print(f"[resume] {a.resume} -> step {start}", flush=True)
        del blob

    if a.eval_only:
        if a.eval_repeats < 1:
            raise ValueError("--eval-repeats must be at least 1")
        if main_rank:
            results = []
            for repeat in range(a.eval_repeats):
                torch.manual_seed(a.eval_seed + repeat)
                metrics, shown = eval_one_pass(
                    ema, test_ds, embedder, tok, cfg, device, a.eval_queries)
                results.append(metrics)
                print("[eval-only {} repeat={}/{} seed={}] {}".format(
                    start, repeat + 1, a.eval_repeats, a.eval_seed + repeat,
                    " ".join(f"{k}={v:.3f}" for k, v in metrics.items())),
                    flush=True)
                if repeat == 0:
                    for sample in shown:
                        print("  " + sample.replace("\n", "\n  "), flush=True)
            if a.eval_repeats > 1:
                names = results[0].keys()
                means = {
                    name: sum(row[name] for row in results) / len(results)
                    for name in names
                }
                accuracies = [row["accuracy"] for row in results]
                print("[eval-only-summary {}] {} accuracy_min={:.3f} "
                      "accuracy_max={:.3f}".format(
                          start,
                          " ".join(f"{k}_mean={v:.3f}"
                                   for k, v in means.items()),
                          min(accuracies), max(accuracies)), flush=True)
        if ddp:
            dist.destroy_process_group()
        return

    t0 = time.time()
    if main_rank:
        print(f"[candidate] temp={a.candidate_temp} weight={a.candidate_weight} "
              f"K={a.max_candidates} same_final={a.same_final}", flush=True)
    first_step = start + 1 if a.resume else 0
    for step in range(first_step, cfg.steps + 1):
        gen_ddp.train()
        q, qm, cids, cmask = train_ds.sample(cfg.queries_per_step, device)
        B, K, T = cids.shape
        qe = embedder.query_embeds(q)
        cemb = embedder.lookup(cids.reshape(B * K, T)).reshape(
            B, K, T, cfg.embed_dim).float()
        z = gen.sample_z(B, cfg.noise_dim, cfg.temp, device)
        amp = bool(cfg.use_bf16 and device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            emb = gen_ddp(qe, qm, z).float()
        if cfg.sphere_norm:
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        loss, info = candidate_attraction_loss(
            emb, cemb, cmask, weight=a.candidate_weight,
            temperature=a.candidate_temp, repel_intra=a.repel_intra,
            sphere=cfg.sphere_norm,
            sphere_step=cfg.sphere_step, sphere_geo_max=cfg.sphere_geo_max)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_ddp.parameters(), cfg.grad_clip)
        opt.step()
        ema_update(ema, gen, cfg.ema_decay)
        if main_rank and step % cfg.log_every == 0:
            print(f"[step {step}] loss={loss.item():.5f} "
                  f"cand_cos={info['candidate_cos']:.4f} "
                  f"cand_dist={info['candidate_dist']:.4f} "
                  f"cand_H={info['candidate_entropy']:.3f} "
                  f"mask={info['candidate_mask_frac']:.3f} "
                  f"intra={a.repel_intra:.2f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
        if main_rank and step > 0 and step % a.eval_every == 0:
            metrics, shown = eval_one_pass(ema, test_ds, embedder, tok, cfg,
                                           device, a.eval_queries)
            print("[eval-test {}] {}".format(
                step, " ".join(f"{k}={v:.3f}" for k, v in metrics.items())),
                flush=True)
            for s in shown:
                print("  " + s.replace("\n", "\n  "), flush=True)
            if a.eval_train_queries > 0:
                train_metrics, _ = eval_one_pass(
                    ema, train_ds, embedder, tok, cfg, device,
                    a.eval_train_queries)
                print("[eval-train {}] {}".format(
                    step, " ".join(
                        f"{k}={v:.3f}" for k, v in train_metrics.items())),
                    flush=True)
            torch.cuda.empty_cache()
        if main_rank and step > 0 and step % a.save_every == 0:
            path = os.path.join(cfg.checkpoint_dir, f"step_{step}.pt")
            torch.save({"step": step, "model": gen.state_dict(),
                        "ema": ema.state_dict(), "cfg": cfg,
                        "optimizer": opt.state_dict()}, path)
            print(f"[ckpt] {path}", flush=True)
        if ddp and step > 0 and step % a.save_every == 0:
            dist.barrier()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
