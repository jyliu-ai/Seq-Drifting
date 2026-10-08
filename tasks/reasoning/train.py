"""Train conditional drifting with query-specific multi-solution attraction.

Unlike teacher mode, this never asks a frozen LM to score the model's own
possibly-wrong prefix.  Each query has a candidate set made from real dataset
solutions.  The candidate closest to the current generated sequence receives
the largest attraction weight.

Run with `python -m tasks.reasoning.train`.
"""
import argparse
import copy
import os
import re
import time
from decimal import Decimal, InvalidOperation

import torch
from common.checkpoint import load_checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from common.config import ReasoningConfig as CondDriftConfig
from models.embeddings import load_teacher as load_qwen, FrozenEmbedder as QwenEmbedder
from models.generators import QwenCondGenerator, CondDriftGenerator
from common.candidates import load_candidate_groups, summarize_groups, CandidateDataset, eval_one_pass
from common.losses import candidate_attraction_loss
from common.ema import ema_update


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
        blob = load_checkpoint(a.resume, map_location=checkpoint_device)
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
