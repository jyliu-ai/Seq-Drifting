"""Conditional text-drifting training (single process / single GPU).

  query -> generator (one forward pass) -> response embeddings -> decode to tokens
       -> Qwen teacher scores [query ++ response] -> per-position support set
       -> drift toward the support set on the sphere (block-diagonal repulsion per query).

Run:
  python -m cond_drift.train --dataset gsm8k --teacher Qwen/Qwen3-4B \
      --queries-per-step 32 --k-samples 4 --resp-len 256 --steps 200000 \
      --ckpt-dir runs/cond_gsm8k_qwen4b

Smoke test (cheap: tiny teacher, few steps) to shake out shape bugs:
  python -m cond_drift.train --teacher Qwen/Qwen3-0.6B \
      --queries-per-step 4 --k-samples 2 --resp-len 64 --query-len 64 \
      --steps 20 --eval-every 10 --eval-queries 8 --no-bf16
"""
import argparse
import copy
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import CondDriftConfig
from .data import build_dataset
from .qwen_features import load_qwen, QwenEmbedder
from .qwen_teacher import build_repairs_cond
from .cond_generator import CondDriftGenerator
from .cond_drift_loss import cond_repair_loss
from .evaluate import run_eval
from seq_drifting_common.ema import ema_update


def build_cfg(a) -> CondDriftConfig:
    cfg = CondDriftConfig()
    for k, v in vars(a).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", dest="dataset_name", default="gsm8k",
                   choices=["gsm8k", "gsm8k_local", "math", "jsonl", "owt"])
    p.add_argument("--gsm8k-json", dest="local_json",
                   help="dataset=gsm8k_local: path to the nested GSM8K.json (offline)")
    p.add_argument("--owt-dir", dest="owt_dir",
                   help="dataset=owt: dir of *.jsonl.zst OpenWebText2 shards (offline)")
    p.add_argument("--owt-docs", dest="owt_docs", type=int,
                   help="dataset=owt: number of docs (prefix->continuation examples) to load")
    p.add_argument("--jsonl-path", dest="jsonl_path")
    p.add_argument("--teacher", dest="teacher_model", default="Qwen/Qwen3-4B")
    p.add_argument("--query-len", dest="query_len", type=int)
    p.add_argument("--resp-len", dest="resp_len", type=int)
    p.add_argument("--n-train", dest="n_train", type=int)
    p.add_argument("--queries-per-step", dest="queries_per_step", type=int)
    p.add_argument("--k-samples", dest="k_samples", type=int)
    p.add_argument("--n-pos", dest="n_pos", type=int)
    p.add_argument("--prob-thresh", dest="prob_thresh", type=float)
    p.add_argument("--support", dest="support", choices=["thresh", "nucleus"])
    p.add_argument("--nucleus-p", dest="nucleus_p", type=float)
    p.add_argument("--no-repeat-ngram", dest="no_repeat_ngram", type=int)
    p.add_argument("--no-repeat-window", dest="no_repeat_window", type=int)
    p.add_argument("--gold-weight", dest="gold_weight", type=float,
                   help="position-aligned attraction toward the REAL response (0=off, try 2-5)")
    p.add_argument("--teacher-weight", dest="teacher_weight", type=float,
                   help="scale the Qwen-support attraction; 0 = pure gold supervision")
    p.add_argument("--eval-on-train", dest="eval_on_train", action="store_true", default=None,
                   help="overfit diagnostic: run eval on the TRAIN queries (memorisation check)")
    p.add_argument("--attract-temp", dest="attract_temp", type=float)
    p.add_argument("--repel", dest="repel", type=float)
    p.add_argument("--repel-intra", dest="repel_intra", type=float)
    p.add_argument("--repel-whole-batch", dest="repel_block", action="store_false", default=None,
                   help="repel across the WHOLE batch (default: only within the same prefix)")
    p.add_argument("--no-repel-perpos", dest="repel_perpos", action="store_false", default=None)
    p.add_argument("--sphere-step", dest="sphere_step", choices=["proj", "retract", "geodesic"])
    p.add_argument("--sphere-geo-max", dest="sphere_geo_max", type=float)
    p.add_argument("--no-query-cross-attn", dest="query_cross_attn", action="store_false",
                   default=None, help="disable the ungated query cross-attention")
    p.add_argument("--ls-init", dest="ls_init", type=float,
                   help="LayerScale init (default 1e-4); raise to 0.1/1.0 to un-gate the stack")
    p.add_argument("--d-model", dest="d_model", type=int)
    p.add_argument("--nhead", dest="nhead", type=int)
    p.add_argument("--ffn-dim", dest="ffn_dim", type=int)
    p.add_argument("--num-layers", dest="num_layers", type=int)
    p.add_argument("--uncond-warmup-steps", dest="uncond_warmup_steps", type=int,
                   help="fill context with a constant token for the first N steps (uncond warmup)")
    p.add_argument("--warmup-ctx", dest="warmup_ctx", choices=["pad", "eos", "bos"],
                   help="which constant token fills the context during warmup")
    p.add_argument("--noise-dim", dest="noise_dim", type=int)
    p.add_argument("--lr", dest="lr", type=float)
    p.add_argument("--steps", dest="steps", type=int)
    p.add_argument("--eval-every", dest="eval_every", type=int)
    p.add_argument("--eval-queries", dest="eval_queries", type=int)
    p.add_argument("--save-every", dest="save_every", type=int)
    p.add_argument("--ckpt-dir", dest="checkpoint_dir")
    p.add_argument("--no-bf16", dest="use_bf16", action="store_false", default=None)
    p.add_argument("--resume", dest="resume", default=None,
                   help="path to a step_N.pt checkpoint to resume from; "
                        "if omitted, auto-detects the latest one in --ckpt-dir")
    a = p.parse_args()
    cfg = build_cfg(a)
    if cfg.dataset_name == "owt":                          # continuation: no exact-match answer
        cfg.eval_accuracy = False
        cfg.use_chat_template = False

    # ── DDP (torchrun --nproc_per_node=N) ────────────────────────────────────
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = (rank == 0)
    torch.manual_seed(cfg.seed + rank)                    # each rank samples its OWN queries / z
    if is_main:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    _, tokenizer = load_qwen(cfg.teacher_model, str(device), "bf16" if cfg.use_bf16 else "fp32")
    # constant token that fills the context during unconditional warmup
    _ctx_id = {"pad": tokenizer.pad_token_id, "eos": tokenizer.eos_token_id,
               "bos": tokenizer.bos_token_id}.get(getattr(cfg, "warmup_ctx", "pad"))
    warmup_fill_id = _ctx_id if _ctx_id is not None else tokenizer.eos_token_id
    if is_main and cfg.uncond_warmup_steps > 0:
        print(f"[cond] warmup fills context with token id {warmup_fill_id} "
              f"({tokenizer.decode([warmup_fill_id])!r}), normal attention")
    embedder = QwenEmbedder(cfg, tokenizer, device)
    cfg.embed_dim, cfg.vocab_size = embedder.H, embedder.V
    print(f"[cond] embed_dim={cfg.embed_dim} vocab={cfg.vocab_size} "
          f"batch=Q{cfg.queries_per_step}xK{cfg.k_samples}={cfg.queries_per_step*cfg.k_samples} "
          f"resp_len={cfg.resp_len} no_repeat={cfg.no_repeat_ngram}")

    train_ds = build_dataset(cfg, tokenizer, "train")
    eval_ds = build_dataset(cfg, tokenizer, "test")
    if getattr(cfg, "eval_on_train", False):             # overfit diagnostic: eval the TRAIN set
        eval_ds = train_ds
    print(f"[cond] train queries={len(train_ds)} eval queries={len(eval_ds)}"
          f"{' (EVAL ON TRAIN)' if getattr(cfg,'eval_on_train',False) else ''}")

    if cfg.repel_abs:                                     # fixed manifold unit = mean pairwise dist
        idx = torch.randint(0, embedder.V, (4096,))
        w = embedder.wte[idx].to(device)
        cfg.abs_scale = float(torch.pdist(w).mean())

    gen = CondDriftGenerator(cfg, cfg.embed_dim).to(device)
    ema = copy.deepcopy(gen).eval()
    for pr in ema.parameters():
        pr.requires_grad_(False)
    # warmup now fills the context with a constant token under NORMAL attention (not masked), so
    # every param -- including the conditioning path -- gets gradient every step; no need for the
    # (slow) find_unused_parameters.
    gen_ddp = (DDP(gen, device_ids=[local_rank]) if ddp else gen)
    opt = torch.optim.AdamW(gen.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_step = 0
    resume_path = getattr(a, "resume", None)
    if resume_path is None and cfg.checkpoint_dir:
        # auto-detect latest step_N.pt in the checkpoint dir
        import glob as _glob
        candidates = _glob.glob(os.path.join(cfg.checkpoint_dir, "step_*.pt"))
        if candidates:
            resume_path = max(candidates,
                              key=lambda p: int(os.path.basename(p)[5:-3]))
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        gen.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_step = int(ckpt.get("step", 0)) + 1
        if is_main:
            print(f"[resume] loaded {resume_path}, resuming from step {start_step}")
    # ─────────────────────────────────────────────────────────────────────────

    if is_main:
        print(f"[cond] generator params={sum(p.numel() for p in gen.parameters())/1e6:.1f}M "
              f"world_size={world_size} effective_batch="
              f"{cfg.queries_per_step*cfg.k_samples*world_size} "
              f"uncond_warmup={cfg.uncond_warmup_steps}")

    Q, K = cfg.queries_per_step, cfg.k_samples
    t0 = time.time()
    for step in range(start_step, cfg.steps + 1):
        gen_ddp.train()
        warmup = step < cfg.uncond_warmup_steps           # UNCONDITIONAL phase
        q_ids, q_msk, r_ids, r_msk, _ = train_ds.sample(Q)
        q_ids, q_msk = q_ids.to(device), q_msk.to(device)
        if warmup:
            # fill the context with a constant special token (pad/eos) and keep NORMAL attention,
            # instead of masking it out. So the generator/teacher condition on a clean, consistent
            # dummy prefix (GPT-2 pad==eos==<|endoftext|> = document start -> truly unconditional),
            # resp[0] gets proper context, and the conditioning params (in_query/query_cross) also
            # receive gradient during warmup (so find_unused_parameters can stay off).
            q_ids = torch.full_like(q_ids, warmup_fill_id)
            q_msk = torch.ones_like(q_msk)
        qi_g = q_ids.repeat_interleave(K, dim=0)          # (G, Lq)
        qm_g = q_msk.repeat_interleave(K, dim=0)
        # supervision mask over the Lr response positions. With a GOLD target it is the gold
        # tokens + EOS delimiter (pad tail excluded). PURE TEACHER (gold_weight=0, e.g. the OWT
        # continuation task) has no pad tail -- every generated position is teacher-supervised
        # -- so the mask is None (all Lr positions count).
        gold_wte = None
        if cfg.gold_weight > 0:                           # real response as the positive
            loss_mask = r_msk.to(device).repeat_interleave(K, dim=0)
            gold_wte = embedder.lookup(r_ids.to(device).repeat_interleave(K, dim=0)).float()
        else:
            loss_mask = None

        query_emb = embedder.query_embeds(qi_g)           # (G, Lq, H) no grad
        z = gen.sample_z(Q * K, cfg.noise_dim, cfg.temp, device)
        emb = gen_ddp(query_emb, qm_g, z)                 # (G, Lr, H) grad (DDP all-reduces)
        if cfg.sphere_norm:
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        with torch.no_grad():
            resp_tokens = embedder.decode(emb)            # (G, Lr)
            support, valid, implausible = build_repairs_cond(
                q_ids, q_msk, resp_tokens, cfg, device, forbid_mask=embedder.forbid_mask)
            repair_wte = embedder.lookup(support).float()  # (G, Lr, n_pos, H)

        loss, info = cond_repair_loss(
            emb, repair_wte, valid, implausible, group_size=K,
            temp=cfg.attract_temp, repel=cfg.repel, repel_perpos=cfg.repel_perpos,
            repel_intra=cfg.repel_intra, repel_abs=cfg.repel_abs, abs_scale=cfg.abs_scale,
            free_prefix=cfg.free_prefix, sphere=cfg.sphere_norm, sphere_step=cfg.sphere_step,
            sphere_geo_max=cfg.sphere_geo_max, temp_list=cfg.temp_list,
            gold_wte=gold_wte, gold_mask=(loss_mask if cfg.gold_weight > 0 else None),
            gold_weight=cfg.gold_weight,
            loss_mask=loss_mask, teacher_weight=cfg.teacher_weight,
            repel_block=cfg.repel_block)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_ddp.parameters(), cfg.grad_clip)
        opt.step()
        ema_update(ema, gen, cfg.ema_decay)

        if is_main and step == cfg.uncond_warmup_steps and cfg.uncond_warmup_steps > 0:
            print(f"[cond] === unconditional warmup done ({step} steps); "
                  f"turning ON prefix conditioning ===")

        if is_main and step % cfg.log_every == 0:
            ph = "warmup" if warmup else "cond"
            print(f"[step {step}|{ph}] loss={loss.item():.4f} scale={info['scale']:.3f} "
                  f"n_implaus={info['n_implaus']:.1f} t={time.time()-t0:.0f}s")

        if is_main and step % cfg.eval_every == 0 and step > 0:
            metrics, shown = run_eval(ema, embedder, eval_ds, cfg, tokenizer, device,
                                      cfg.eval_queries, cfg.eval_n_show)
            ms = " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in metrics.items())
            print(f"  [eval {step}] {ms}")
            for s in shown:
                print("   " + s.replace("\n", "\n   "))
            torch.cuda.empty_cache()

        if is_main and step % cfg.save_every == 0 and step > 0:
            path = os.path.join(cfg.checkpoint_dir, f"step_{step}.pt")
            torch.save({"step": step, "model": gen.state_dict(),
                        "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "cfg": cfg, "embed_dim": cfg.embed_dim}, path)
            print(f"  [ckpt] {path}")

    if is_main:
        print("Training complete.")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
