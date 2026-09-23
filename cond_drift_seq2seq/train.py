"""DDP training for one-step WMT14 De-En and XSum conditional generation."""

import argparse
import copy
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .cond_drift_loss import cond_repair_loss, cosine_ce_loss
from .cond_generator import CondDriftGenerator
from .config import Seq2SeqDriftConfig
from .data import build_dataset
from .evaluate import run_eval
from .qwen_features import QwenEmbedder, load_qwen
from .qwen_teacher import build_repairs_cond
from seq_drifting_common.ema import ema_update


def build_cfg(args):
    cfg = Seq2SeqDriftConfig()
    cfg.dataset_name = args.dataset_name
    if args.dataset_name == "xsum":
        cfg.data_dir = "data/xsum"
        cfg.query_len = 1024
        cfg.queries_per_step = 1
        cfg.eval_queries = 100
        cfg.eval_batch_size = 2
        cfg.gradient_checkpointing = True
    for key, value in vars(args).items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset_name",
                        choices=["wmt14_de_en", "xsum"], default="wmt14_de_en")
    parser.add_argument("--data-dir")
    parser.add_argument("--train-split", dest="train_split", default=None)
    parser.add_argument("--eval-split", choices=["validation", "test"])
    parser.add_argument("--condition-len", dest="query_len", type=int)
    parser.add_argument("--target-len", dest="resp_len", type=int)
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-eval", type=int)

    parser.add_argument("--teacher", dest="teacher_model")
    parser.add_argument("--no-sphere", dest="sphere_norm",
                        action="store_false", default=None)
    parser.add_argument("--sphere-step", choices=["proj", "retract", "geodesic"])
    parser.add_argument("--sphere-geo-max", type=float)
    parser.add_argument("--decode-chunk", type=int)
    parser.add_argument("--no-mask-illegal", dest="mask_illegal",
                        action="store_false", default=None)

    parser.add_argument("--gold-weight", type=float)
    parser.add_argument("--teacher-weight", type=float)
    parser.add_argument("--pos-teacher-decay", type=float,
                        help="teacher weight at position T-1 (linear ramp from 1.0; "
                             "default 1.0 = flat)")
    parser.add_argument("--pos-gold-boost", type=float,
                        help="gold weight multiplier at position T-1 (linear ramp from 1.0; "
                             "default 1.0 = flat)")
    parser.add_argument("--eos-repel", type=float,
                        help="repulsion strength pushing content positions away from the "
                             "EOS token embedding before the gold EOS position (0=off)")
    parser.add_argument("--eos-repel-tail", type=int,
                        help="only apply EOS repulsion to the last N positions before "
                             "the gold EOS (0 = all content positions)")
    parser.add_argument("--gold-warmup-steps", type=int)
    parser.add_argument("--gold-warmup-ramp", type=int)
    parser.add_argument("--gold-decay-steps", type=int,
                        help="linearly decay gold_weight to gold_weight*gold_min_ratio "
                             "over this many steps, counted from the resume step (0=off)")
    parser.add_argument("--gold-min-ratio", type=float,
                        help="floor for the gold decay, as a fraction of --gold-weight")
    parser.add_argument("--ce-weight", type=float)
    parser.add_argument("--ce-tau", type=float)
    parser.add_argument("--ce-ramp", type=int)
    parser.add_argument("--no-unforbid-gold", dest="unforbid_gold",
                        action="store_false", default=None)
    parser.add_argument("--attract-temp", type=float)
    parser.add_argument("--div-mask", choices=["none", "prefix", "first", "teacher"])
    parser.add_argument("--repel", type=float)
    parser.add_argument("--repel-intra", type=float)
    parser.add_argument("--repel-whole-batch", dest="repel_block",
                        action="store_false", default=None)
    parser.add_argument("--n-pos", type=int)
    parser.add_argument("--prob-thresh", type=float)
    parser.add_argument("--support", choices=["thresh", "nucleus"])
    parser.add_argument("--nucleus-p", type=float)
    parser.add_argument("--no-repeat-ngram", type=int)
    parser.add_argument("--no-repeat-window", type=int)

    parser.add_argument("--queries-per-step", type=int)
    parser.add_argument("--k-samples", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--nhead", type=int)
    parser.add_argument("--ffn-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--noise-dim", type=int)
    parser.add_argument("--no-query-cross-attn", dest="query_cross_attn",
                        action="store_false", default=None)
    parser.add_argument("--ls-init", type=float)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=None)

    parser.add_argument("--steps", type=int)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--ema-decay", type=float)
    parser.add_argument("--lr-schedule", choices=["none", "cosine"])
    parser.add_argument("--lr-warmup", type=int)
    parser.add_argument("--lr-min-ratio", type=float)
    parser.add_argument("--lr-decay-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-bf16", dest="use_bf16",
                        action="store_false", default=None)

    parser.add_argument("--log-every", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--eval-queries", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--ckpt-dir", dest="checkpoint_dir")
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_cfg(args)
    if cfg.decode_chunk <= 0:
        raise ValueError("decode_chunk must be positive")
    if cfg.queries_per_step <= 0 or cfg.k_samples <= 0:
        raise ValueError("queries_per_step and k_samples must be positive")
    if cfg.gold_weight <= 0 and cfg.teacher_weight <= 0:
        raise ValueError("at least one training signal must be enabled")

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
    is_main = rank == 0
    torch.manual_seed(cfg.seed + rank)
    sample_rng = torch.Generator().manual_seed(cfg.seed + 10_000 * rank)
    if is_main:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    frozen_teacher, tokenizer = load_qwen(
        cfg.teacher_model, str(device), "bf16" if cfg.use_bf16 else "fp32")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    embedder = QwenEmbedder(cfg, tokenizer, device)
    cfg.embed_dim, cfg.vocab_size = embedder.H, embedder.V
    train_ds = build_dataset(cfg, tokenizer, cfg.train_split)
    eval_ds = build_dataset(cfg, tokenizer, cfg.eval_split)

    if cfg.unforbid_gold:
        # The allowlist in qwen_features is a heuristic over each token's DECODED string, so
        # it rejects the BPE byte fragments that spell accented characters -- which do occur
        # in real reference translations. Forbidding them makes those positions both
        # unlearnable (no attracting target) and undecodable (never a NN candidate), a hard
        # ceiling on token accuracy. A token that appears in a gold target is legal by
        # construction, so let the data override the heuristic. Every rank scans the same
        # memmaps deterministically and so reaches the same mask; no broadcast needed.
        seen = torch.zeros(embedder.V, dtype=torch.bool)
        for ds in (train_ds, eval_ds):
            rows, tlen = ds.target_ids, ds.target_lens
            for s in range(0, len(ds), 65536):
                block = torch.from_numpy(
                    np.asarray(rows[s:s + 65536], dtype=np.int64))
                lens = torch.from_numpy(
                    np.asarray(tlen[s:s + 65536], dtype=np.int64))
                valid = torch.arange(block.shape[1]).unsqueeze(0) < lens.unsqueeze(1)
                ids = block[valid]
                seen[ids[(ids >= 0) & (ids < embedder.V)]] = True
        freed = int((seen & embedder.forbid_mask.cpu()).sum())
        if freed:
            embedder.forbid_mask = (embedder.forbid_mask
                                    & ~seen.to(embedder.forbid_mask.device))
        if is_main:
            print(f"[unforbid] gold tokens={int(seen.sum())} freed={freed} "
                  f"forbidden now={int(embedder.forbid_mask.sum())}/{embedder.V}")
    micro_query_batch = cfg.queries_per_step * world_size
    if cfg.grad_accum_steps > 0:
        grad_accum = cfg.grad_accum_steps
    else:
        if cfg.global_batch_size % micro_query_batch:
            raise ValueError(
                f"global_batch_size={cfg.global_batch_size} must be divisible by "
                f"queries_per_step*world_size={micro_query_batch}")
        grad_accum = cfg.global_batch_size // micro_query_batch
    if grad_accum <= 0:
        raise ValueError("gradient accumulation must be positive")
    effective_query_batch = micro_query_batch * grad_accum
    total_steps = cfg.steps
    derived_steps = total_steps <= 0
    if total_steps <= 0:
        total_steps = math.ceil(len(train_ds) / effective_query_batch * cfg.epochs)
    cfg.grad_accum_steps = grad_accum
    cfg.steps = total_steps

    if cfg.repel_abs:
        idx = torch.randint(0, embedder.V, (4096,))
        cfg.abs_scale = float(torch.pdist(embedder.wte[idx].to(device)).mean())

    # Unit vocabulary rows for the CE logits. With sphere_norm the embedder already holds
    # exactly this, so reuse it rather than keeping a second (V, H) copy on the GPU.
    ce_wte_unit = None
    if cfg.ce_weight > 0:
        if cfg.sphere_norm:
            ce_wte_unit = embedder.compute_wte
        else:
            w = embedder.compute_wte.float()
            ce_wte_unit = (w / w.norm(dim=-1, keepdim=True).clamp_min(1e-8)).to(
                embedder.compute_wte.dtype)
        if is_main:
            print(f"[ce] weight={cfg.ce_weight} tau={cfg.ce_tau} ramp={cfg.ce_ramp} "
                  f"logits={cfg.queries_per_step * cfg.resp_len}x{embedder.V} (pre-mask)")

    gen = CondDriftGenerator(cfg, cfg.embed_dim).to(device)
    ema = copy.deepcopy(gen).eval()
    for param in ema.parameters():
        param.requires_grad_(False)
    gen_ddp = DDP(gen, device_ids=[local_rank]) if ddp else gen
    opt = torch.optim.AdamW(gen.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_step = 1
    best_metric = float("-inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        gen.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        if "optimizer" in state:
            opt.load_state_dict(state["optimizer"])
            # load_state_dict restores param_groups too, which silently overwrites --lr with
            # whatever the checkpoint was trained at. Re-apply the requested values so the
            # command line wins; the Adam moment estimates in state["state"] are kept.
            for group in opt.param_groups:
                group["lr"] = cfg.lr
                group["weight_decay"] = cfg.weight_decay
        start_step = int(state.get("step", 0)) + 1
        best_metric = float(state.get("best_metric", best_metric))
        if is_main:
            print(f"[resume] {args.resume} -> step {start_step}")

    # Built after --resume rather than beside the optimizer: opt.load_state_dict()
    # replaces param_groups wholesale, which would drop the initial_lr LambdaLR writes
    # there and leave base_lrs pointing at the checkpoint's lr instead of --lr.
    sched = None
    if cfg.lr_schedule != "none":
        decay_steps = cfg.lr_decay_steps if cfg.lr_decay_steps > 0 else total_steps
        warm = max(0, cfg.lr_warmup)
        floor = cfg.lr_min_ratio

        def lr_lambda(done):
            # `done` counts updates already taken, so step N is evaluated at done=N-1.
            if warm > 0 and done < warm:
                return (done + 1) / float(warm)
            progress = (done - warm) / max(1, decay_steps - warm)
            progress = min(1.0, max(0.0, progress))
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        # Fast-forward on resume so the decay continues instead of restarting at peak.
        # Restarting at peak lr is what let the resumed runs walk one-way out of the
        # good region: every checkpoint on that trajectory scored worse than the last.
        for _ in range(start_step - 1):
            sched.step()
        if is_main:
            print(f"[lr] schedule={cfg.lr_schedule} warmup={warm} "
                  f"decay_steps={decay_steps:,} min_ratio={floor} "
                  f"lr_now={opt.param_groups[0]['lr']:.3e}")
            if start_step - 1 >= decay_steps:
                print("[lr] WARNING: resume step is already past decay_steps, so the "
                      "whole run would train at the lr floor. Pass --lr-decay-steps "
                      "to extend the horizon.")

    if is_main:
        # Print the lr the optimizer will ACTUALLY use, read back from param_groups rather
        # than from cfg: resuming used to overwrite it from the checkpoint, and with no log
        # line to check against, several runs silently trained at the wrong lr.
        print(f"[seq2seq] dataset={cfg.dataset_name} train={len(train_ds):,} "
              f"eval={len(eval_ds):,} condition={cfg.query_len} target={cfg.resp_len}")
        print(f"[seq2seq] lr={opt.param_groups[0]['lr']:.3e} "
              f"wd={opt.param_groups[0]['weight_decay']} "
              f"grad_clip={cfg.grad_clip} ema={cfg.ema_decay}")
        print(f"[seq2seq] params={sum(p.numel() for p in gen.parameters())/1e6:.1f}M "
              f"world={world_size} micro/rank={cfg.queries_per_step} accum={grad_accum} "
              f"effective_query_batch={effective_query_batch} K={cfg.k_samples}")
        budget = (f"~{cfg.epochs:g} epochs" if derived_steps else "explicit step budget")
        print(f"[seq2seq] steps={total_steps:,} ({budget}) "
              f"gold={cfg.gold_weight} teacher={cfg.teacher_weight} "
              f"warmup={cfg.gold_warmup_steps} ramp={cfg.gold_warmup_ramp}")

    q_per_rank = cfg.queries_per_step
    wall_start = time.time()
    for step in range(start_step, total_steps + 1):
        gen_ddp.train()
        opt.zero_grad(set_to_none=True)
        sums = {"loss": 0.0, "scale": 0.0, "gold_f": 0.0,
                "gold_rmse": 0.0, "gold_l2": 0.0, "gold_cos": 0.0,
                "n_implaus": 0.0, "ce": 0.0, "ce_acc": 0.0, "ce_skip": 0.0,
                "div_pos": 0.0, "div_keep": 0.0}

        warmup = cfg.gold_warmup_steps > 0 and step <= cfg.gold_warmup_steps
        if warmup:
            teacher_frac = 0.0
        elif cfg.gold_warmup_steps > 0 and cfg.gold_warmup_ramp > 0:
            teacher_frac = min(
                1.0, (step - cfg.gold_warmup_steps) / float(cfg.gold_warmup_ramp))
        else:
            teacher_frac = 1.0
        effective_teacher = cfg.teacher_weight * teacher_frac
        effective_repel = cfg.repel * teacher_frac
        effective_intra = cfg.repel_intra * teacher_frac
        step_k = 1 if warmup else cfg.k_samples
        # CE ramps from the step training RESUMED at, so adding it to a run in progress
        # does not jolt a converged model with a full-strength new gradient.
        if cfg.ce_weight > 0 and cfg.ce_ramp > 0:
            effective_ce = cfg.ce_weight * min(
                1.0, max(0, step - start_step) / float(cfg.ce_ramp))
        else:
            effective_ce = cfg.ce_weight
        # Gold decay: linear from gold_weight down to gold_weight*gold_min_ratio over
        # gold_decay_steps, counted from the resume step like the CE ramp.
        if cfg.gold_decay_steps > 0 and cfg.gold_min_ratio < 1.0:
            gold_prog = min(1.0, max(0, step - start_step) / float(cfg.gold_decay_steps))
            gold_frac_sched = 1.0 - (1.0 - cfg.gold_min_ratio) * gold_prog
        else:
            gold_frac_sched = 1.0
        effective_gold = cfg.gold_weight * gold_frac_sched

        for micro in range(grad_accum):
            sync_context = (gen_ddp.no_sync() if ddp and micro < grad_accum - 1
                            else nullcontext())
            with sync_context:
                query_ids, query_mask, target_ids, target_mask, _ = train_ds.sample(
                    q_per_rank, generator=sample_rng)
                query_ids = query_ids.to(device)
                query_mask = query_mask.to(device)
                target_ids = target_ids.to(device).repeat_interleave(step_k, dim=0)
                target_mask = target_mask.to(device).repeat_interleave(step_k, dim=0)
                query_ids_g = query_ids.repeat_interleave(step_k, dim=0)
                query_mask_g = query_mask.repeat_interleave(step_k, dim=0)
                query_emb = embedder.query_embeds(query_ids_g)
                z = gen.sample_z(q_per_rank * step_k, cfg.noise_dim, cfg.temp, device)
                amp = bool(cfg.use_bf16 and device.type == "cuda")
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    emb = gen_ddp(query_emb, query_mask_g, z)
                emb = emb.float()
                if cfg.sphere_norm:
                    emb = torch.nn.functional.normalize(emb, dim=-1)

                gold_wte = embedder.lookup(target_ids).float()
                if effective_teacher > 0:
                    with torch.no_grad():
                        response_tokens = embedder.decode(emb)
                        support, valid, implausible = build_repairs_cond(
                            query_ids, query_mask, response_tokens, cfg, device,
                            forbid_mask=embedder.forbid_mask)
                        repair_wte = embedder.lookup(support).float()
                else:
                    # Warmup uses the original loss with teacher_weight=0. A singleton
                    # placeholder keeps its input contract intact while skipping decode
                    # and the teacher forward entirely.
                    repair_wte = gold_wte.unsqueeze(2)
                    valid = target_mask.unsqueeze(2)
                    implausible = torch.zeros_like(target_mask)

                loss, info = cond_repair_loss(
                    emb, repair_wte, valid, implausible, group_size=step_k,
                    temp=cfg.attract_temp, repel=effective_repel,
                    repel_perpos=cfg.repel_perpos, repel_intra=effective_intra,
                    repel_abs=cfg.repel_abs, abs_scale=cfg.abs_scale,
                    free_prefix=cfg.free_prefix, sphere=cfg.sphere_norm,
                    sphere_step=cfg.sphere_step, sphere_geo_max=cfg.sphere_geo_max,
                    temp_list=cfg.temp_list, gold_wte=gold_wte,
                    gold_mask=target_mask, gold_weight=effective_gold,
                    loss_mask=target_mask, teacher_weight=effective_teacher,
                    repel_block=cfg.repel_block, div_mask=cfg.div_mask,
                    pos_teacher_decay=cfg.pos_teacher_decay,
                    pos_gold_boost=cfg.pos_gold_boost,
                    eos_wte=embedder.wte[embedder.eos_id].to(emb.device),
                    eos_repel=cfg.eos_repel,
                    eos_repel_tail=cfg.eos_repel_tail)

                # Discriminative auxiliary term. `emb` is already sphere-normalised above
                # when sphere_norm is set, so these logits are exactly the cosines the
                # evaluation decode ranks -- the CE optimises the decision the decoder makes.
                if effective_ce > 0 and ce_wte_unit is not None:
                    ce_loss, ce_info = cosine_ce_loss(
                        emb, target_ids, ce_wte_unit, loss_mask=target_mask,
                        tau=cfg.ce_tau, forbid_mask=embedder.forbid_mask)
                    loss = loss + effective_ce * ce_loss
                    sums["ce"] += ce_info["ce"]
                    sums["ce_acc"] += ce_info["ce_acc"]
                    sums["ce_skip"] += ce_info["ce_skip"]

                (loss / grad_accum).backward()

            sums["loss"] += float(loss.detach())
            sums["scale"] += info["scale"]
            sums["gold_f"] += info["gold_f"]
            sums["gold_rmse"] += info["gold_rmse"]
            sums["gold_l2"] += info["gold_l2"]
            sums["gold_cos"] += info["gold_cos"]
            sums["n_implaus"] += info["n_implaus"]
            sums["div_pos"] += info.get("div_pos", 0.0)
            sums["div_keep"] += info.get("div_keep", 0.0)

        grad_norm = torch.nn.utils.clip_grad_norm_(gen_ddp.parameters(), cfg.grad_clip)
        opt.step()
        if sched is not None:
            sched.step()
        ema_update(ema, gen, cfg.ema_decay)

        if is_main and step % cfg.log_every == 0:
            avg = {key: value / grad_accum for key, value in sums.items()}
            param_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
            for param in gen_ddp.parameters():
                param_norm_sq += param.detach().float().pow(2).sum()
            param_norm = param_norm_sq.sqrt()
            grad_ratio = float(grad_norm) / float(param_norm.clamp_min(1e-12))
            phase = "warmup" if warmup else "train"
            # ce_acc is the teacher-forced argmax accuracy over the vocabulary -- the
            # quantity the 12.4%-token-accuracy diagnostic measured, now visible per step.
            ce_text = (f"ce={avg['ce']:.4f} ce_acc={avg['ce_acc']:.4f} "
                       f"ce_skip={avg['ce_skip']:.4f} "
                       f"cw={effective_ce:.2f} " if effective_ce > 0 else "")
            # div_pos is the progress signal to watch under --div-mask: as the student
            # improves, the teacher's first complaint moves later in the response, so this
            # climbs even while the loss is flat. div_keep is the fraction of supervised
            # positions still receiving force.
            div_text = (f"div_pos={avg['div_pos']:.2f} div_keep={avg['div_keep']:.3f} "
                        if cfg.div_mask != "none" else "")
            print(f"[step {step}|{phase}] loss={avg['loss']:.4f} "
                  f"scale={avg['scale']:.3f} gold_f={avg['gold_f']:.4f} "
                  f"gold_rmse={avg['gold_rmse']:.3e} gold_l2={avg['gold_l2']:.3e} "
                  f"gold_cos={avg['gold_cos']:.5f} "
                  f"{ce_text}{div_text}"
                  f"n_implaus={avg['n_implaus']:.1f} K={step_k} "
                  f"tw={effective_teacher:.2f} gw={effective_gold:.2f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} "
                  f"grad={float(grad_norm):.3e} grad/param={grad_ratio:.3e} "
                  f"t={time.time()-wall_start:.0f}s")

        do_eval = step % cfg.eval_every == 0
        do_save = step % cfg.save_every == 0
        if ddp and (do_eval or do_save):
            dist.barrier()
        if is_main and do_eval:
            metrics, shown = run_eval(
                ema, embedder, eval_ds, cfg, tokenizer, device,
                cfg.eval_queries, cfg.eval_n_show)
            metric_text = " ".join(f"{key}={value:.3f}" for key, value in metrics.items())
            print(f"  [eval {step}] {metric_text}")
            for sample in shown:
                print("   " + sample.replace("\n", "\n   "))
            primary = "bleu" if cfg.dataset_name == "wmt14_de_en" else "rougeL"
            score = metrics.get(primary, float("-inf"))
            if cfg.eval_split == "validation" and score > best_metric:
                best_metric = score
                path = os.path.join(cfg.checkpoint_dir, "best.pt")
                torch.save({"step": step, "model": gen.state_dict(),
                            "ema": ema.state_dict(), "optimizer": opt.state_dict(),
                            "cfg": cfg, "embed_dim": cfg.embed_dim,
                            "best_metric": best_metric}, path)
                print(f"  [best] {primary}={best_metric:.3f} -> {path}")
            torch.cuda.empty_cache()
        if is_main and do_save:
            path = os.path.join(cfg.checkpoint_dir, f"step_{step}.pt")
            torch.save({"step": step, "model": gen.state_dict(), "ema": ema.state_dict(),
                        "optimizer": opt.state_dict(), "cfg": cfg,
                        "embed_dim": cfg.embed_dim,
                        "best_metric": best_metric}, path)
            print(f"  [ckpt] {path}")
        if ddp and (do_eval or do_save):
            dist.barrier()

    if is_main:
        print("Training complete.")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
