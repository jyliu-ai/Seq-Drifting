"""Training loop for the text drifting sanity check (single-GPU or DDP).

Run (from the package's PARENT dir):
    python -m uncond_drift.train --steps 20000 --gen-per-step 64                  # 1 GPU
    torchrun --standalone --nproc_per_node=4 -m uncond_drift.train --steps 20000  # 4 GPUs

Each step (per rank):
  1. sample continuous z, generate a continuous embedding sequence (B, T, 768)
  2. form its 2-gram token embeddings (consecutive concat) -- NO GPT-2 forward
  3. pick the positive cloud by min-2-gram-distance nearest neighbour over the bank
  4. drift_loss_2gram: pull each generation's matched 2-gram toward the matched
     real 2-gram, repel generations from each other (min-2-gram distance)
  5. backprop into the generator (DDP all-reduces grads)

Eval (rank 0, EMA params): decode via cosine-NN, print n_show samples, report
gen_ppl / entropy (GPT-2 Large) + distinct_seqs / distinct_reals_hit / exact_match.
"""
import argparse
import contextlib
import copy
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import TextDriftConfig
from .data import load_token_chunks
from .drift_loss import drift_loss_2gram, match_loss_2gram, gpt2_repair_loss
from .evaluate import run_eval, nearest_real_report, matched_positives_report
from .generator import TextDriftGenerator
from .gpt2_features import TokenEmbedder
from .gpt2_teacher import build_positives as gpt2_build_positives
from .gpt2_teacher import build_repairs as gpt2_build_repairs
from .memory_bank import TextMemoryBank


def _init_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    return dist.get_rank(), dist.get_world_size(), local_rank


def _cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def pick_device(local_rank: int):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _ema_update(ema_model, model, decay: float):
    for e, p in zip(ema_model.parameters(), model.parameters()):
        e.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for eb, b in zip(ema_model.buffers(), model.buffers()):
        eb.copy_(b)


def train(cfg: TextDriftConfig):
    rank, world_size, local_rank = _init_distributed()
    is_main = (rank == 0)
    ddp = (world_size > 1)
    device = pick_device(local_rank)
    torch.manual_seed(cfg.seed + rank)

    # TF32: the drift path does large fp32 matmuls outside autocast (n-gram
    # construction, bank nearest-neighbour, cosine decode, sinkhorn). PyTorch
    # defaults matmul TF32 to False, so those run as true fp32 and leave the
    # tensor cores idle; 10-bit mantissa is ample for ranking/distances.
    # DISABLED for the entropy-regression test vs ruizhi (which ran true fp32).
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    if is_main:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.output_dir, exist_ok=True)

    if not is_main:
        _barrier()
    tokens, tokenizer = load_token_chunks(cfg, verbose=is_main)
    if is_main:
        _barrier()
    cfg.vocab_size = len(tokenizer)
    bos_id = tokenizer.bos_token_id

    embedder = TokenEmbedder(cfg.embed_model, tokenizer=tokenizer,
                             mask_illegal=cfg.mask_illegal_tokens,
                             sphere=cfg.sphere_norm).to(device).eval()
    # fixed embedding-space unit for absolute repulsion (gpt2 repair + repel_abs):
    # mean pairwise distance over a sample of wte rows -- a CONSTANT (inflation-independent)
    # scale, so the repulsion can't run away with the embedding magnitude.
    with torch.no_grad():
        _s = embedder.wte[torch.randperm(embedder.vocab_size, device=device)[:1024]].float()
        manifold_scale = float(torch.pdist(_s).mean())
        # mean per-row wte norm (~3.1 for GPT-2): the scale the generator output must
        # live on. Raw generator output has |emb|~0.68, a tiny ball, so in EUCLIDEAN
        # mode every real n-gram sits ~wte_shell away while sibling generations are
        # ~0 apart -> the sharp affinity ignores the reals -> drift force is identically
        # 0 (loss=0). Project the euclidean output onto this shell so gens and reals
        # are comparable (mirrors the sphere branch, which uses radius 1).
        wte_shell = float(embedder.wte.float().norm(dim=-1).mean())
    if is_main:
        print(f"  manifold_scale (wte mean pairwise dist) = {manifold_scale:.3f}"
              f" | wte_shell (mean row norm) = {wte_shell:.3f}", flush=True)
        _ws = int(getattr(cfg, "attr_temp_warmup_steps", 0))
        if _ws > 0:
            print(f"  tau warm-up: {getattr(cfg,'attr_temp_warmup_start',1.0):.3f} -> "
                  f"{cfg.attract_temp:.3f} over {_ws} steps", flush=True)
        else:
            print(f"  tau warm-up: OFF (constant tau={cfg.attract_temp:.3f})", flush=True)
    bank = TextMemoryBank(tokens, embedder, cfg, device, verbose=is_main,
                          build_ngram=not cfg.skip_bank)
    raw_model = TextDriftGenerator(cfg, wte=embedder.wte_weight).to(device)
    ema_model = copy.deepcopy(raw_model).to(device)
    if cfg.init_from:
        ckpt = torch.load(cfg.init_from, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        if is_main:
            print(f"[init] loaded weights from {cfg.init_from} "
                  f"(checkpoint step {ckpt.get('step', '?')})", flush=True)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.eval()
    model = DDP(raw_model, device_ids=[local_rank]) if ddp else raw_model
    opt = torch.optim.AdamW(raw_model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

    if is_main:
        n_p = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"device={device}  world_size={world_size}  "
              f"generator params={n_p:,}  effective drift batch={world_size * cfg.gen_per_step}")

    def autocast():
        if cfg.use_bf16 and device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def ngram_at(step):
        """n-gram curriculum: n grows by 1 at each step threshold, capped at max."""
        n = cfg.ngram_min + sum(1 for s in cfg.ngram_grow_steps if step >= s)
        return min(n, cfg.ngram_max)

    # ── one-time init diagnostic (rank 0): first-token homogenisation + GPT-2's BOS support ──
    if is_main:
        with torch.no_grad():
            zc = raw_model.sample_z(16, cfg.noise_dim, cfg.temp, device)
            e0 = raw_model(zc).float()[:, 0, :]                       # (16, D) first-token embedding
            if cfg.sphere_norm:
                e0 = e0 / e0.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            wte = embedder.wte.float()                               # (V, D)
            d = torch.sqrt(((e0 * e0).sum(-1, keepdim=True)
                            + (wte * wte).sum(-1).unsqueeze(0)
                            - 2.0 * e0 @ wte.t()).clamp_min(0.0))     # (16, V) L2 dist
            if cfg.mask_illegal_tokens:
                d = d.masked_fill(embedder.illegal_mask.unsqueeze(0), float("inf"))
            topd, topi = d.topk(3, dim=-1, largest=False)
            print("\n[init-diag] first token of 16 random inputs -> nearest-3 wte tokens (dist):",
                  flush=True)
            for g in range(16):
                trips = ", ".join(f"{tokenizer.decode([int(topi[g, k])])!r}@{float(topd[g, k]):.2f}"
                                  for k in range(3))
                print(f"  z#{g:2d}: {trips}", flush=True)
            uniq = torch.unique(topi[:, 0]).numel()
            print(f"  -> distinct nearest-1 token across the 16 inputs: {uniq}/16", flush=True)

        if cfg.positive_source == "gpt2":
            from .gpt2_teacher import _get_teacher
            gpt2 = _get_teacher(cfg.gpt2_teacher, str(device))
            with torch.no_grad():
                inp0 = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
                probs0 = torch.softmax(gpt2(input_ids=inp0).logits[0, -1], dim=-1)   # (V,)
                if cfg.mask_illegal_tokens:
                    probs0 = probs0.masked_fill(embedder.illegal_mask, 0.0)
                keep = (probs0 > cfg.gpt2_prob_thresh).nonzero().flatten()
                pv = probs0[keep]
                order = pv.argsort(descending=True)
                keep, pv = keep[order], pv[order]
                print(f"[init-diag] GPT-2 first token after [BOS] with prob>{cfg.gpt2_prob_thresh}: "
                      f"{keep.numel()} tokens", flush=True)
                shown = ", ".join(f"{tokenizer.decode([int(keep[k])])!r}:{float(pv[k]):.3f}"
                                  for k in range(min(keep.numel(), 30)))
                print(f"  {shown}\n", flush=True)

    model.train()
    ema = None
    t0 = time.time()
    cur_n = bank.n
    gpt2_on = False
    # Repel ring. With repel_reuse the field is exactly one optimizer step of generations,
    # so it costs no forward at all and trails the live generator by at most one step.
    repel_q = cfg.repel_queue
    if repel_q <= 0 and cfg.repel_reuse:
        repel_q = cfg.grad_accum * cfg.gen_per_step
    repel_buf, repel_ptr, repel_fill = None, 0, 0
    for step in range(1, cfg.steps + 1):
        n = ngram_at(step)
        if n != cur_n:                                          # curriculum bumped n
            bank.set_ngram(n)
            cur_n = n
            if is_main:
                print(f"[step {step:>6d}] >>> n-gram curriculum: now {n}-gram", flush=True)
        opt.zero_grad(set_to_none=True)
        # attraction-temperature warm-up (default OFF): geometrically anneal tau from a
        # softer attr_temp_warmup_start down to the steady attract_temp over
        # attr_temp_warmup_steps, so far off-policy positives can still pull clustered
        # siblings apart out of the init-collapse deadlock before the sharp steady tau
        # takes over. warmup_steps=0 -> eff_attract_temp == attract_temp, so teacher and
        # every existing run are byte-identical.
        _wsteps = int(getattr(cfg, "attr_temp_warmup_steps", 0))
        if _wsteps > 0 and step < _wsteps:
            _frac = step / _wsteps
            eff_attract_temp = (float(getattr(cfg, "attr_temp_warmup_start", 1.0)) ** (1.0 - _frac)
                                * cfg.attract_temp ** _frac)
        else:
            eff_attract_temp = cfg.attract_temp
        lv_acc = 0.0
        for _micro in range(cfg.grad_accum):
            z = raw_model.sample_z(cfg.gen_per_step, cfg.noise_dim, cfg.temp, device)
            with autocast():
                emb = model(z)                                      # (G, T, D) continuous
                if cfg.sphere_norm:                                 # project generator output to unit sphere
                    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                else:                                               # euclidean: put output on the wte-norm shell
                    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8) * wte_shell
                gen_ng = embedder.to_ngrams(emb.float(), n)         # (G, T-n+1, n*D), grad

            use_gpt2 = (cfg.positive_source == "gpt2" and step >= cfg.gpt2_start_step)
            if use_gpt2 and not gpt2_on:
                gpt2_on = True
                if is_main:
                    print(f"[step {step:>6d}] >>> switching to GPT-2 teacher positives", flush=True)
                    if cfg.gpt2_start_step > 0:   # save the 3-gram-trained weights for reuse
                        ck = os.path.join(cfg.checkpoint_dir, f"pre_gpt2_step{step - 1}.pt")
                        torch.save({"step": step - 1, "model": raw_model.state_dict(),
                                    "ema": ema_model.state_dict(), "cfg": cfg}, ck)
                        print(f"[step {step:>6d}] saved 3-gram checkpoint -> {ck}", flush=True)
            if use_gpt2:
                sel_alpha = 0.0
                pos_match_pos = pos_own = None
                with torch.no_grad():
                    gen_tokens = embedder.decode(emb.detach())
                # (b) diagnostic: build the teacher support from a REAL prefix instead of the model's
                # own (garbage) prefix, to test whether on-policy bootstrapping is what blocks fluency
                # at long length. Everything else (sphere, geodesic, repel) is unchanged; only the
                # tokens fed to the teacher change. Off => original on-policy behaviour.
                if getattr(cfg, "teacher_real_prefix", False):
                    _ridx = torch.randint(0, bank.tokens.shape[0], (emb.shape[0],),
                                          device=bank.tokens.device)
                    teacher_tokens = bank.tokens[_ridx].to(device).long()   # (G, T) REAL prefixes
                else:
                    teacher_tokens = gen_tokens
                if cfg.gpt2_method == "repair":
                    # Repel field source. reuse: the ring is fed from the gradient
                    # batch itself (below, after each backward) -- free, and at most one
                    # optimizer step stale. Otherwise draw dedicated extras.
                    if cfg.repel_reuse:
                        repel_extra = repel_buf[:repel_fill] if repel_fill > 0 else None
                    else:
                        repel_extra = None
                    # --- repulsion field: extra no-grad generations ---------------------------
                    # These join the gen-gen repulsion ONLY: no attraction, no loss, no gradient.
                    # The repel block reads `old` (the raw embedding) and nothing else, so they
                    # skip embedder.decode AND the GPT-2 teacher pass -- one generator forward
                    # each, no teacher time. raw_model, not model: a DDP forward would arm a
                    # backward that never arrives.
                    if cfg.repel_extra > 0 and not cfg.repel_reuse:
                        with torch.no_grad():
                            # Chunked: a no_grad forward frees layer by layer but its per-layer
                            # transient still scales with batch, so drawing all K at once peaks at
                            # K/gen_per_step times the forward this card was sized for. Only the
                            # (K, T, D) result is kept -- write it straight into a preallocated
                            # buffer so no concatenation doubles it either.
                            mb = cfg.repel_chunk if cfg.repel_chunk > 0 else cfg.gen_per_step
                            e_ex = torch.empty(cfg.repel_extra, *emb.shape[1:],
                                               dtype=torch.float32, device=device)
                            for i in range(0, cfg.repel_extra, mb):
                                b = min(mb, cfg.repel_extra - i)
                                z_ex = raw_model.sample_z(b, cfg.noise_dim, cfg.temp, device)
                                with autocast():
                                    e = raw_model(z_ex)
                                    if cfg.sphere_norm:      # same projection as the real batch
                                        e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                                e_ex[i:i + b] = e.float()    # `old` is fp32 -- match it
                                del e
                            if repel_q > 0:                        # MoCo ring: refresh K of Q per step
                                if repel_buf is None:
                                    repel_buf = e_ex.new_zeros(repel_q, *e_ex.shape[1:])
                                k = min(e_ex.shape[0], repel_q)
                                idx = (torch.arange(k, device=device) + repel_ptr) % repel_q
                                repel_buf[idx] = e_ex[:k]
                                repel_ptr = (repel_ptr + k) % repel_q
                                repel_fill = min(repel_fill + k, repel_q)
                                repel_extra = repel_buf[:repel_fill]
                            else:
                                repel_extra = e_ex
                    # per-position repair: flag every implausible token, resample plausible
                    # tokens there, attract that position by TOKEN distance only
                    with torch.no_grad():
                        repair_tokens, repair_valid, implausible = gpt2_build_repairs(
                            emb.detach(), teacher_tokens, cfg.gpt2_n_pos, cfg.gpt2_prob_thresh,
                            bos_id, device, model_name=cfg.gpt2_teacher,
                            illegal_mask=embedder.illegal_mask if cfg.mask_illegal_tokens else None,
                            target_mode=cfg.gpt2_repair_target,
                            support=cfg.gpt2_support, nucleus_p=cfg.gpt2_nucleus_p,
                            no_repeat_ngram=cfg.gpt2_no_repeat_ngram,
                            no_repeat_window=cfg.gpt2_no_repeat_window)
                        repair_wte = embedder.wte[repair_tokens].float()    # (G, T, n_pos, D)
                        own_wte = embedder.wte[teacher_tokens].float()      # (G, T, D) own token (snap)
                    gen_emb = emb.float()                                   # (G, T, D) raw output (grad)
                    loss, info = gpt2_repair_loss(gen_emb, repair_wte, repair_valid, own_wte,
                                                  implausible, temp=cfg.gpt2_temp, repel=cfg.gpt2_repel,
                                                  target_mode=cfg.gpt2_repair_target,
                                                  repel_abs=cfg.gpt2_repel_abs, abs_scale=manifold_scale,
                                                  repel_perpos=cfg.gpt2_repel_perpos,
                                                  repel_intra=cfg.gpt2_repel_intra,
                                                  free_prefix=cfg.gpt2_free_prefix,
                                                  sphere=cfg.sphere_norm, sphere_step=cfg.sphere_step,
                                                  sphere_geo_max=cfg.sphere_geo_max,
                                                  temp_list=cfg.gpt2_temp_list,
                                                  repel_extra=repel_extra)
                    pos_idx = emb.new_zeros(int(implausible.sum().item()))  # logging: #repair positions
                else:
                    # continuation: prefix + top-k sample at p* + GPT-2 continuation; whole-
                    # sequence affinity, force masked to [0..p*]
                    with torch.no_grad():
                        pos_tokens, p_star = gpt2_build_positives(
                            gen_tokens, cfg.gpt2_n_pos, cfg.gpt2_topk, bos_id, device,
                            model_name=cfg.gpt2_teacher)
                        pos_ng = embedder.ngrams_from_ids(pos_tokens, n)    # (G*n_pos, M, n*D)
                    M = gen_ng.shape[1]
                    force_pos_mask = (torch.arange(M, device=device).unsqueeze(0)
                                      <= p_star.unsqueeze(1))               # (G, M) force on [0..p*]
                    loss, info = drift_loss_2gram(gen_ng, pos_ng, None, R_list=cfg.R_list,
                                                  attract_temp=eff_attract_temp,
                                                  sinkhorn_iters=cfg.sinkhorn_iters,
                                                  per_position=False,
                                                  force_pos_mask=force_pos_mask)
                    pos_idx = pos_tokens[:, 0]                              # logging: #positives
            elif cfg.loss_mode == "match":
                # greedy one-to-one matching against random (unbiased) real candidates
                n_real = cfg.match_n_real if cfg.match_n_real > 0 else 2 * cfg.gen_per_step
                with torch.no_grad():
                    real_idx = bank.sample_negatives(n_real)        # random reals
                real_ng = bank.gather_ng(real_idx)
                pos_idx = real_idx
                sel_alpha = 0.0
                loss, info = match_loss_2gram(gen_ng, real_ng)
            else:
                sel_alpha = min(1.0, step / max(1, cfg.select_anneal_steps))
                use_token = (cfg.select_mode == "token_2gram_match"
                             and step >= cfg.token_match_start)   # repr bootstrap -> token refine
                pos_match_pos = None
                pos_own = None
                with torch.no_grad():
                    if use_token:
                        gen_tokens = embedder.decode(emb.detach())
                        if cfg.per_position_force:
                            pos_idx, pos_match_pos, pos_own = bank.select_positives_token(
                                gen_tokens, gen_ng.detach(), cfg.n_pos,
                                per_position=cfg.rank_at_match_pos, return_pos=True)
                        else:
                            pos_idx = bank.select_positives_token(gen_tokens, gen_ng.detach(), cfg.n_pos)
                    elif cfg.per_position_force:
                        pos_idx, pos_match_pos, pos_own = bank.select_positives(
                            gen_ng.detach(), cfg.n_pos, alpha=sel_alpha, return_pos=True)
                    else:
                        pos_idx = bank.select_positives(gen_ng.detach(), cfg.n_pos, alpha=sel_alpha)
                pos_ng = bank.gather_ng(pos_idx)
                if is_main and step % cfg.log_every == 0:            # SCALE DIAGNOSTIC (temp)
                    with torch.no_grad():
                        _gn = gen_ng.detach().float(); _pn = pos_ng.float()
                        _gg = torch.cdist(_gn[:8, 0, :], _gn[:64, 0, :]).mean().item()
                        _gp = (torch.cdist(_gn[:8, 0, :], _pn[:64, 0, :]).mean().item()
                               if _pn.shape[0] else float("nan"))
                        print(f"  [diag] tau={eff_attract_temp:.3f} "
                              f"|emb|={emb.detach().float().norm(dim=-1).mean().item():.3f} "
                              f"|gen_ng|={_gn.norm(dim=-1).mean().item():.3f} "
                              f"|pos_ng|={_pn.norm(dim=-1).mean().item() if _pn.shape[0] else float('nan'):.3f} "
                              f"d(gen,gen)={_gg:.3f} d(gen,pos)={_gp:.3f}", flush=True)
                neg_ng = None
                if cfg.n_neg > 0:
                    with torch.no_grad():
                        neg_idx = bank.sample_negatives(cfg.n_neg)
                    neg_ng = bank.gather_ng(neg_idx)
                # --drift-cross-2gram: route the per-position force by the CROSS-position
                # min-2-gram (force only at the single argmin 2-gram position), instead of
                # the same-position (i=j) alignment. The min-2-gram always finds a match, so
                # the force is non-zero even for long sequences where same-position exact
                # alignment is essentially never hit.
                _pmp = None if getattr(cfg, "drift_cross_2gram", False) else pos_match_pos
                loss, info = drift_loss_2gram(gen_ng, pos_ng, neg_ng, R_list=cfg.R_list,
                                              attract_temp=eff_attract_temp,
                                              sinkhorn_iters=cfg.sinkhorn_iters,
                                              per_position=cfg.per_position_force,
                                              pos_match_pos=_pmp,
                                              pos_own=(pos_own if cfg.attract_own_only else None),
                                              repel_intra=cfg.gpt2_repel_intra)

            # Scale so grad_accum micro-steps equal one batch of grad_accum*gen_per_step,
            # and skip DDP's all-reduce until the last one -- without no_sync every micro-step
            # would pay a full gradient sync and accumulation would cost N syncs instead of 1.
            loss = loss / cfg.grad_accum
            if ddp and _micro < cfg.grad_accum - 1:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()
            lv_acc += float(loss.detach()) * cfg.grad_accum

            if repel_q > 0 and cfg.repel_reuse:
                # Feed the repel ring from the gradient batch itself: these generations were
                # produced for the backward anyway, so the field costs no extra forward. Push
                # AFTER the loss -- pushing first would let a micro-batch repel from itself.
                with torch.no_grad():
                    e_r = emb.detach().float()
                    kr = min(e_r.shape[0], repel_q)
                    if repel_buf is None:
                        repel_buf = e_r.new_zeros(repel_q, *e_r.shape[1:])
                    idx = (torch.arange(kr, device=device) + repel_ptr) % repel_q
                    repel_buf[idx] = e_r[:kr]
                    repel_ptr = (repel_ptr + kr) % repel_q
                    repel_fill = min(repel_fill + kr, repel_q)

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        _ema_update(ema_model, raw_model, cfg.ema_decay)

        lv = lv_acc / cfg.grad_accum
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv

        if is_main and step % cfg.log_every == 0:
            rep_s = f"rep_n={int(info['repel_n'])} " if "repel_n" in info else ""
            print(f"[step {step:>6d}] n={n}-gram loss={lv:.4f} ema={ema:.4f} "
                  f"n_pos={pos_idx.numel()} sel_alpha={sel_alpha:.2f} "
                  f"scale={info['scale']:.3f} {rep_s}"
                  f"lr={sched.get_last_lr()[0]:.2e} t={time.time()-t0:.0f}s", flush=True)
            # which real is each generation pulled toward? (collapse check; needs the bank)
            # a full-bank cdist -- skip it for large banks (it would need 100GB+ and is
            # only a diagnostic print; training's own chunked select is unaffected).
            if not cfg.skip_bank and bank.N <= int(getattr(cfg, "nn_report_max_n", 20000)):
                nn_distinct, nn_top1, nn_topk, nn_lines = nearest_real_report(
                    gen_ng.detach(), bank, tokenizer)
                print(f"  [batch nn] {nn_distinct}/{gen_ng.shape[0]} distinct reals  "
                      f"top1={nn_top1:.1%} top10={nn_topk:.1%}", flush=True)
                for ln in nn_lines:
                    print(ln, flush=True)
            if cfg.log_gen_samples:
                gen_tokens = embedder.decode(emb.detach())
                if use_gpt2 and cfg.gpt2_method == "repair":
                    # show each gen's first implausible token -> its resampled plausible tokens
                    for g in range(min(3, gen_ng.shape[0])):
                        gtxt = tokenizer.decode(gen_tokens[g].tolist(), skip_special_tokens=True)
                        bad = implausible[g].nonzero().flatten()
                        if bad.numel() > 0:
                            bp = int(bad[0])
                            orig = tokenizer.decode([int(gen_tokens[g, bp])])
                            reps = [tokenizer.decode([int(t)])
                                    for t, v in zip(repair_tokens[g, bp].tolist(),
                                                    repair_valid[g, bp].tolist()) if v]
                            print(f"  [gpt2 g#{g}] {gtxt!r}  bad@{bp} {orig!r} -> {reps}", flush=True)
                        else:
                            print(f"  [gpt2 g#{g}] {gtxt!r}  (all plausible)", flush=True)
                    print(f"  [gpt2 repair] avg #implausible/seq={info['n_repair']:.1f}", flush=True)
                elif use_gpt2:
                    # continuation: positives share the gen's plausible prefix [0..p*-1]
                    for g in range(min(3, gen_ng.shape[0])):
                        gtxt = tokenizer.decode(gen_tokens[g].tolist(), skip_special_tokens=True)
                        ptxt = tokenizer.decode(pos_tokens[g * cfg.gpt2_n_pos].tolist(),
                                                skip_special_tokens=True)
                        print(f"  [gpt2 p*={int(p_star[g])}] gen#{g} {gtxt!r}  -> pos {ptxt!r}",
                              flush=True)
                    print(f"  [gpt2 teacher] p*_mean={float(p_star.float().mean()):.1f}  "
                          f"#positives={pos_tokens.shape[0]}", flush=True)
                elif bank.N <= int(getattr(cfg, "nn_report_max_n", 20000)):
                    # per-generation matched positives, consistent with the ACTIVE selection
                    # (a full-bank op; skipped for large banks, same as the nn report above)
                    active_mode = ("token_2gram_match"
                                   if (cfg.select_mode == "token_2gram_match"
                                       and step >= cfg.token_match_start)
                                   else "repr_min2gram")
                    mp_n, mp_lines = matched_positives_report(
                        gen_ng.detach(), gen_tokens, bank, tokenizer, cfg.n_pos,
                        select_mode=active_mode)
                    hdr = (f"{mp_n}/{min(3, gen_ng.shape[0])} shown gens have an exact match"
                           if active_mode == "token_2gram_match"
                           else f"union={mp_n} reals across batch top-{cfg.n_pos}")
                    print(f"  [matched pos | {active_mode}] {hdr}", flush=True)
                    for ln in mp_lines:
                        print(ln, flush=True)

        if is_main and step % cfg.eval_every == 0:
            metrics, texts = run_eval(
                ema_model, embedder, bank, cfg, device, tokenizer,
                n_samples=cfg.eval_samples, n_show=cfg.eval_n_show,
                compute_ppl=cfg.eval_compute_ppl)
            ppl_str = (f"gen_ppl={metrics['gen_ppl']:.2f} entropy={metrics['entropy']:.3f} "
                       if cfg.eval_compute_ppl else "")
            sb_str = f"self_bleu={metrics['self_bleu']:.3f} " if "self_bleu" in metrics else ""
            print(f"  [eval {step}] {ppl_str}{sb_str}"
                  f"time/sample={metrics['gen_time_per_sample']*1000:.2f}ms "
                  f"(gen {metrics['gen_time_total']:.1f}s/{cfg.eval_samples}) "
                  f"distinct_seqs={metrics['distinct_seqs']}/{cfg.eval_samples} "
                  f"distinct_reals_hit={metrics['distinct_reals_hit']}/{cfg.eval_samples} "
                  f"exact_match={metrics['exact_match']}/{cfg.eval_samples}", flush=True)
            print(f"  [eval {step}] {cfg.eval_n_show} samples:", flush=True)
            for i, t in enumerate(texts):
                print(f"    [{i+1}] {t!r}")
            if device.type == "cuda":            # release eval's fragmented/reserved memory
                torch.cuda.empty_cache()

        if is_main and (step % cfg.save_every == 0 or step == cfg.steps):
            torch.save({"step": step, "model": raw_model.state_dict(),
                        "ema": ema_model.state_dict(), "cfg": cfg},
                       os.path.join(cfg.checkpoint_dir, f"step_{step}.pt"))

    if is_main:
        print("Training complete.")
    _cleanup()
    return raw_model


def main():
    p = argparse.ArgumentParser(description="Text drifting sanity check (min-2-gram drift)")
    p.add_argument("--dataset", default="openwebtext", dest="dataset_name")
    p.add_argument("--data-path", default="", help="local JSONL corpus file or directory")
    p.add_argument("--tokenizer-name", default="gpt2")
    p.add_argument("--embed-model", default="gpt2")
    p.add_argument("--seq-len", type=int, default=8, dest="seq_len")
    p.add_argument("--n-train", type=int, default=1000, dest="n_train")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--gen-per-step", type=int, default=64, dest="gen_per_step")
    p.add_argument("--repel-extra", type=int, default=0, dest="repel_extra",
                   help="no-grad generations drawn per step for the repulsion field "
                        "only (no attraction, no loss, no gradient); 0 = off")
    p.add_argument("--repel-queue", type=int, default=0, dest="repel_queue",
                   help="MoCo-style FIFO: repel against the last N extras while "
                        "refreshing --repel-extra per step; 0 = fresh extras only")
    p.add_argument("--repel-chunk", type=int, default=0, dest="repel_chunk",
                   help="generate the extras this many at a time (peak forward "
                        "memory); 0 = same as --gen-per-step")
    p.add_argument("--grad-accum", type=int, default=1, dest="grad_accum",
                   help="micro-steps per optimizer step; gradient batch becomes "
                        "grad_accum * gen_per_step * world_size")
    p.add_argument("--repel-reuse", action="store_true", dest="repel_reuse",
                   help="repel against the gradient batch itself (free, <=1 step "
                        "stale) instead of drawing dedicated --repel-extra samples")
    p.add_argument("--n-pos", type=int, default=32, dest="n_pos")
    p.add_argument("--n-neg", type=int, default=0, dest="n_neg")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--noise-dim", type=int, default=128, dest="noise_dim")
    p.add_argument("--num-layers", type=int, default=6, dest="num_layers")
    p.add_argument("--ema-decay", type=float, default=0.999, dest="ema_decay")
    p.add_argument("--attract-temp", type=float, default=1.0, dest="attract_temp",
                   help="affinity temperature; <1 sharpens toward one-hot (anti-collapse)")
    p.add_argument("--attr-temp-warmup-start", type=float, default=1.0, dest="attr_temp_warmup_start",
                   help="starting (softer) tau for the attraction-temperature warm-up")
    p.add_argument("--attr-temp-warmup-steps", type=int, default=0, dest="attr_temp_warmup_steps",
                   help="anneal tau from --attr-temp-warmup-start to --attract-temp over this many "
                        "steps (0 = off, tau constant; breaks off-policy init-collapse deadlock)")
    p.add_argument("--sinkhorn-iters", type=int, default=0, dest="sinkhorn_iters",
                   help=">0 enables balanced Sinkhorn affinity (caps per-real intake; ~5-20 iters)")
    p.add_argument("--per-position-force", action="store_true", dest="per_position_force",
                   help="per-position affinity+force (each position pulled only by reals matching there)")
    p.add_argument("--drift-cross-2gram", action="store_true", dest="drift_cross_2gram",
                   help="per-position force routed by the CROSS-position min-2-gram (force at the "
                        "single argmin 2-gram position), not same-position (i=j) alignment")
    p.add_argument("--rank-at-match-pos", action="store_true", dest="rank_at_match_pos",
                   help="Q3: rank token-match candidates by match-position distance (default: whole-sequence). "
                        "WARNING: at small n this can cause function-word collapse")
    p.add_argument("--attract-own-only", action="store_true", dest="attract_own_only",
                   help="Q2: per-position attraction only toward a gen's OWN picks (default: shared union cloud)")
    p.add_argument("--positive-source", default="dataset", choices=["dataset", "gpt2"],
                   dest="positive_source",
                   help="dataset = match the data bank; gpt2 = GPT-2 teacher builds positives")
    p.add_argument("--teacher-real-prefix", action="store_true", dest="teacher_real_prefix",
                   help="diagnostic: build the GPT-2 teacher support from REAL data prefixes "
                        "(off-policy) instead of the model's own generated prefix")
    p.add_argument("--gpt2-method", default="repair", choices=["continuation", "repair"],
                   dest="gpt2_method",
                   help="gpt2 positives: 'continuation' (prefix+continuation) or 'repair' (per-position)")
    p.add_argument("--gpt2-repair-target", default="self", choices=["self", "teacher"],
                   dest="gpt2_repair_target",
                   help="repair target: 'self' (repair implausible, snap plausible) or "
                        "'teacher' (A2/reverse-KL: every pos -> nearest GPT-2-support token)")
    p.add_argument("--gpt2-repel-abs", action="store_true", dest="gpt2_repel_abs",
                   help="repair: scale gen-gen repulsion to an absolute (candidate-distance) "
                        "unit instead of the attraction RMS (B; anti-collapse at degenerate pt)")
    p.add_argument("--gpt2-repel-perpos", action="store_true", dest="gpt2_repel_perpos",
                   help="repair: per-position gen-gen repulsion (diversify every position incl. "
                        "the prefix) instead of whole-sequence (tail-only diversity)")
    p.add_argument("--gpt2-repel-intra", type=float, default=0.0, dest="gpt2_repel_intra",
                   help="repair: intra-sequence repulsion weight (push a gen's own positions "
                        "apart; fights within-sequence token repetition). 0 = off")
    p.add_argument("--gpt2-free-prefix", type=int, default=0, dest="gpt2_free_prefix",
                   help="repair: leave the first N positions force-free (no attraction/repulsion/"
                        "gradient), keeping the generator's own first token(s). 1 = free position 0")
    p.add_argument("--sphere-norm", action="store_true", dest="sphere_norm",
                   help="put wte rows + generator output on the unit sphere (angular/cosine "
                        "geometry); repulsion becomes purely directional, no norm inflation")
    p.add_argument("--sphere-step", default="proj", choices=["proj", "retract", "geodesic"],
                   dest="sphere_step",
                   help="sphere goal step: proj (chord+project) | retract (tangent+project) | "
                        "geodesic (exp-map great circle)")
    p.add_argument("--sphere-geo-max", type=float, default=1.5708, dest="sphere_geo_max",
                   help="geodesic step: max per-step arc length in radians (clamp)")
    p.add_argument("--gpt2-teacher", default="gpt2", dest="gpt2_teacher",
                   help="HF model name for the GPT-2 teacher (positive-source=gpt2)")
    p.add_argument("--gpt2-topk", type=int, default=50, dest="gpt2_topk",
                   help="continuation method: top-k for plausibility + sampling")
    p.add_argument("--gpt2-n-pos", type=int, default=16, dest="gpt2_n_pos",
                   help="positives / plausible tokens sampled per position (positive-source=gpt2)")
    p.add_argument("--gpt2-prob-thresh", type=float, default=0.01, dest="gpt2_prob_thresh",
                   help="repair method: a gen token is implausible if its GPT-2 prob <= this")
    p.add_argument("--gpt2-support", default="thresh", choices=["thresh", "nucleus"],
                   dest="gpt2_support",
                   help="teacher support set: thresh (prob>prob-thresh) | nucleus (cumulative top-p)")
    p.add_argument("--gpt2-nucleus-p", type=float, default=0.99, dest="gpt2_nucleus_p",
                   help="nucleus support: cumulative-probability cutoff")
    p.add_argument("--gpt2-no-repeat-ngram", type=int, default=0, dest="gpt2_no_repeat_ngram",
                   help="k>=2: drop from the teacher support set any token completing a k-gram "
                        "already in the gen's own prefix (breaks the repetition loop); 0=off")
    p.add_argument("--gpt2-no-repeat-window", type=int, default=0, dest="gpt2_no_repeat_window",
                   help="no-repeat-ngram: only look back this many tokens (0 = whole prefix)")
    p.add_argument("--gpt2-temp-list", default="", dest="gpt2_temp_list",
                   type=lambda s: tuple(float(x) for x in s.split(",") if x.strip()),
                   help="attraction multi-temp fusion, comma-separated (e.g. 0.1,0.3,1.0); "
                        "empty = single --gpt2-temp")
    p.add_argument("--gpt2-temp", type=float, default=1.0, dest="gpt2_temp",
                   help="repair method: token-distance affinity temperature")
    p.add_argument("--gpt2-repel", type=float, default=1.0, dest="gpt2_repel",
                   help="repair method: weight of whole-sequence gen-gen repulsion (0=off)")
    p.add_argument("--gpt2-start-step", type=int, default=0, dest="gpt2_start_step",
                   help="positive-source=gpt2: dataset matching before this step, GPT-2 teacher after")
    p.add_argument("--init-from", default="", dest="init_from",
                   help="load generator+EMA weights from this checkpoint at start (skip 2/3-gram bootstrap)")
    p.add_argument("--loss-mode", default="drift", choices=["drift", "match"],
                   dest="loss_mode", help="drift=drift_loss_2gram; match=greedy one-to-one")
    p.add_argument("--match-n-real", type=int, default=0, dest="match_n_real",
                   help="match mode: random real candidates per step (0 -> 2*gen_per_step)")
    p.add_argument("--select-anneal-steps", type=int, default=10_000,
                   dest="select_anneal_steps",
                   help="steps to anneal positive-selection alpha 0(min-2gram)->1(L2)")
    p.add_argument("--select-mode", default="repr_min2gram",
                   choices=["repr_min2gram", "token_2gram_match"], dest="select_mode",
                   help="positive selection: repr-space min-2gram, or exact same-position token 2-gram match")
    p.add_argument("--token-match-start", type=int, default=0, dest="token_match_start",
                   help="with select-mode token_2gram_match: repr-min2gram before this step, token-match after")
    p.add_argument("--ngram-min", type=int, default=2, dest="ngram_min",
                   help="n-gram curriculum: starting window size n")
    p.add_argument("--ngram-max", type=int, default=4, dest="ngram_max",
                   help="n-gram curriculum: max window size n (cap)")
    p.add_argument("--ngram-grow-steps", default="", dest="ngram_grow_steps",
                   type=lambda s: tuple(int(x) for x in s.split(",") if x.strip()),
                   help="comma list of step thresholds; n increments by 1 at each (e.g. 40000,70000)")
    p.add_argument("--bf16", action="store_true", dest="use_bf16")
    p.add_argument("--ckpt-dir", default="runs/unconditional", dest="checkpoint_dir")
    p.add_argument("--eval-every", type=int, default=1_000, dest="eval_every")
    p.add_argument("--eval-samples", type=int, default=1000, dest="eval_samples")
    p.add_argument("--eval-show", type=int, default=10, dest="eval_n_show")
    p.add_argument("--no-ppl", action="store_false", dest="eval_compute_ppl")
    p.add_argument("--skip-bank", action="store_true", dest="skip_bank",
                   help="do not build the real n-gram bank (gpt2 mode). Skips distinct_reals_hit "
                        "and the [batch nn] diagnostic; avoids the ~100GB bank at long seq_len")
    p.add_argument("--log-gen-samples", action="store_true", dest="log_gen_samples",
                   help="print the per-generation [gpt2 g#N] repair/positive diagnostics at log_every")
    p.add_argument("--no-mask-illegal", action="store_false", dest="mask_illegal_tokens",
                   help="allow decode/repair to pick control/byte-fragment tokens "
                        "(default: forbidden -> nearest clean token instead)")
    args = p.parse_args()

    cfg = TextDriftConfig(**vars(args))
    cfg.output_dir = os.path.join(cfg.checkpoint_dir, "output")
    train(cfg)


if __name__ == "__main__":
    main()
