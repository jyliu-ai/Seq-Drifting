"""Fine-tune a causal GPT-2 teacher on WMT14 De-En or XSum.

The teacher sees [source, target] and is trained with causal cross-entropy only on
target tokens. This objective trains the external teacher; it is not part of the
student's drift loss.
"""

import argparse
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path

import torch
from common.checkpoint import load_checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from common.config import Seq2SeqDriftConfig
from .data import build_dataset


class _Indices(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


def _max_positions(model):
    for name in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(model.config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("the teacher model does not expose a maximum context length")


def _make_collate(dataset, pad_id, context_limit):
    # Reserve the full response width. This matches student training, where teacher
    # support is requested for all response slots even when a gold target is shorter.
    max_source = context_limit - dataset.target_len
    if max_source <= 0:
        raise ValueError(
            f"teacher context {context_limit} must exceed target length {dataset.target_len}")

    def collate(indices):
        source, source_mask, target, target_mask = dataset.rows(torch.tensor(indices))
        sequences, labels = [], []
        for src, smask, tgt, tmask in zip(source, source_mask, target, target_mask):
            src = src[smask][:max_source]
            tgt = tgt[tmask]
            seq = torch.cat([src, tgt])
            label = torch.cat([torch.full_like(src, -100), tgt])
            sequences.append(seq)
            labels.append(label)

        width = max(row.numel() for row in sequences)
        width = min(context_limit, ((width + 7) // 8) * 8)
        input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.bool)
        label_ids = torch.full((len(sequences), width), -100, dtype=torch.long)
        for i, (seq, label) in enumerate(zip(sequences, labels)):
            length = seq.numel()
            input_ids[i, :length] = seq
            attention_mask[i, :length] = True
            label_ids[i, :length] = label
        return input_ids, attention_mask, label_ids

    return collate, max_source


@torch.no_grad()
def _evaluate(model, loader, device, use_bf16):
    model.eval()
    total_nll = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        amp = bool(use_bf16 and device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            loss = model(input_ids=input_ids, attention_mask=attention_mask,
                         labels=labels, use_cache=False).loss
        count = (labels[:, 1:] != -100).sum()
        total_nll += loss.double() * count
        total_tokens += count
    if dist.is_initialized():
        dist.all_reduce(total_nll)
        dist.all_reduce(total_tokens)
    mean_nll = (total_nll / total_tokens.clamp_min(1)).item()
    model.train()
    return mean_nll, math.exp(min(mean_nll, 20.0))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset_name",
                        choices=["wmt14_de_en", "xsum"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--base-model", default="gpt2")
    parser.add_argument("--from-scratch", action="store_true",
                        help="initialize model weights randomly using the base model configuration")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition-len", type=int, default=0)
    parser.add_argument("--target-len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--per-device-batch", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-bf16", dest="use_bf16", action="store_false", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", default="",
                        help="path to a mid-training checkpoint dir saved by --save-every")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="save a resumable checkpoint every N optimizer updates (0=off)")
    return parser.parse_args()


def main():
    args = _parse_args()
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
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("the teacher tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.from_scratch:
        model_config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
        # Initialize GPT-2 attention buffers and positional embeddings at the
        # required size together (XSum uses 1024 + 64 positions).
        need_len = (args.condition_len or (1024 if args.dataset_name == "xsum" else 64)) + args.target_len
        if model_config.model_type == "gpt2":
            model_config.n_positions = max(model_config.n_positions, need_len)
            model_config.n_ctx = model_config.n_positions
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.float32, trust_remote_code=True).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # Extend position embeddings if condition_len + target_len exceeds the model's
    # built-in maximum (GPT-2 has 1024; XSum needs 1024+64=1088). We copy existing
    # rows and interpolate new ones so fine-tuning starts from a reasonable initialisation.
    need_len = (args.condition_len or (1024 if args.dataset_name == "xsum" else 64)) + args.target_len
    cur_max = _max_positions(model)
    if need_len > cur_max:
        if is_main:
            print(f"[teacher] extending position embeddings {cur_max} -> {need_len}")
        # GPT-2 stores position embeddings in model.transformer.wpe (nn.Embedding)
        wpe = model.transformer.wpe
        old_w = wpe.weight.data                          # (cur_max, H)
        H = old_w.shape[1]
        new_w = torch.zeros(need_len, H, dtype=old_w.dtype)
        new_w[:cur_max] = old_w                          # copy existing rows verbatim
        # For new positions, linearly interpolate from the last two rows so gradients
        # flow sensibly from the start of fine-tuning rather than from zeros.
        for i in range(cur_max, need_len):
            t = (i - cur_max + 1) / max(need_len - cur_max, 1)
            new_w[i] = old_w[-2] + t * (old_w[-1] - old_w[-2])
        wpe.weight = torch.nn.Parameter(new_w.to(device))
        model.config.n_positions = need_len
        model.config.n_ctx = need_len

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    cfg = Seq2SeqDriftConfig()
    cfg.dataset_name = args.dataset_name
    cfg.data_dir = args.data_dir
    cfg.query_len = args.condition_len or (1024 if args.dataset_name == "xsum" else 64)
    cfg.resp_len = args.target_len
    cfg.n_train = args.max_train
    cfg.n_eval = args.max_eval
    train_data = build_dataset(cfg, tokenizer, "train")
    eval_data = build_dataset(cfg, tokenizer, "validation")

    collate, teacher_source_len = _make_collate(
        train_data, tokenizer.pad_token_id, _max_positions(model))
    eval_collate, _ = _make_collate(
        eval_data, tokenizer.pad_token_id, _max_positions(model))
    train_indices = _Indices(len(train_data))
    train_sampler = (DistributedSampler(
        train_indices, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        if ddp else None)
    train_loader = DataLoader(
        train_indices, batch_size=args.per_device_batch, sampler=train_sampler,
        shuffle=train_sampler is None, collate_fn=collate, pin_memory=True)
    eval_indices = Subset(_Indices(len(eval_data)), range(rank, len(eval_data), world_size))
    eval_loader = DataLoader(
        eval_indices, batch_size=args.per_device_batch, shuffle=False,
        collate_fn=eval_collate, pin_memory=True)

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_updates = args.max_steps or (updates_per_epoch * args.epochs)
    loop_epochs = (math.ceil(total_updates / updates_per_epoch)
                   if args.max_steps else args.epochs)
    warmup_updates = int(total_updates * args.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_updates, total_updates)
    train_model = DDP(model, device_ids=[local_rank]) if ddp else model

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[teacher] model={args.base_model} dataset={args.dataset_name} "
              f"world={world_size} batch/rank={args.per_device_batch} "
              f"accum={args.grad_accum_steps} updates={total_updates}")
        print(f"[teacher] student condition={cfg.query_len}; teacher condition="
              f"{teacher_source_len}; target={cfg.resp_len}")

    update = 0
    best_val_nll = float("inf")
    resume_epoch = 0

    if args.resume_from:
        resume_dir = Path(args.resume_from)
        opt_path = resume_dir / "optimizer.pt"
        meta_path = resume_dir / "resume_meta.json"
        if is_main:
            print(f"[teacher] resuming from {resume_dir}")
        # Load model weights (saved as HF format)
        raw_model = train_model.module if ddp else train_model
        raw_model.load_state_dict(
            load_checkpoint(resume_dir / "pytorch_model.bin", map_location=device,
                       weights_only=True),
            strict=False)
        if opt_path.exists():
            opt_state = load_checkpoint(opt_path, map_location=device)
            optimizer.load_state_dict(opt_state["optimizer"])
            scheduler.load_state_dict(opt_state["scheduler"])
            update = opt_state["update"]
            best_val_nll = opt_state.get("best_val_nll", float("inf"))
            resume_epoch = opt_state.get("epoch", 0)
            if is_main:
                print(f"[teacher] resumed at update={update} epoch={resume_epoch}")
        if ddp:
            dist.barrier()

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(resume_epoch, loop_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_model.train()
        for micro, (input_ids, attention_mask, labels) in enumerate(train_loader):
            if update >= total_updates:
                break
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            should_step = ((micro + 1) % args.grad_accum_steps == 0
                           or micro + 1 == len(train_loader))
            sync = (train_model.no_sync() if ddp and not should_step else nullcontext())
            with sync:
                amp = bool(args.use_bf16 and device.type == "cuda")
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=amp):
                    loss = train_model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels, use_cache=False).loss
                (loss / args.grad_accum_steps).backward()
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if is_main and update % args.log_every == 0:
                    print(f"[teacher step {update}/{total_updates}] "
                          f"loss={loss.item():.4f} grad={float(grad_norm):.3f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")
                if (is_main and args.save_every > 0
                        and update % args.save_every == 0
                        and update < total_updates):
                    ckpt_dir = output_dir / f"resume_step_{update}"
                    raw_model = train_model.module if ddp else train_model
                    raw_model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    torch.save({
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "update": update,
                        "epoch": epoch,
                        "best_val_nll": best_val_nll,
                    }, ckpt_dir / "optimizer.pt")
                    # Keep only the two most recent resume checkpoints.
                    import shutil
                    old_ckpts = sorted(output_dir.glob("resume_step_*"),
                                       key=lambda p: int(p.name.split("_")[-1]))
                    for old_dir in old_ckpts[:-2]:
                        shutil.rmtree(old_dir, ignore_errors=True)
                    print(f"[teacher] resume checkpoint saved -> {ckpt_dir}")

        # Evaluate the local underlying module. Ranks may own different numbers of
        # validation batches, so DDP forward hooks must not synchronize per batch.
        val_nll, val_ppl = _evaluate(model, eval_loader, device, args.use_bf16)
        if is_main:
            print(f"[teacher epoch {epoch + 1}] val_nll={val_nll:.4f} val_ppl={val_ppl:.2f}")
            epoch_dir = output_dir / f"epoch_{epoch + 1}"
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_dir = output_dir / "best"
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                print(f"[teacher] new best validation NLL -> {best_dir}")
        if ddp:
            dist.barrier()
        if update >= total_updates:
            break

    if is_main:
        final_dir = output_dir / "final"
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        metadata = vars(args) | {
            "student_condition_len": cfg.query_len,
            "teacher_condition_len": teacher_source_len,
            "context_limit": _max_positions(model),
            "updates": update,
            "best_val_nll": best_val_nll,
        }
        (output_dir / "training_config.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"[teacher] saved final checkpoint to {final_dir}")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
