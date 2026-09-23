"""Evaluate saved seq2seq checkpoints, including gold-embedding diagnostics."""

import argparse
import csv
import glob
import os
import re
from pathlib import Path

import numpy as np
import torch

from .cond_generator import CondDriftGenerator
from .data import build_dataset
from .evaluate import run_eval
from .qwen_features import QwenEmbedder, load_qwen


def _checkpoint_paths(args):
    paths = list(args.ckpt or [])
    if args.ckpt_dir:
        paths.extend(glob.glob(os.path.join(args.ckpt_dir, args.pattern)))
    paths = sorted(set(paths), key=lambda path: (
        int(re.search(r"step_(\d+)", os.path.basename(path)).group(1))
        if re.search(r"step_(\d+)", os.path.basename(path)) else -1,
        path))
    if not paths:
        raise FileNotFoundError("no checkpoints found; pass --ckpt or --ckpt-dir")
    return paths


def _fixed_indices(dataset, n, seed):
    count = len(dataset) if n <= 0 else min(n, len(dataset))
    if count == len(dataset):
        return torch.arange(count, dtype=torch.long)
    rng = np.random.default_rng(seed)
    return torch.from_numpy(
        rng.choice(len(dataset), size=count, replace=False).astype(np.int64))


@torch.inference_mode()
def _gold_embedding_error(gen, embedder, dataset, indices, z_all, cfg, device,
                          batch_size):
    squared_sum = 0.0
    cosine_sum = 0.0
    position_count = 0
    hidden = int(embedder.H)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        query_ids, query_mask, target_ids, target_mask = dataset.rows(batch_indices)
        query_ids = query_ids.to(device)
        query_mask = query_mask.to(device)
        target_ids = target_ids.to(device)
        target_mask = target_mask.to(device)
        query_emb = embedder.query_embeds(query_ids)
        amp = bool(cfg.use_bf16 and device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            pred = gen(query_emb, query_mask,
                       z_all[start:start + len(batch_indices)])
        pred = pred.float()
        if cfg.sphere_norm:
            pred = torch.nn.functional.normalize(pred, dim=-1)
        gold = embedder.lookup(target_ids).float()

        mask = target_mask.to(device=device, dtype=pred.dtype)
        diff2 = (pred - gold).pow(2).sum(dim=-1)
        cosine = torch.nn.functional.cosine_similarity(pred, gold, dim=-1, eps=1e-8)
        squared_sum += float((diff2 * mask).sum())
        cosine_sum += float((cosine * mask).sum())
        position_count += int(mask.sum())

    denom = max(position_count, 1)
    l2 = (squared_sum / denom) ** 0.5
    rmse = (squared_sum / (denom * hidden)) ** 0.5
    return {"gold_rmse": rmse, "gold_l2": l2,
            "gold_cos": cosine_sum / denom,
            "gold_positions": position_count}


def _run_gold_diagnostic(paths, first_state, args, cfg, embedder, tokenizer,
                          device):
    dataset = build_dataset(cfg, tokenizer, args.split)
    diag_n = args.n if args.n > 0 else 2048
    indices = _fixed_indices(dataset, diag_n, args.seed)
    cpu_rng = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    z_all = torch.randn(
        len(indices), cfg.noise_dim, generator=cpu_rng, dtype=torch.float32) * cfg.temp
    z_all = z_all.to(device)

    gen = CondDriftGenerator(cfg, cfg.embed_dim).to(device)
    gen.eval()
    which_list = ["model", "ema"] if args.which == "both" else [args.which]
    rows = []
    print(f"[gold-diagnostic] split={args.split} n={len(indices):,} "
          f"batch={args.eval_batch_size} checkpoints={len(paths)} "
          f"which={','.join(which_list)} seed={args.seed}")

    for path in paths:
        state = first_state if path == paths[0] else torch.load(
            path, map_location=device, weights_only=False)
        step = int(state.get("step", -1))
        match = re.search(r"step_(\d+)", os.path.basename(path))
        if step < 0 and match:
            step = int(match.group(1))
        for which in which_list:
            if which not in state:
                raise KeyError(f"{which} weights missing from {path}")
            gen.load_state_dict(state[which])
            metrics = _gold_embedding_error(
                gen, embedder, dataset, indices, z_all, cfg, device,
                args.eval_batch_size)
            row = {"step": step, "which": which, "checkpoint": path, **metrics}
            rows.append(row)
            print(f"[step {step:>7} {which:>5}] "
                  f"gold_rmse={metrics['gold_rmse']:.6e} "
                  f"gold_l2={metrics['gold_l2']:.6e} "
                  f"gold_cos={metrics['gold_cos']:.6f} "
                  f"n={metrics['gold_positions']}")
        if path != paths[0]:
            del state

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[gold-diagnostic] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", action="append", default=[],
                        help="checkpoint path; may be repeated")
    parser.add_argument("--ckpt-dir", default="",
                        help="directory containing step_*.pt checkpoints")
    parser.add_argument("--pattern", default="step_*.pt")
    parser.add_argument("--which", choices=["ema", "model", "both"], default="ema")
    parser.add_argument("--gold-diagnostic", action="store_true",
                        help="measure output-vs-gold embeddings instead of BLEU")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--teacher", default="", help="relocated copy of the original task teacher")
    parser.add_argument("--n", type=int, default=0,
                        help="BLEU: <=0 means all; gold diagnostics: default 2048")
    parser.add_argument("--n-show", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="",
                        help="optional CSV output path for gold diagnostics")
    args = parser.parse_args()
    if not args.ckpt and not args.ckpt_dir:
        parser.error("one of --ckpt or --ckpt-dir is required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = _checkpoint_paths(args)
    first_state = torch.load(paths[0], map_location=device, weights_only=False)
    cfg = first_state["cfg"]
    # Back-compat: checkpoints saved before new config fields were added may be
    # missing them when the class is reconstructed at load time.
    for _field, _default in [("gold_decay_steps", 0), ("gold_min_ratio", 1.0),
                              ("no_repeat_window", 0)]:
        if not hasattr(cfg, _field):
            setattr(cfg, _field, _default)
    seed = cfg.seed if args.seed is None else args.seed
    torch.manual_seed(seed)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.teacher:
        cfg.teacher_model = args.teacher
    if args.eval_batch_size:
        cfg.eval_batch_size = args.eval_batch_size

    frozen_teacher, tokenizer = load_qwen(
        cfg.teacher_model, str(device), "bf16" if cfg.use_bf16 else "fp32")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    embedder = QwenEmbedder(cfg, tokenizer, device)
    cfg.embed_dim, cfg.vocab_size = embedder.H, embedder.V
    load_qwen.cache_clear()
    del frozen_teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.gold_diagnostic:
        args.seed = 1234 if args.seed is None else args.seed
        if args.eval_batch_size <= 0:
            args.eval_batch_size = cfg.eval_batch_size
        _run_gold_diagnostic(
            paths, first_state, args, cfg, embedder, tokenizer, device)
        return

    if len(paths) != 1:
        raise ValueError("BLEU mode accepts exactly one checkpoint; use --gold-diagnostic")
    if args.which == "both":
        raise ValueError("--which both is only valid with --gold-diagnostic")
    state = first_state
    gen = CondDriftGenerator(cfg, cfg.embed_dim).to(device)
    gen.load_state_dict(state[args.which])
    gen.eval()

    dataset = build_dataset(cfg, tokenizer, args.split)
    n = len(dataset) if args.n <= 0 else args.n
    metrics, shown = run_eval(
        gen, embedder, dataset, cfg, tokenizer, device, n, args.n_show)
    print("[eval RESULT] " + " ".join(
        f"{key}={value:.3f}" for key, value in metrics.items()))
    for sample in shown:
        print("   " + sample.replace("\n", "\n   "))


if __name__ == "__main__":
    main()
