"""Evaluate released math / ProofWriter checkpoints with their saved configuration."""

import argparse
import json
from pathlib import Path

import torch
from common.checkpoint import load_checkpoint

from common.candidates import load_candidate_groups
from models.generators import CondDriftGenerator, QwenCondGenerator
from models.embeddings import FrozenEmbedder as QwenEmbedder, load_teacher as load_qwen
from common.candidates import CandidateDataset, eval_one_pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--teacher", default="", help="original frozen embedding model or relocated copy")
    parser.add_argument("--backbone", default="", help="original generator backbone or relocated copy")
    parser.add_argument("--which", choices=["ema", "model"], default="ema")
    parser.add_argument("--max-candidates", type=int, default=1, help="1 for math; 8 for ProofWriter; 0 keeps all")
    parser.add_argument("--n", type=int, default=0, help="0 evaluates the complete test set")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="", help="optional JSON metrics file")
    args = parser.parse_args()
    if args.repeats < 1 or args.n < 0 or args.max_candidates < 0:
        parser.error("--repeats must be positive; --n and --max-candidates must be nonnegative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_checkpoint(args.ckpt, map_location="cpu")
    cfg = state["cfg"]
    if args.teacher:
        cfg.teacher_model = args.teacher
    if args.backbone:
        cfg.backbone_model = args.backbone
    torch.manual_seed(args.seed)
    _, tokenizer = load_qwen(cfg.teacher_model, str(device),
                             "bf16" if cfg.use_bf16 else "fp32")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    embedder = QwenEmbedder(cfg, tokenizer, device)
    cfg.embed_dim, cfg.vocab_size = embedder.H, embedder.V
    generator_cls = QwenCondGenerator if cfg.qwen_backbone else CondDriftGenerator
    gen = generator_cls(cfg, cfg.embed_dim).to(device)
    gen.load_state_dict(state[args.which])
    gen.eval()
    step = state.get("step")
    del state

    groups = load_candidate_groups(args.test_json, same_final=True,
                                    max_candidates=args.max_candidates)
    if not groups:
        raise ValueError("no examples found in --test-json")
    if args.n:
        groups = groups[:args.n]
    dataset = CandidateDataset(groups, tokenizer, cfg.query_len, cfg.resp_len,
                               args.max_candidates)
    print(f"[data] n={len(groups)} query_len={cfg.query_len} resp_len={cfg.resp_len} "
          f"query_truncated={dataset.truncated_queries} target_truncated={dataset.truncated_targets}")
    if dataset.truncated_targets:
        raise ValueError("test targets exceed the checkpoint response length; use compatible prepared data")

    results = []
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        torch.manual_seed(seed)
        metrics, shown = eval_one_pass(gen, dataset, embedder, tokenizer, cfg,
                                       device, len(groups))
        results.append(metrics)
        print(f"[eval repeat={repeat + 1}/{args.repeats} seed={seed}] " +
              " ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
        if repeat == 0:
            for sample in shown:
                print("  " + sample.replace("\n", "\n  "))
    mean = {key: sum(row[key] for row in results) / len(results)
            for key in results[0]}
    std = {key: (sum((row[key] - mean[key]) ** 2 for row in results) /
                  len(results)) ** 0.5 for key in mean}
    print("[eval mean +/- std] " + " ".join(
        f"{key}={mean[key]:.4f}+/-{std[key]:.4f}" for key in mean))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"checkpoint": args.ckpt, "step": step,
                                    "which": args.which, "test_json": args.test_json,
                                    "seed": args.seed, "repeats": results,
                                    "mean": mean, "std": std}, indent=2) + "\n",
                        encoding="utf-8")


if __name__ == "__main__":
    main()
