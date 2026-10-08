"""Evaluate a continuation checkpoint on held-out LM1B or OpenWebText prefixes."""

import argparse

import torch
from common.checkpoint import load_checkpoint

from models.generators import CondDriftGenerator
from .data import build_dataset
from .evaluate import run_eval
from models.embeddings import FrozenEmbedder as QwenEmbedder, load_teacher as load_qwen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", choices=["lm1b", "owt"], default="",
                        help="verify the checkpoint's continuation dataset")
    parser.add_argument("--owt-dir", default="")
    parser.add_argument("--test-jsonl-path", default="")
    parser.add_argument("--teacher", default="", help="original embedding model or relocated copy")
    parser.add_argument("--which", choices=["ema", "model"], default="ema")
    parser.add_argument("--n", type=int, default=0, help="0 evaluates all held-out prefixes")
    parser.add_argument("--n-show", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-ppl-model", default="gpt2-large")
    parser.add_argument("--no-gen-ppl", action="store_true")
    args = parser.parse_args()
    if args.n < 0 or args.n_show < 0:
        parser.error("--n and --n-show must be nonnegative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_checkpoint(args.ckpt, map_location="cpu")
    cfg = state["cfg"]
    if cfg.dataset_name not in {"lm1b", "owt"}:
        parser.error("expected an LM1B or OpenWebText continuation checkpoint")
    if args.dataset and args.dataset != cfg.dataset_name:
        parser.error(f"checkpoint dataset is {cfg.dataset_name}, requested {args.dataset}")
    if args.owt_dir:
        cfg.owt_dir = args.owt_dir
    if args.test_jsonl_path:
        cfg.test_jsonl_path = args.test_jsonl_path
    cfg.use_chat_template = False
    if args.teacher:
        cfg.teacher_model = args.teacher
    cfg.eval_accuracy = False
    cfg.eval_gen_ppl = not args.no_gen_ppl
    cfg.eval_ppl_model = args.eval_ppl_model
    torch.manual_seed(args.seed)

    _, tokenizer = load_qwen(cfg.teacher_model, str(device),
                             "bf16" if cfg.use_bf16 else "fp32")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    embedder = QwenEmbedder(cfg, tokenizer, device)
    gen = CondDriftGenerator(cfg, embed_dim=embedder.H).to(device)
    gen.load_state_dict(state[args.which])
    gen.eval()
    del state

    dataset = build_dataset(cfg, tokenizer, "test")
    if len(dataset) == 0:
        raise ValueError("no held-out prefixes found in the evaluation data")
    n = min(args.n, len(dataset)) if args.n else len(dataset)
    metrics, shown = run_eval(gen, embedder, dataset, cfg, tokenizer, device,
                              n, args.n_show)
    print("[eval RESULT] " + " ".join(f"{key}={value:.4f}"
                                     for key, value in metrics.items()))
    for sample in shown:
        print("  " + sample.replace("\n", "\n  "))


if __name__ == "__main__":
    main()
