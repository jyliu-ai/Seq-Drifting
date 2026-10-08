"""Evaluate 128-token conditional warm-up weights or the original long unconditional model."""
import argparse
import json
import time
from pathlib import Path

import torch

from common.checkpoint import load_checkpoint
from common.metrics import generative_ppl, token_stats, self_bleu, gpt2_large_ppl_entropy
from models.embeddings import FrozenEmbedder, TokenEmbedder, load_teacher
from models.generators import CondDriftGenerator, TextDriftGenerator


def warmup_context(cfg, tokenizer, batch, device, mode="constant"):
    """Reproduce training's constant prefix with visible attention by default."""
    kind = getattr(cfg, "warmup_ctx", "pad")
    token_id = getattr(tokenizer, f"{kind}_token_id", None)
    if token_id is None:
        token_id = tokenizer.eos_token_id
    if token_id is None:
        raise ValueError("warm-up requires a pad / EOS / BOS token")
    ids = torch.full((batch, cfg.query_len), token_id, dtype=torch.long, device=device)
    mask = torch.full_like(ids, mode == "constant", dtype=torch.bool)
    return ids, mask


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="local path, release filename, or hf://owner/repo/file.pt")
    parser.add_argument("--which", choices=["ema", "model"], default="ema")
    parser.add_argument("--architecture", choices=["auto", "warmup", "unconditional"], default="auto")
    parser.add_argument("--teacher", default="", help="original frozen embedding model or relocated copy")
    parser.add_argument("--tokenizer-name", default="")
    parser.add_argument("--embed-model", default="")
    parser.add_argument("--warmup-context", choices=["constant", "masked"], default="constant",
                        help="constant matches training; masked is only for historical all-mask runs")
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--show", type=int, default=5)
    parser.add_argument("--no-ppl", action="store_true")
    parser.add_argument("--eval-ppl-model", default="gpt2-large")
    parser.add_argument("--ppl-batch", type=int, default=16)
    parser.add_argument("--no-self-bleu", action="store_true")
    parser.add_argument("--self-bleu-samples", type=int, default=500)
    parser.add_argument("--out", default="", help="optional generated-sample JSONL")
    args = parser.parse_args()
    if args.n_samples < 1 or args.chunk < 1 or args.ppl_batch < 1:
        parser.error("sample count and batch sizes must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_checkpoint(args.ckpt)
    cfg = state["cfg"]
    weights = state[args.which]
    has_query = "in_query.weight" in weights
    architecture = "warmup" if has_query else "unconditional"
    if args.architecture != "auto" and args.architecture != architecture:
        parser.error(f"checkpoint architecture is {architecture}, requested {args.architecture}")
    torch.manual_seed(args.seed)
    temp = cfg.temp if args.temp is None else args.temp

    if has_query:
        if args.teacher or args.embed_model:
            cfg.teacher_model = args.teacher or args.embed_model
        _, tokenizer = load_teacher(cfg.teacher_model, str(device),
                                     "bf16" if cfg.use_bf16 else "fp32")
        if args.tokenizer_name:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        embedder = FrozenEmbedder(cfg, tokenizer, device)
        gen = CondDriftGenerator(cfg, embedder.H).to(device)
        length = cfg.resp_len
    else:
        from transformers import GPT2TokenizerFast
        tokenizer_name = args.tokenizer_name or cfg.tokenizer_name
        embedding_name = args.embed_model or args.teacher or cfg.embed_model
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        embedder = TokenEmbedder(embedding_name, tokenizer=tokenizer,
                                mask_illegal=cfg.mask_illegal_tokens,
                                sphere=cfg.sphere_norm).to(device).eval()
        gen = TextDriftGenerator(cfg, wte=embedder.wte_weight).to(device)
        length = cfg.seq_len
    gen.load_state_dict(weights)
    gen.eval()
    del state, weights
    print(f"[eval] architecture={architecture} length={length} which={args.which} "
          f"warmup_context={args.warmup_context if has_query else 'none'}")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    tokens, generation_seconds = [], 0.0
    for start in range(0, args.n_samples, args.chunk):
        batch = min(args.chunk, args.n_samples - start)
        z = gen.sample_z(batch, cfg.noise_dim, temp, device)
        if has_query:
            ids, mask = warmup_context(cfg, tokenizer, batch, device, args.warmup_context)
            query = embedder.query_embeds(ids)
        sync()
        begin = time.perf_counter()
        # Match the original open-ended training / evaluation's fp32 generator path.
        output = gen(query, mask, z) if has_query else gen(z)
        sync()
        generation_seconds += time.perf_counter() - begin
        if has_query and cfg.sphere_norm:
            output = torch.nn.functional.normalize(output.float(), dim=-1)
        tokens.append(embedder.decode(output).cpu())
    tokens = torch.cat(tokens)
    texts = [tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in tokens]
    entropy, unique = token_stats(tokens.tolist())
    metrics = {"n_samples": args.n_samples, "entropy": entropy, "uniq_tok": unique,
               "generation_seconds_per_sample": generation_seconds / args.n_samples}
    if not args.no_ppl:
        if has_query:
            metrics["gen_ppl"] = generative_ppl(texts, device, args.eval_ppl_model,
                                                 batch_size=args.ppl_batch, max_len=length)
        else:
            # Historical unconditional scoring wraps BOS + content + EOS.
            bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
            wrapped = torch.cat([torch.full((len(tokens), 1), bos, dtype=torch.long), tokens,
                                 torch.full((len(tokens), 1), tokenizer.eos_token_id, dtype=torch.long)], dim=1)
            if args.eval_ppl_model == "gpt2-large":
                metrics["gen_ppl"], metrics["lm_entropy"] = gpt2_large_ppl_entropy(
                    wrapped, device, batch_size=args.ppl_batch)
            else:
                metrics["gen_ppl"] = generative_ppl(texts, device, args.eval_ppl_model,
                                                     batch_size=args.ppl_batch, max_len=length)
    if not args.no_self_bleu:
        metrics["self_bleu"] = self_bleu(tokens, tokenizer.eos_token_id, tokenizer,
                                          sample_size=args.self_bleu_samples, seed=args.seed)
    print("[eval RESULT] " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()
                                     if value is not None))
    for text in texts[:args.show]:
        print("  " + repr(text))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for index, text in enumerate(texts):
                handle.write(json.dumps({"index": index, "text": text}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
