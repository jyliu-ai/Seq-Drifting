"""Sample an unconditional drifting checkpoint and score the saved generations.

Run from the parent directory, for example:

    python -m uncond_drift.sample_ckpt \
      --ckpt runs/unconditional/step_200000.pt \
      --out uncond_1000_samples.jsonl \
      --n-samples 1000 --length 128 --which ema

The script saves one JSON object per generated sample and computes Gen-PPL and
entropy on exactly the same saved token sequences.
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent.parent))
    __package__ = here.parent.name

from .config import TextDriftConfig  # noqa: F401 needed when unpickling cfg
from .evaluate import gpt2_large_ppl_entropy
from .generator import TextDriftGenerator
from .gpt2_features import TokenEmbedder


def _alias_legacy_package():
    """Allow loading checkpoints pickled under either package name."""
    pkg = __package__ or "uncond_drift"
    sys.modules.setdefault("uncond_drift", importlib.import_module(pkg))
    for sub in (
        "config",
        "generator",
        "gpt2_features",
        "gpt2_teacher",
        "drift_loss",
        "memory_bank",
        "evaluate",
    ):
        try:
            sys.modules.setdefault(
                f"uncond_drift.{sub}", importlib.import_module(f"{pkg}.{sub}")
            )
        except Exception:
            pass


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint path")
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--length", type=int, default=None, help="tokens to save/score")
    p.add_argument("--chunk", type=int, default=64, help="generation chunk size")
    p.add_argument("--which", default="ema", choices=["ema", "model"],
                   help="which checkpoint weights to sample")
    p.add_argument("--temp", type=float, default=None,
                   help="latent noise temperature; default uses checkpoint cfg.temp")
    p.add_argument("--ppl-batch", type=int, default=32)
    p.add_argument("--no-metrics", action="store_true",
                   help="save samples without computing Gen-PPL/entropy")
    p.add_argument("--show", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _alias_legacy_package()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    length = args.length if args.length is not None else cfg.seq_len
    if length > cfg.seq_len:
        raise ValueError(
            f"--length {length} is larger than checkpoint seq_len {cfg.seq_len}. "
            "This generator can only save/score up to its trained sequence length."
        )

    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg.tokenizer_name)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    embedder = TokenEmbedder(
        cfg.embed_model,
        tokenizer=tokenizer,
        mask_illegal=cfg.mask_illegal_tokens,
        sphere=cfg.sphere_norm,
    ).to(device).eval()
    gen = TextDriftGenerator(cfg, wte=embedder.wte_weight).to(device)
    gen.load_state_dict(ckpt[args.which])
    gen.eval()

    temp = args.temp if args.temp is not None else cfg.temp
    print(
        f"[sample] ckpt={args.ckpt} step={ckpt.get('step', '?')} which={args.which} "
        f"ckpt_T={cfg.seq_len} save_T={length} temp={temp}",
        flush=True,
    )

    tokens_all = []
    t_gen = t_dec = 0.0
    for start in range(0, args.n_samples, args.chunk):
        n = min(args.chunk, args.n_samples - start)
        z = gen.sample_z(n, cfg.noise_dim, temp, device)
        _sync(device)
        t0 = time.perf_counter()
        emb = gen(z)
        _sync(device)
        t1 = time.perf_counter()
        ids = embedder.decode(emb)[:, :length].contiguous()
        _sync(device)
        t2 = time.perf_counter()
        tokens_all.append(ids.cpu())
        t_gen += t1 - t0
        t_dec += t2 - t1

    tokens = torch.cat(tokens_all, dim=0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(tokens.tolist()):
            text = tokenizer.decode(row, skip_special_tokens=True)
            rec = {
                "idx": i,
                "step": ckpt.get("step"),
                "which": args.which,
                "temp": temp,
                "length": length,
                "tokens": row,
                "text": text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"[sample] saved {tokens.shape[0]} samples to {out_path} "
        f"(gen {t_gen:.3f}s, decode {t_dec:.3f}s)",
        flush=True,
    )
    print(
        f"[sample] time/sample: gen {t_gen / tokens.shape[0] * 1000:.3f}ms "
        f"decode {t_dec / tokens.shape[0] * 1000:.3f}ms",
        flush=True,
    )
    distinct = torch.unique(tokens, dim=0).shape[0]
    print(f"[sample] distinct_seqs={distinct}/{tokens.shape[0]}", flush=True)

    if not args.no_metrics:
        wrapped = torch.cat(
            [
                torch.full((tokens.shape[0], 1), bos_id, dtype=tokens.dtype),
                tokens,
                torch.full((tokens.shape[0], 1), eos_id, dtype=tokens.dtype),
            ],
            dim=1,
        ).to(device)
        ppl, ent = gpt2_large_ppl_entropy(wrapped, device, batch_size=args.ppl_batch)
        print(f"[sample] gen_ppl={ppl:.3f} entropy={ent:.3f}", flush=True)

    for i in range(min(args.show, tokens.shape[0])):
        print(f"[sample {i}] {tokenizer.decode(tokens[i].tolist(), skip_special_tokens=True)!r}",
              flush=True)


if __name__ == "__main__":
    main()
