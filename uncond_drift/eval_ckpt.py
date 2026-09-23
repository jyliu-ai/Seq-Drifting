"""Standalone evaluation of a trained checkpoint: Self-BLEU, entropy, and gen-PPL.

Loads a checkpoint, draws one-step samples, decodes them to tokens, and reports
the same three metrics the training-time eval uses, computed the same way so the
numbers line up with the run logs:

  - gen_ppl : exp(mean NLL) of GPT-2 Large over the decoded sequences (lower better)
  - entropy : GPT-2 Large mean per-token distribution entropy (nats)
  - self_bleu : Texygen-style Self-BLEU-4 among the samples (lower = more diverse)

The generation + decode path mirrors evaluate.run_eval, and the metrics reuse the
exact functions from evaluate.py, so this is a drop-in re-measurement of a saved
model (e.g. the best LM1B checkpoint) without rerunning training.

Run it as a module from the package's parent directory (imports are relative):

    python -m uncond_drift.eval_ckpt --ckpt runs/.../step_XXXX.pt --n-samples 1024 --which ema
"""
import argparse
import time
import importlib
import sys

import torch

from .config import TextDriftConfig  # noqa: F401  (needed to unpickle cfg from the ckpt)
from .gpt2_features import TokenEmbedder
from .generator import TextDriftGenerator
from .evaluate import gpt2_large_ppl_entropy, self_bleu


def _alias_legacy_package():
    """Some checkpoints were saved when this package was imported as 'uncond_drift';
    the pickled cfg then references 'uncond_drift.config' etc. Alias that name to the
    current package so torch.load can unpickle those checkpoints too."""
    pkg = __package__ or "uncond_drift"
    sys.modules.setdefault("uncond_drift", importlib.import_module(pkg))
    for sub in ("config", "generator", "gpt2_features", "gpt2_teacher",
                "drift_loss", "memory_bank", "evaluate"):
        try:
            sys.modules.setdefault(f"uncond_drift.{sub}",
                                   importlib.import_module(f"{pkg}.{sub}"))
        except Exception:
            pass


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to a saved checkpoint (.pt)")
    p.add_argument("--tokenizer-name", default="", help="relocated copy of the training tokenizer")
    p.add_argument("--embed-model", default="", help="relocated copy of the frozen embedding model")
    p.add_argument("--n-samples", type=int, default=1024, help="samples to draw and score")
    p.add_argument("--chunk", type=int, default=64, help="generation chunk (caps peak memory)")
    p.add_argument("--warmup", type=int, default=1,
                   help="untimed warm-up chunks before measuring (CUDA/cuDNN init)")
    p.add_argument("--which", default="ema", choices=["ema", "model"],
                   help="which weights to evaluate (EMA is what the run reports)")
    p.add_argument("--self-bleu-samples", type=int, default=500,
                   help="subsample size for the O(n^2) Self-BLEU (0 = use all n_samples)")
    p.add_argument("--self-bleu-gram", type=int, default=4,
                   help="Self-BLEU n-gram order (Texygen's default is 3)")
    p.add_argument("--self-bleu-mode", default="pairwise", choices=["pairwise", "multiref"],
                   help="pairwise (single-ref mean, discriminative) or multiref (Texygen, saturates)")
    p.add_argument("--self-bleu-refs", type=int, default=50,
                   help="pairwise mode: how many other samples each sample is compared against")
    p.add_argument("--ppl-batch", type=int, default=32, help="GPT-2 Large scoring batch")
    p.add_argument("--no-ppl", action="store_true", help="skip gen-PPL / entropy (Self-BLEU only)")
    p.add_argument("--no-self-bleu", action="store_true", help="skip Self-BLEU (gen-PPL only)")
    p.add_argument("--temp", type=float, default=None,
                   help="latent noise temperature for z (default: the checkpoint's cfg.temp)")
    p.add_argument("--show", type=int, default=5, help="decoded samples to print")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _alias_legacy_package()   # allow loading ckpts pickled under the old 'uncond_drift' name
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    if args.tokenizer_name:
        cfg.tokenizer_name = args.tokenizer_name
    if args.embed_model:
        cfg.embed_model = args.embed_model
    print(f"[eval] {args.ckpt} (step {ckpt.get('step', '?')})  seq_len={cfg.seq_len}  "
          f"sphere={cfg.sphere_norm}  teacher={cfg.gpt2_teacher}  which={args.which}", flush=True)

    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained(cfg.tokenizer_name)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    embedder = TokenEmbedder(cfg.embed_model, tokenizer=tokenizer,
                             mask_illegal=cfg.mask_illegal_tokens,
                             sphere=cfg.sphere_norm).to(device).eval()
    gen = TextDriftGenerator(cfg, wte=embedder.wte_weight).to(device)
    gen.load_state_dict(ckpt[args.which])
    gen.eval()

    temp = args.temp if args.temp is not None else cfg.temp
    print(f"[eval] latent temp={temp}  (ckpt cfg.temp={cfg.temp})", flush=True)

    # ---- draw n_samples one-step generations and decode to token ids
    chunks = []
    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()   # CUDA is async: without this we would time launches

    # Warm-up. The first forward pays CUDA context init, cuDNN autotune and allocator
    # growth -- a large one-off cost that would inflate time/sample if it were timed.
    with torch.no_grad():
        for _ in range(args.warmup):
            zw = gen.sample_z(min(args.chunk, args.n_samples), cfg.noise_dim, temp, device)
            embedder.decode(gen(zw))
    _sync()

    t_gen = t_dec = 0.0
    for s in range(0, args.n_samples, args.chunk):
        n = min(args.chunk, args.n_samples - s)
        with torch.no_grad():                           # gen.eval() does NOT stop autograd;
            z = gen.sample_z(n, cfg.noise_dim, temp, device)   # building a graph here would
            _sync(); t0 = time.perf_counter()           # cost both time and memory
            emb = gen(z)                                # (n, T, D), single forward pass
            _sync(); t1 = time.perf_counter()
            ids = embedder.decode(emb)                  # (n, T), nearest-token ids
            _sync(); t2 = time.perf_counter()
        t_gen += t1 - t0
        t_dec += t2 - t1
        chunks.append(ids)
    tokens = torch.cat(chunks, dim=0)                   # (N, T)

    N = tokens.shape[0]
    print(f"[eval] T={cfg.seq_len}  chunk={args.chunk}  gen {t_gen:.3f}s + decode "
          f"{t_dec:.3f}s for {N} samples", flush=True)
    print(f"[eval] time/sample: gen {t_gen / N * 1000:.3f}ms  decode {t_dec / N * 1000:.3f}ms"
          f"  total {(t_gen + t_dec) / N * 1000:.3f}ms"
          f"   ({(t_gen + t_dec) / N / cfg.seq_len * 1e6:.2f}us/token)", flush=True)

    distinct = torch.unique(tokens, dim=0).shape[0]
    print(f"[eval] N={tokens.shape[0]}  distinct_seqs={distinct}/{tokens.shape[0]}", flush=True)

    # ---- gen-PPL + entropy under GPT-2 Large (wrap [BOS] content [EOS], as run_eval does)
    if not args.no_ppl:
        bos = torch.full((tokens.shape[0], 1), bos_id, dtype=tokens.dtype, device=device)
        eos = torch.full((tokens.shape[0], 1), eos_id, dtype=tokens.dtype, device=device)
        wrapped = torch.cat([bos, tokens, eos], dim=1)
        ppl, ent = gpt2_large_ppl_entropy(wrapped, device, batch_size=args.ppl_batch)
        print(f"[eval] gen_ppl={ppl:.3f}  entropy={ent:.3f}", flush=True)

    # ---- Self-BLEU (diversity across samples; lower = more repetitive)
    if not args.no_self_bleu:
        ss = args.self_bleu_samples if args.self_bleu_samples > 0 else tokens.shape[0]
        sb = self_bleu(tokens, eos_id, tokenizer, n_gram=args.self_bleu_gram, sample_size=ss,
                       mode=args.self_bleu_mode, n_refs=args.self_bleu_refs)
        if sb is not None:
            print(f"[eval] self_bleu={sb:.4f}  (mode={args.self_bleu_mode}, "
                  f"gram={args.self_bleu_gram}, sample_size={ss})", flush=True)

    for i in range(min(args.show, tokens.shape[0])):
        print("  " + repr(tokenizer.decode(tokens[i].tolist(), skip_special_tokens=True)), flush=True)


if __name__ == "__main__":
    main()
