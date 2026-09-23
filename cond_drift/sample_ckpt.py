"""Generate and save samples from a conditional-drift checkpoint.

For checkpoints taken during unconditional warm-up, pass --uncond so the query is
an all-masked dummy context. The generator then samples from z only, matching the
warm-up behavior used before real prefixes are enabled.
"""
import argparse
import importlib
import json
import os
import sys

import torch


def _alias_packages():
    pkg = __package__ or "cond_drift"
    try:
        sys.modules.setdefault("cond_drift", importlib.import_module(pkg))
    except Exception:
        pass


def _import_local():
    try:
        from .config import CondDriftConfig  # noqa: F401
        from .cond_generator import CondDriftGenerator
        from .evaluate import generative_ppl, token_stats
        from .qwen_features import QwenEmbedder
        return CondDriftGenerator, QwenEmbedder, generative_ppl, token_stats
    except ImportError:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        from cond_drift.config import CondDriftConfig  # noqa: F401
        from cond_drift.cond_generator import CondDriftGenerator
        from cond_drift.evaluate import generative_ppl, token_stats
        from cond_drift.qwen_features import QwenEmbedder
        return CondDriftGenerator, QwenEmbedder, generative_ppl, token_stats


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True, help="jsonl output path")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--which", choices=["ema", "model"], default="ema")
    p.add_argument("--temp", type=float, default=None)
    p.add_argument("--eval-ppl-model", default=None,
                   help="reference LM for gen-PPL; default uses cfg.eval_ppl_model or gpt2-large")
    p.add_argument("--no-metrics", action="store_true",
                   help="skip gen-PPL / entropy computation after writing samples")
    p.add_argument("--uncond", action="store_true",
                   help="use an all-masked dummy query for unconditional warm-up sampling")
    p.add_argument("--show", type=int, default=10)
    args = p.parse_args()

    _alias_packages()
    CondDriftGenerator, QwenEmbedder, generative_ppl, token_stats = _import_local()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    temp = cfg.temp if args.temp is None else args.temp

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    embedder = QwenEmbedder(cfg, tokenizer, device)
    gen = CondDriftGenerator(cfg, embed_dim=embedder.H).to(device)
    gen.load_state_dict(ckpt[args.which])
    gen.eval()

    pad_id = tokenizer.pad_token_id
    q_ids = torch.full((args.batch, cfg.query_len), pad_id, dtype=torch.long, device=device)
    if args.uncond:
        q_msk = torch.zeros((args.batch, cfg.query_len), dtype=torch.bool, device=device)
    else:
        q_msk = torch.ones((args.batch, cfg.query_len), dtype=torch.bool, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    written = 0
    shown = []
    texts = []
    id_lists = []
    with open(args.out, "w", encoding="utf-8") as f:
        for start in range(0, args.n, args.batch):
            b = min(args.batch, args.n - start)
            qi = q_ids[:b]
            qm = q_msk[:b]
            qe = embedder.query_embeds(qi)
            z = gen.sample_z(b, cfg.noise_dim, temp, device)
            emb = gen(qe, qm, z)
            if cfg.sphere_norm:
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            toks = embedder.decode(emb)
            for row in toks:
                ids = row.tolist()
                if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in ids:
                    ids = ids[:ids.index(tokenizer.eos_token_id)]
                text = tokenizer.decode(ids, skip_special_tokens=True)
                texts.append(text)
                id_lists.append(ids)
                rec = {"idx": written, "step": int(ckpt.get("step", -1)),
                       "which": args.which, "temp": temp, "text": text}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if len(shown) < args.show:
                    shown.append(text)
                written += 1

    print(f"[sample] wrote {written} samples to {args.out}")
    if not args.no_metrics:
        ent, uniq = token_stats(id_lists)
        ppl_model = args.eval_ppl_model or getattr(cfg, "eval_ppl_model", "gpt2-large")
        ppl = generative_ppl(texts, device, eval_model=ppl_model)
        print(f"[metrics] gen_ppl={ppl:.3f} entropy={ent:.3f} uniq_tok={uniq:.3f} "
              f"eval_ppl_model={ppl_model}")
    for i, text in enumerate(shown, 1):
        print(f"\n[{i}] {text!r}")


if __name__ == "__main__":
    main()
