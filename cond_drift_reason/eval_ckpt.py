"""Evaluate a TRAINED checkpoint (no training): run generation over a split, report
accuracy / entropy / gen-ppl and print sample cases. Single GPU.

  # test set of the dataset it was trained on (arch/cfg read from the ckpt):
  python -m cond_drift_reason.eval_ckpt --ckpt runs/reason_.../step_5000.pt
  # overfit check -- eval the TRAIN split instead:
  python -m cond_drift_reason.eval_ckpt --ckpt runs/.../step_5000.pt --eval-on-train
  # eval a DIFFERENT dataset with the same model:
  python -m cond_drift_reason.eval_ckpt --ckpt runs/.../step_5000.pt \
      --dataset svamp --svamp-path .../svamp.json
"""
import argparse
import json

import torch

from .data import build_dataset
from .qwen_features import load_qwen, QwenEmbedder
from .cond_generator import QwenCondGenerator, CondDriftGenerator
from .evaluate import run_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to a step_*.pt checkpoint")
    p.add_argument("--teacher", default="", help="relocated copy of the original embedding model")
    p.add_argument("--backbone", default="", help="relocated copy of the original generator backbone")
    p.add_argument("--which", choices=["ema", "model"], default="ema",
                   help="which weights to eval (ema is what training reports)")
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument("--eval-on-train", dest="eval_on_train", action="store_true",
                   help="alias for --split train (overfit diagnostic)")
    # optional overrides (default: reuse the checkpoint's own cfg)
    p.add_argument("--dataset", dest="dataset_name", default="")
    p.add_argument("--gsm8k-json", dest="local_json", default="")
    p.add_argument("--test-json", dest="test_json", default="")
    p.add_argument("--svamp-path", dest="svamp_path", default="")
    p.add_argument("--gpqa-path", dest="gpqa_path", default="")
    p.add_argument("--n", type=int, default=0, help="queries to eval (0 = whole split)")
    p.add_argument("--n-show", dest="n_show", type=int, default=8, help="cases to print")
    p.add_argument("--no-gen-ppl", dest="eval_gen_ppl", action="store_false", default=None,
                   help="skip the gpt2-large gen-ppl (faster)")
    p.add_argument("--out-jsonl", dest="out_jsonl", default="",
                   help="write one JSON record per example to this file")
    p.add_argument("--self-cond-steps", dest="self_cond_steps", type=int, default=0,
                   help="eval-time self-conditioning passes (0 = use cfg default; 1 = no self-cond)")
    p.add_argument("--self-cond-thresh", dest="self_cond_thresh", type=float, default=-1.0,
                   help="cosine-sim threshold for confident positions to feed back (default 0.9)")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(a.ckpt, map_location=device, weights_only=False)
    cfg = sd["cfg"]                                       # arch + data settings as trained
    # Back-compat: fill config fields added after this checkpoint was saved.
    for _f, _d in [("gold_decay_steps", 0), ("gold_min_ratio", 1.0),
                   ("no_repeat_window", 0), ("ce_weight", 0.0), ("ce_tau", 0.07),
                   ("ce_ramp", 0), ("lr_schedule", "none"), ("lr_warmup", 0),
                   ("lr_decay_steps", 0), ("lr_min_ratio", 0.1),
                   ("global_batch_size", 0), ("grad_accum_steps", 0),
                   ("self_cond", False), ("self_cond_prob", 0.5), ("self_cond_steps", 1),
                   ("self_cond_thresh", 0.9)]:
        if not hasattr(cfg, _f):
            setattr(cfg, _f, _d)
    print(f"[eval] loaded {a.ckpt} (step={sd.get('step','?')}); reusing its cfg "
          f"(backbone={cfg.teacher_model} Lq={cfg.query_len} Lr={cfg.resp_len} "
          f"chat={getattr(cfg,'use_chat_template',None)} strip={getattr(cfg,'strip_calc_annot',None)})")

    # eval-only overrides (leave everything else exactly as trained)
    if a.teacher:
        cfg.teacher_model = a.teacher
    if a.backbone:
        cfg.backbone_model = a.backbone
    if a.dataset_name:
        cfg.dataset_name = a.dataset_name
    if a.local_json:
        cfg.local_json = a.local_json
    if a.test_json:
        cfg.test_json = a.test_json
    if a.svamp_path:
        cfg.svamp_path = a.svamp_path
    if a.gpqa_path:
        cfg.gpqa_path = a.gpqa_path
    if a.eval_gen_ppl is not None:
        cfg.eval_gen_ppl = a.eval_gen_ppl
    if a.self_cond_steps > 0:
        cfg.self_cond_steps = a.self_cond_steps
    if a.self_cond_thresh >= 0:
        cfg.self_cond_thresh = a.self_cond_thresh
    split = "train" if a.eval_on_train else a.split

    _, tokenizer = load_qwen(cfg.teacher_model, str(device), "bf16" if cfg.use_bf16 else "fp32")
    embedder = QwenEmbedder(cfg, tokenizer, device)
    cfg.embed_dim, cfg.vocab_size = embedder.H, embedder.V

    gen = (QwenCondGenerator(cfg, cfg.embed_dim) if getattr(cfg, "qwen_backbone", False)
           else CondDriftGenerator(cfg, cfg.embed_dim)).to(device)
    missing, unexpected = gen.load_state_dict(sd[a.which], strict=False)
    if missing:
        print(f"[eval] load_state_dict missing keys (zero-init): {missing}")
    if unexpected:
        print(f"[eval] load_state_dict unexpected keys (ignored): {unexpected}")
    gen.eval()

    ds = build_dataset(cfg, tokenizer, split)
    n = a.n if a.n > 0 else len(ds)
    print(f"[eval] dataset={cfg.dataset_name} split={split} n={min(n,len(ds))}/{len(ds)} "
          f"weights={a.which}")

    need_records = bool(a.out_jsonl)
    result = run_eval(gen, embedder, ds, cfg, tokenizer, device, n, a.n_show,
                      return_records=need_records)
    metrics, shown = result[:2]
    ms = " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                  for k, v in metrics.items())
    print(f"[eval RESULT] {ms}")
    for s in shown:
        print("   " + s.replace("\n", "\n   "))
    if a.out_jsonl:
        records = result[2]
        with open(a.out_jsonl, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[eval] wrote {len(records)} records to {a.out_jsonl}")


if __name__ == "__main__":
    main()
