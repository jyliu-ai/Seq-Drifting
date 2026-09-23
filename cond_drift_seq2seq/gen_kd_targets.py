"""Generate sequence-level KD targets by running beam search on the teacher.

The teacher is the same causal GPT-2 fine-tuned on source -> target that training
uses for its support sets. Beam search produces one deterministic sequence per source
sentence; that sequence is then used as the target in place of the human reference.

Why this is different from CE supervision: the beam output is a fixed function f(x),
so training sees a delta distribution over targets. A factorised NAR student CAN fit a
delta distribution (each position's marginal is also a delta), whereas it cannot fit
the human reference distribution (which has multi-modal cross-sentence variance that
factorisation cannot express).

This script writes its output in the same memmap format that prepare_data.py produces,
so data.py and the training loop need no changes. The written split is named by --out-tag
and shares the same condition and target lengths as the original data.

Usage (validation first, to check C_top1 in diag_prefix before committing GPU-hours):

    # single-GPU, ~5-10 min for 3000 sentences
    python -m cond_drift_seq2seq.gen_kd_targets \
        --ckpt runs/interp_best.pt \
        --split validation --num-beams 4 --batch-size 32 \
        --out-tag kd4

    # then check C_top1 against the KD target instead of human gold:
    python -m cond_drift_seq2seq.diag_prefix \
        --ckpt runs/interp_best.pt --which ema \
        --split validation --n 512 --kd-target data/wmt14_de_en/validation_kd4

Full training set (4.5M sentences, 4 GPUs, shard 0-3):
    for SHARD in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$SHARD python -m cond_drift_seq2seq.gen_kd_targets \
          --ckpt runs/interp_best.pt --split train --num-beams 4 --batch-size 32 \
          --out-tag kd4 --shard $SHARD --n-shards 4 &
    done
    wait
    # then merge the shard files with gen_kd_targets.py --merge
"""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Seq2SeqDriftConfig
from .data import Seq2SeqDataset, build_dataset


def _load_teacher(cfg, device):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.teacher_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if cfg.use_bf16 and device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model, torch_dtype=dtype, local_files_only=True).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


@torch.no_grad()
def _beam_batch(model, tokenizer, query_ids, query_mask, target_len, num_beams, device):
    """Run beam search on one batch. Returns (B, target_len) token ids, right-padded."""
    B = query_ids.shape[0]
    # Clamp source length so source + target fits in the model's position table.
    max_pos = getattr(model.config, "n_positions",
                      getattr(model.config, "max_position_embeddings", None))
    if max_pos is not None and query_ids.shape[1] > max_pos - target_len:
        keep = max_pos - target_len
        # Source is left-padded; drop the leftmost (padding) tokens.
        query_ids = query_ids[:, -keep:]
        query_mask = query_mask[:, -keep:]
    input_ids = query_ids.to(device)
    attn = query_mask.to(device).long()
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=target_len,
        num_beams=num_beams,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # out is (B, Lq + up_to_target_len); strip the query prefix
    gen = out[:, query_ids.shape[1]:]
    # Pad or truncate to exactly target_len
    if gen.shape[1] < target_len:
        pad = torch.full(
            (B, target_len - gen.shape[1]), tokenizer.pad_token_id,
            dtype=torch.long, device=device)
        gen = torch.cat([gen, pad], dim=1)
    else:
        gen = gen[:, :target_len]
    return gen.cpu()


def _write_memmap(path: Path, data: np.ndarray):
    mm = np.memmap(str(path), dtype=data.dtype, mode="w+", shape=data.shape)
    mm[:] = data
    mm.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="",
                    help="student checkpoint -- used only for cfg; omit when using --teacher")
    ap.add_argument("--teacher", default="",
                    help="HuggingFace teacher directory; alternative to --ckpt when no "
                         "student checkpoint exists yet")
    ap.add_argument("--dataset", default="wmt14_de_en",
                    help="dataset name when using --teacher (default: wmt14_de_en)")
    ap.add_argument("--condition-len", type=int, default=0,
                    help="condition length when using --teacher (0 = use dataset default)")
    ap.add_argument("--target-len", type=int, default=64,
                    help="target length when using --teacher")
    ap.add_argument("--split", choices=["train", "validation", "test"],
                    default="validation")
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--out-dir", default="",
                    help="directory to write the KD memmap files into; "
                         "defaults to the same directory as the source data")
    ap.add_argument("--out-tag", default="kd4",
                    help="tag appended to the split name: validation_kd4")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--shard", type=int, default=0,
                    help="0-based shard index for parallel generation")
    ap.add_argument("--n-shards", type=int, default=1,
                    help="total number of shards; 1 = no sharding")
    ap.add_argument("--merge", action="store_true",
                    help="instead of generating, merge shard files written by "
                         "--n-shards>1 into a single split")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most this many sentences (0 = all); useful for "
                         "a quick sanity-check before committing to the full split")
    args = ap.parse_args()

    if not args.ckpt and not args.teacher:
        ap.error("one of --ckpt or --teacher is required")

    if args.teacher:
        # Build a minimal cfg from command-line args when no student checkpoint exists.
        from .config import Seq2SeqDriftConfig
        _COND_DEFAULTS = {"wmt14_de_en": 64, "xsum": 1024}
        cfg = Seq2SeqDriftConfig()
        cfg.teacher_model = args.teacher
        cfg.dataset_name = args.dataset
        clen = args.condition_len or _COND_DEFAULTS.get(args.dataset, 64)
        cfg.query_len = clen
        cfg.resp_len = args.target_len
        cfg.use_bf16 = True
        if args.data_dir:
            cfg.data_dir = args.data_dir
        else:
            ap.error("--data-dir is required when using --teacher")
    else:
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg: Seq2SeqDriftConfig = blob["cfg"]
        if args.data_dir:
            cfg.data_dir = args.data_dir
    data_dir = Path(cfg.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir

    if args.merge:
        _merge(data_dir, out_dir, args.split, args.out_tag, args.n_shards, cfg)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_teacher(cfg, device)

    # Build the dataset using the ORIGINAL split to get source ids and text refs.
    ds = Seq2SeqDataset(
        data_dir, args.split, tokenizer, cfg.dataset_name,
        cfg.query_len, cfg.resp_len)
    total = len(ds) if args.limit == 0 else min(args.limit, len(ds))

    # Shard: this worker processes rows [shard_start, shard_end).
    shard_size = math.ceil(total / args.n_shards)
    shard_start = args.shard * shard_size
    shard_end = min(shard_start + shard_size, total)
    n = shard_end - shard_start
    if n <= 0:
        print(f"[shard {args.shard}] nothing to do (total={total})")
        return

    tag = args.out_tag
    shard_suffix = f"_s{args.shard}of{args.n_shards}" if args.n_shards > 1 else ""
    out_prefix = out_dir / f"{args.split}_{tag}{shard_suffix}"

    target_ids_out = np.full((n, cfg.resp_len), tokenizer.pad_token_id, dtype="<u4")
    target_lens_out = np.zeros(n, dtype="<u2")
    refs_out = []

    print(f"[gen_kd] split={args.split} shard={args.shard}/{args.n_shards} "
          f"rows={shard_start}-{shard_end} beams={args.num_beams} "
          f"device={device} teacher={cfg.teacher_model}")

    for batch_start in range(0, n, args.batch_size):
        batch_end = min(batch_start + args.batch_size, n)
        global_start = shard_start + batch_start
        global_end = shard_start + batch_end
        idx = torch.arange(global_start, global_end)
        query_ids, query_mask, _, _ = ds.rows(idx)
        gen = _beam_batch(model, tokenizer, query_ids, query_mask,
                          cfg.resp_len, args.num_beams, device)
        for i, row in enumerate(gen):
            ids = row.tolist()
            # compute effective length: stop at the first EOS (inclusive)
            if tokenizer.eos_token_id in ids:
                L = ids.index(tokenizer.eos_token_id) + 1
            else:
                L = cfg.resp_len
            target_ids_out[batch_start + i, :len(ids)] = ids
            target_lens_out[batch_start + i] = L
            # decode for the .refs.jsonl (source text from the original refs)
            text = tokenizer.decode(
                ids[:L - 1] if ids[L - 1] == tokenizer.eos_token_id else ids[:L],
                skip_special_tokens=True)
            refs_out.append(text)
        if (batch_start // args.batch_size) % 10 == 0:
            pct = (batch_end / n) * 100
            print(f"  [{pct:5.1f}%] {batch_end}/{n}", flush=True)

    _write_memmap(Path(str(out_prefix) + ".target.u32"), target_ids_out)
    _write_memmap(Path(str(out_prefix) + ".target_len.u16"), target_lens_out)
    print(f"[gen_kd] wrote {out_prefix}.target.u32  "
          f"shape={target_ids_out.shape}")

    # Write a minimal refs JSONL so diag_prefix --kd-target can read back both sides.
    refs_path = Path(str(out_prefix) + ".refs.jsonl")
    # Pull source text from original refs for cross-reference.
    orig_refs = None
    orig_refs_path = data_dir / f"{args.split}.refs.jsonl"
    if orig_refs_path.exists():
        with orig_refs_path.open(encoding="utf-8") as f:
            orig_refs = [json.loads(line) for line in f if line.strip()]
    with refs_path.open("w", encoding="utf-8") as f:
        for i, kd_ref in enumerate(refs_out):
            src = orig_refs[shard_start + i]["source"] if orig_refs else ""
            human = orig_refs[shard_start + i]["reference"] if orig_refs else ""
            f.write(json.dumps(
                {"source": src, "reference": kd_ref, "human_reference": human},
                ensure_ascii=False) + "\n")
    print(f"[gen_kd] wrote {refs_path}")

    # Write a metadata stub so Seq2SeqDataset can open this prefix as a split.
    meta_path = Path(str(out_prefix) + ".meta.json")
    orig_meta = json.loads((data_dir / f"{args.split}.meta.json").read_text())
    kd_meta = dict(orig_meta)
    kd_meta["kd_tag"] = tag
    kd_meta["kd_beams"] = args.num_beams
    kd_meta["count"] = n
    kd_meta["source_truncated"] = orig_meta.get("source_truncated", 0)
    kd_meta["target_truncated"] = 0    # KD targets are already capped at resp_len
    meta_path.write_text(json.dumps(kd_meta, indent=2), encoding="utf-8")
    print(f"[gen_kd] wrote {meta_path}")
    print(f"[gen_kd] done. verify with diag_prefix --kd-target {out_prefix}")


def _merge(data_dir, out_dir, split, tag, n_shards, cfg):
    """Concatenate per-shard target memmaps into a single split."""
    # Load first shard's meta to get resp_len
    shard0_meta = json.loads(
        (out_dir / f"{split}_{tag}_s0of{n_shards}.meta.json").read_text())
    resp_len = shard0_meta["target_len"]
    # Count total rows
    counts = []
    for s in range(n_shards):
        meta = json.loads(
            (out_dir / f"{split}_{tag}_s{s}of{n_shards}.meta.json").read_text())
        counts.append(meta["count"])
    total = sum(counts)
    ids_out = np.zeros((total, resp_len), dtype="<u4")
    lens_out = np.zeros(total, dtype="<u2")
    refs_out = []
    pos = 0
    for s in range(n_shards):
        c = counts[s]
        prefix = out_dir / f"{split}_{tag}_s{s}of{n_shards}"
        ids = np.memmap(str(prefix) + ".target.u32", dtype="<u4", mode="r",
                        shape=(c, resp_len))
        lens = np.memmap(str(prefix) + ".target_len.u16", dtype="<u2", mode="r",
                         shape=(c,))
        ids_out[pos:pos + c] = ids
        lens_out[pos:pos + c] = lens
        rp = Path(str(prefix) + ".refs.jsonl")
        if rp.exists():
            with rp.open(encoding="utf-8") as f:
                refs_out.extend(json.loads(l) for l in f if l.strip())
        pos += c
    out_prefix = out_dir / f"{split}_{tag}"
    _write_memmap(Path(str(out_prefix) + ".target.u32"), ids_out)
    _write_memmap(Path(str(out_prefix) + ".target_len.u16"), lens_out)
    if refs_out:
        with (Path(str(out_prefix) + ".refs.jsonl")).open("w", encoding="utf-8") as f:
            for r in refs_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = dict(shard0_meta)
    meta["count"] = total
    meta.pop("kd_tag", None)
    meta["kd_tag"] = tag
    (Path(str(out_prefix) + ".meta.json")).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[merge] wrote {out_prefix}.* total={total}")

    # Also copy source memmaps (unchanged) to the merged prefix so Seq2SeqDataset
    # can open the merged KD split directly.
    for ext in (".source.u32", ".source_len.u16"):
        src = data_dir / f"{split}{ext}"
        dst = Path(str(out_prefix) + ext)
        if src.exists() and not dst.exists():
            import shutil
            shutil.copy2(src, dst)
            print(f"[merge] linked {dst.name}")


if __name__ == "__main__":
    main()
