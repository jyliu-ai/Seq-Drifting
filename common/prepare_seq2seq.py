"""Download and tokenize the ELF WMT14/XSum seq2seq protocol into memmaps.

Examples:
  python -m common.prepare_seq2seq --dataset wmt14_de_en \
      --output-dir data/wmt14_de_en --tokenizer gpt2
  python -m common.prepare_seq2seq --dataset xsum \
      --output-dir data/xsum --tokenizer gpt2

Training reads the fixed-width binary files directly with numpy.memmap, so WMT14
does not get duplicated into every torchrun process's RAM.
"""

import argparse
import json
from pathlib import Path

import numpy as np


SPECS = {
    "wmt14_de_en": {
        "hf_name": "wmt/wmt14",
        "hf_config": "de-en",
        "condition_len": 64,
        "target_len": 64,
    },
    "xsum": {
        "hf_name": "EdinburghNLP/xsum",
        "hf_config": None,
        "condition_len": 1024,
        "target_len": 64,
    },
}


def _extract(row, dataset_name):
    if dataset_name == "wmt14_de_en":
        pair = row["translation"]
        return str(pair["de"]).strip(), str(pair["en"]).strip(), str(row.get("id", ""))
    return (str(row["document"]).strip(), str(row["summary"]).strip(),
            str(row.get("id", "")))


def _paths(output_dir, split):
    root = Path(output_dir)
    return {
        "source": root / f"{split}.source.u32",
        "target": root / f"{split}.target.u32",
        "source_len": root / f"{split}.source_len.u16",
        "target_len": root / f"{split}.target_len.u16",
        "refs": root / f"{split}.refs.jsonl",
        "meta": root / f"{split}.meta.json",
    }


def _write_batch(batch, tokenizer, condition_len, target_len, handles, refs_handle,
                 counters):
    sources = [x[0] for x in batch]
    targets = [x[1] for x in batch]
    source_tok = tokenizer(sources, add_special_tokens=False, truncation=False)["input_ids"]
    target_tok = tokenizer(targets, add_special_tokens=False, truncation=False)["input_ids"]
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    src_rows, tgt_rows, src_lens, tgt_lens = [], [], [], []
    kept_text = []
    for record, src_ids, tgt_ids in zip(batch, source_tok, target_tok):
        if not src_ids or not tgt_ids:
            counters["skipped_empty"] += 1
            continue
        counters["source_truncated"] += int(len(src_ids) > condition_len)
        counters["target_truncated"] += int(len(tgt_ids) + 1 > target_len)

        # ELF fixes the condition length. Keep the beginning of long documents (important
        # for XSum) and left-pad so the last non-pad source state remains adjacent to target 0.
        src_ids = src_ids[:condition_len]
        src_len = len(src_ids)
        src_rows.append([pad_id] * (condition_len - src_len) + src_ids)
        src_lens.append(src_len)

        tgt_ids = tgt_ids[:target_len - 1] + [eos_id]
        tgt_len = len(tgt_ids)
        tgt_rows.append(tgt_ids + [pad_id] * (target_len - tgt_len))
        tgt_lens.append(tgt_len)
        kept_text.append(record)

    if not src_rows:
        return
    np.asarray(src_rows, dtype="<u4").tofile(handles["source"])
    np.asarray(tgt_rows, dtype="<u4").tofile(handles["target"])
    np.asarray(src_lens, dtype="<u2").tofile(handles["source_len"])
    np.asarray(tgt_lens, dtype="<u2").tofile(handles["target_len"])
    if refs_handle is not None:
        for source, target, item_id in kept_text:
            refs_handle.write(json.dumps(
                {"id": item_id, "source": source, "reference": target},
                ensure_ascii=False) + "\n")
    counters["count"] += len(src_rows)


def prepare_split(dataset, dataset_name, split, output_dir, tokenizer,
                  condition_len, target_len, batch_size, max_examples,
                  keep_text, overwrite):
    paths = _paths(output_dir, split)
    known = list(paths.values())
    existing = [p for p in known if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{existing[0]} already exists; pass --overwrite to replace this split")
    if overwrite:
        for path in existing:
            path.unlink()

    binary_keys = ("source", "target", "source_len", "target_len")
    handles = {key: paths[key].open("wb") for key in binary_keys}
    refs_handle = paths["refs"].open("w", encoding="utf-8") if keep_text else None
    counters = {"count": 0, "skipped_empty": 0,
                "source_truncated": 0, "target_truncated": 0}
    batch = []
    try:
        for row in dataset:
            record = _extract(row, dataset_name)
            if not record[0] or not record[1]:
                counters["skipped_empty"] += 1
                continue
            batch.append(record)
            if len(batch) >= batch_size:
                if max_examples:
                    batch = batch[:max_examples - counters["count"]]
                _write_batch(batch, tokenizer, condition_len, target_len,
                             handles, refs_handle, counters)
                batch.clear()
                if counters["count"] and counters["count"] % 100_000 < batch_size:
                    print(f"[{split}] {counters['count']:,} examples")
                if max_examples and counters["count"] >= max_examples:
                    break
        if batch and (not max_examples or counters["count"] < max_examples):
            if max_examples:
                batch = batch[:max_examples - counters["count"]]
            _write_batch(batch, tokenizer, condition_len, target_len,
                         handles, refs_handle, counters)
    finally:
        for handle in handles.values():
            handle.close()
        if refs_handle is not None:
            refs_handle.close()

    if counters["count"] == 0:
        raise RuntimeError(f"no usable records were produced for split {split}")
    meta = {
        "dataset": dataset_name,
        "split": split,
        "count": counters["count"],
        "condition_len": condition_len,
        "target_len": target_len,
        "tokenizer": tokenizer.name_or_path,
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        **{k: v for k, v in counters.items() if k != "count"},
    }
    paths["meta"].write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(f"[{split}] wrote {counters['count']:,} examples to {output_dir}; "
          f"source_truncated={counters['source_truncated']:,}, "
          f"target_truncated={counters['target_truncated']:,}")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--hf-name", default="", help="override the Hugging Face dataset repo")
    parser.add_argument("--hf-config", default="", help="override its dataset config")
    parser.add_argument("--condition-len", type=int, default=0)
    parser.add_argument("--target-len", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=0)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", default=True)
    parser.add_argument("--keep-train-text", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    spec = SPECS[args.dataset]
    condition_len = args.condition_len or spec["condition_len"]
    target_len = args.target_len or spec["target_len"]
    if condition_len <= 0 or target_len < 2 or args.batch_size <= 0:
        raise ValueError("condition_len and batch_size must be positive; target_len must be >= 2")
    if condition_len > np.iinfo(np.uint16).max or target_len > np.iinfo(np.uint16).max:
        raise ValueError("condition/target lengths must fit uint16")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("the tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_meta = {}
    for split in args.splits:
        hf_name = args.hf_name or spec["hf_name"]
        hf_config = args.hf_config or spec["hf_config"]
        print(f"[load] {hf_name} {hf_config or ''} split={split} "
              f"streaming={args.streaming}")
        ds = load_dataset(hf_name, hf_config, split=split,
                          streaming=args.streaming)
        max_examples = args.max_train if split == "train" else args.max_eval
        all_meta[split] = prepare_split(
            ds, args.dataset, split, output_dir, tokenizer,
            condition_len, target_len, args.batch_size, max_examples,
            keep_text=(split != "train" or args.keep_train_text),
            overwrite=args.overwrite)

    manifest = {
        "format": "seq_drifting_memmap_v1",
        "dataset": args.dataset,
        "condition_len": condition_len,
        "target_len": target_len,
        "tokenizer": tokenizer.name_or_path,
        "splits": {split: meta["count"] for split, meta in all_meta.items()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
