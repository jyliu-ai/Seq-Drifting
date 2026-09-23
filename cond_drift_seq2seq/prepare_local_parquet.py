"""Convert local WMT14 Parquet shards into the seq2seq memmap format."""

import argparse
from pathlib import Path

from .prepare_data import prepare_split


def _iter_parquet(paths, rows_per_batch):
    import pyarrow.parquet as pq

    for path in paths:
        parquet = pq.ParquetFile(path)
        if "translation" not in parquet.schema_arrow.names:
            raise ValueError(f"{path} does not contain a translation column")
        for batch in parquet.iter_batches(
                batch_size=rows_per_batch, columns=["translation"]):
            yield from batch.to_pylist()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--validation", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--condition-len", type=int, default=64)
    parser.add_argument("--target-len", type=int, default=64)
    parser.add_argument("--tokenize-batch-size", type=int, default=4096)
    parser.add_argument("--parquet-batch-size", type=int, default=65536)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.condition_len <= 0 or args.target_len < 2:
        raise ValueError("condition_len must be positive and target_len must be at least 2")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=args.local_files_only)
    if tokenizer.eos_token_id is None:
        raise ValueError("the tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_files = {
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
    }
    for split, files in split_files.items():
        missing = [path for path in files if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"missing {split} Parquet files: {missing}")
        print(f"[{split}] reading {len(files)} local Parquet shard(s)", flush=True)
        prepare_split(
            _iter_parquet(files, args.parquet_batch_size),
            "wmt14_de_en",
            split,
            output_dir,
            tokenizer,
            args.condition_len,
            args.target_len,
            args.tokenize_batch_size,
            max_examples=0,
            keep_text=split != "train",
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
