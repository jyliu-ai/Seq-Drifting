"""Memory-mapped WMT14/XSum datasets produced by common.prepare_seq2seq."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class Seq2SeqDataset:
    def __init__(self, data_dir, split, tokenizer, expected_dataset,
                 condition_len, target_len, limit=0):
        self.root = Path(data_dir)
        self.split = split
        meta_path = self.root / f"{split}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"missing {meta_path}; run python -m common.prepare_seq2seq "
                f"--dataset {expected_dataset} --output-dir {self.root}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta["dataset"] != expected_dataset:
            raise ValueError(
                f"prepared dataset is {self.meta['dataset']}, requested {expected_dataset}")
        if self.meta["condition_len"] != condition_len or self.meta["target_len"] != target_len:
            raise ValueError(
                "prepared sequence lengths do not match the run: "
                f"prepared=({self.meta['condition_len']},{self.meta['target_len']}) "
                f"run=({condition_len},{target_len})")
        if self.meta["vocab_size"] != len(tokenizer):
            raise ValueError(
                f"prepared vocab size {self.meta['vocab_size']} != tokenizer vocab {len(tokenizer)}")
        if self.meta["eos_token_id"] != tokenizer.eos_token_id:
            raise ValueError("prepared data and run tokenizer have different EOS ids")

        count = int(self.meta["count"])
        self.count = min(count, limit) if limit else count
        self.condition_len = condition_len
        self.target_len = target_len
        self.pad_id = int(self.meta["pad_token_id"])
        self.eos_id = int(self.meta["eos_token_id"])
        self.source_ids = np.memmap(
            self.root / f"{split}.source.u32", mode="r", dtype="<u4",
            shape=(count, condition_len))
        self.target_ids = np.memmap(
            self.root / f"{split}.target.u32", mode="r", dtype="<u4",
            shape=(count, target_len))
        self.source_lens = np.memmap(
            self.root / f"{split}.source_len.u16", mode="r", dtype="<u2",
            shape=(count,))
        self.target_lens = np.memmap(
            self.root / f"{split}.target_len.u16", mode="r", dtype="<u2",
            shape=(count,))

        refs_path = self.root / f"{split}.refs.jsonl"
        self.text_rows = None
        if refs_path.exists():
            with refs_path.open(encoding="utf-8") as handle:
                self.text_rows = [json.loads(line) for line in handle if line.strip()]
            if len(self.text_rows) < self.count:
                raise ValueError(f"{refs_path} has fewer rows than the binary split")

    def __len__(self):
        return self.count

    def rows(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        indices = np.asarray(indices, dtype=np.int64)
        source = torch.from_numpy(np.asarray(self.source_ids[indices], dtype=np.int64))
        target = torch.from_numpy(np.asarray(self.target_ids[indices], dtype=np.int64))
        source_len = torch.from_numpy(np.asarray(self.source_lens[indices], dtype=np.int64))
        target_len = torch.from_numpy(np.asarray(self.target_lens[indices], dtype=np.int64))
        spos = torch.arange(self.condition_len).unsqueeze(0)
        tpos = torch.arange(self.target_len).unsqueeze(0)
        # Sources are left-padded; targets are right-padded and include one supervised EOS.
        source_mask = spos >= (self.condition_len - source_len.unsqueeze(1))
        target_mask = tpos < target_len.unsqueeze(1)
        return source, source_mask, target, target_mask

    def sample(self, n: int, generator: Optional[torch.Generator] = None):
        idx = torch.randint(0, len(self), (n,), generator=generator)
        return (*self.rows(idx), idx)

    def texts(self, indices, tokenizer):
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().tolist()
        if self.text_rows is not None:
            sources = [self.text_rows[int(i)]["source"] for i in indices]
            refs = [self.text_rows[int(i)]["reference"] for i in indices]
            return sources, refs
        source, source_mask, target, target_mask = self.rows(indices)
        sources = [tokenizer.decode(row[mask].tolist(), skip_special_tokens=True)
                   for row, mask in zip(source, source_mask)]
        refs = [tokenizer.decode(row[mask].tolist(), skip_special_tokens=True)
                for row, mask in zip(target, target_mask)]
        return sources, refs


def build_dataset(cfg, tokenizer, split):
    limit = cfg.n_train if split == "train" else cfg.n_eval
    ds = Seq2SeqDataset(
        cfg.data_dir, split, tokenizer, cfg.dataset_name,
        cfg.query_len, cfg.resp_len, limit=limit)
    print(f"[data:{split}] N={len(ds):,} condition={cfg.query_len} target={cfg.resp_len} "
          f"source_truncated={ds.meta.get('source_truncated', 0):,} "
          f"target_truncated={ds.meta.get('target_truncated', 0):,}")
    return ds
