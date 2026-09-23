"""Build clean, leakage-safe 2-step and 3-step equation curricula.

Each retained solution has exactly N valid binary arithmetic annotations.  The
result of every step must be used as an operand by the next step, and the last
result must equal the final ``####`` answer.  Conflicting duplicate questions
and questions occurring at more than one retained step count are removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from decimal import InvalidOperation
from pathlib import Path

from .filter_single_equation import (
    BLOCK_RE,
    EQUATION_RE,
    FINAL_RE,
    as_decimal,
    equation_is_exact,
    normalize_question,
    write_jsonl,
)


def stable_test_split(question: str, fraction: float, seed: str) -> bool:
    digest = hashlib.blake2b(
        f"{seed}\0{normalize_question(question)}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") / float(1 << 64) < fraction


def clean_target(blocks: list[str], final: str) -> str:
    equations = "\n".join(f"<<{block.strip()}>>" for block in blocks)
    return f"{equations}\n#### {final}"


def parse_chain(answer: str, steps: int, stats: Counter[str]) -> str | None:
    blocks = BLOCK_RE.findall(answer)
    if len(blocks) != steps:
        return None
    matches = [EQUATION_RE.fullmatch(block) for block in blocks]
    if not all(matches):
        stats[f"step_{steps}_nonbinary"] += 1
        return None
    final = FINAL_RE.search(answer)
    if final is None:
        stats[f"step_{steps}_missing_final"] += 1
        return None
    try:
        if not all(equation_is_exact(match) for match in matches):
            stats[f"step_{steps}_incorrect_equation"] += 1
            return None
        if as_decimal(matches[-1].group(4)) != as_decimal(final.group(1)):
            stats[f"step_{steps}_final_mismatch"] += 1
            return None
        previous = as_decimal(matches[0].group(4))
        for match in matches[1:]:
            operands = {as_decimal(match.group(1)), as_decimal(match.group(3))}
            if previous not in operands:
                stats[f"step_{steps}_not_chain"] += 1
                return None
            previous = as_decimal(match.group(4))
    except (InvalidOperation, ZeroDivisionError):
        stats[f"step_{steps}_invalid_number"] += 1
        return None
    return clean_target(blocks, final.group(1))


def collect(path: Path, step_counts: tuple[int, ...]):
    stats: Counter[str] = Counter()
    candidates: dict[int, dict[str, dict[str, str]]] = {
        step: {} for step in step_counts
    }
    conflicts: dict[int, set[str]] = defaultdict(set)

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            stats["input"] += 1
            source = json.loads(line)
            question = str(source.get("question", "")).strip()
            answer = str(source.get("answer", source.get("generated", ""))).strip()
            if not question or not answer:
                stats["empty"] += 1
                continue
            step = len(BLOCK_RE.findall(answer))
            if step not in candidates:
                continue
            target = parse_chain(answer, step, stats)
            if target is None:
                continue
            key = normalize_question(question)
            row = {"question": question, "answer": target}
            previous = candidates[step].get(key)
            if previous is None:
                candidates[step][key] = row
            elif previous["answer"] == target:
                stats[f"step_{step}_exact_duplicate"] += 1
            else:
                conflicts[step].add(key)
                stats[f"step_{step}_conflicting_duplicate"] += 1

    for step, keys in conflicts.items():
        for key in keys:
            candidates[step].pop(key, None)

    owners: dict[str, set[int]] = defaultdict(set)
    for step, rows in candidates.items():
        for key in rows:
            owners[key].add(step)
    cross_step = {key for key, steps in owners.items() if len(steps) > 1}
    for step, rows in candidates.items():
        removed = 0
        for key in cross_step:
            removed += rows.pop(key, None) is not None
        stats[f"step_{step}_cross_step_removed"] = removed
        stats[f"step_{step}_clean_unique"] = len(rows)
    return candidates, stats


def filter_token_length(
    rows, tokenizer, query_len: int, resp_len: int, batch_size: int = 1024
):
    kept: list[dict[str, str]] = []
    target_removed = 0
    query_removed = 0
    max_target_tokens = 0
    max_query_tokens = 0
    values = list(rows.values())
    for start in range(0, len(values), batch_size):
        batch = values[start:start + batch_size]
        target_ids = tokenizer(
            [row["answer"] for row in batch], add_special_tokens=False
        )["input_ids"]
        prompts = [f"Question: {row['question']}\nAnswer:" for row in batch]
        query_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        for row, target, query in zip(batch, target_ids, query_ids):
            target_needed = len(target) + 1  # supervised EOS
            max_target_tokens = max(max_target_tokens, target_needed)
            max_query_tokens = max(max_query_tokens, len(query))
            if target_needed > resp_len:
                target_removed += 1
            elif len(query) > query_len:
                query_removed += 1
            else:
                kept.append(row)
    return kept, target_removed, query_removed, max_target_tokens, max_query_tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", default="equation-chain-curriculum-v1")
    parser.add_argument(
        "--tokenizer",
        help="optional tokenizer path/name used to reject over-length examples",
    )
    parser.add_argument("--query-len", type=int, default=128)
    parser.add_argument("--resp-len", type=int, default=48)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    step_counts = tuple(sorted(set(args.steps)))
    if not step_counts or any(step < 1 for step in step_counts):
        parser.error("--steps must contain positive integers")
    if not 0.0 < args.test_fraction < 1.0:
        parser.error("--test-fraction must be between 0 and 1")
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")

    output_paths = {
        step: (
            args.output_dir / f"gsm8k_chain_{step}step_train.jsonl",
            args.output_dir / f"gsm8k_chain_{step}step_test.jsonl",
        )
        for step in step_counts
    }
    for paths in output_paths.values():
        for path in paths:
            if path.exists() and not args.overwrite:
                parser.error(f"output exists (pass --overwrite): {path}")

    candidates, stats = collect(args.input, step_counts)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True
        )
    print(f"input: {args.input}")
    for name in sorted(stats):
        print(f"{name}: {stats[name]:,}")

    for step in step_counts:
        train: list[dict[str, str]] = []
        test: list[dict[str, str]] = []
        rows = list(candidates[step].values())
        if tokenizer is not None:
            (rows, target_removed, query_removed, max_target_tokens,
             max_query_tokens) = filter_token_length(
                candidates[step], tokenizer, args.query_len, args.resp_len
            )
            print(
                f"{step}-step token filter: target_removed={target_removed:,} "
                f"query_removed={query_removed:,} "
                f"target_max_with_eos={max_target_tokens} "
                f"query_max={max_query_tokens} "
                f"limits={args.query_len}/{args.resp_len}"
            )
        for row in rows:
            destination = test if stable_test_split(
                row["question"], args.test_fraction, args.seed
            ) else train
            destination.append(row)
        train_path, test_path = output_paths[step]
        write_jsonl(train_path, train)
        write_jsonl(test_path, test)
        print(f"{step}-step train: {len(train):,} -> {train_path}")
        print(f"{step}-step test:  {len(test):,} -> {test_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
