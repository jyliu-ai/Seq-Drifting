"""Build a clean, leakage-safe single-binary-equation JSONL dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path


NUMBER = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
EQUATION_RE = re.compile(
    rf"^\s*({NUMBER})\s*([+\-*/])\s*({NUMBER})\s*=\s*({NUMBER})\s*$"
)
FINAL_RE = re.compile(rf"####\s*\$?\s*({NUMBER})")
BLOCK_RE = re.compile(r"<<([^<>]*)>>")


def normalize_question(value: object) -> str:
    return " ".join(str(value).split()).casefold()


def as_decimal(value: str) -> Decimal:
    return Decimal(value.replace(",", ""))


def equation_is_exact(match: re.Match[str]) -> bool:
    left = as_decimal(match.group(1))
    right = as_decimal(match.group(3))
    expected = as_decimal(match.group(4))
    op = match.group(2)
    with localcontext() as ctx:
        ctx.prec = 50
        if op == "+":
            actual = left + right
        elif op == "-":
            actual = left - right
        elif op == "*":
            actual = left * right
        else:
            if right == 0:
                return False
            actual = left / right
    return actual == expected


def select_clean_rows(path: Path) -> tuple[list[dict[str, str]], Counter[str]]:
    stats: Counter[str] = Counter()
    by_question: dict[str, dict[str, str]] = {}
    conflicts: set[str] = set()

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            stats["input"] += 1
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", row.get("generated", ""))).strip()
            if not question or not answer:
                stats["empty"] += 1
                continue

            blocks = BLOCK_RE.findall(answer)
            if len(blocks) != 1:
                stats["not_one_equation"] += 1
                continue
            equation = EQUATION_RE.fullmatch(blocks[0])
            if equation is None:
                stats["not_binary_numeric"] += 1
                continue
            final = FINAL_RE.search(answer)
            if final is None:
                stats["missing_numeric_final"] += 1
                continue
            try:
                if not equation_is_exact(equation):
                    stats["incorrect_equation"] += 1
                    continue
                if as_decimal(equation.group(4)) != as_decimal(final.group(1)):
                    stats["final_mismatch"] += 1
                    continue
            except (InvalidOperation, ZeroDivisionError):
                stats["invalid_number"] += 1
                continue

            key = normalize_question(question)
            cleaned = {"question": question, "answer": answer}
            previous = by_question.get(key)
            if previous is None:
                by_question[key] = cleaned
            elif previous["answer"] != answer:
                conflicts.add(key)
                stats["conflicting_duplicate"] += 1
            else:
                stats["exact_duplicate"] += 1

    for key in conflicts:
        by_question.pop(key, None)
    stats["clean_unique"] = len(by_question)
    return list(by_question.values()), stats


def is_test(question: str, test_fraction: float, seed: str) -> bool:
    digest = hashlib.blake2b(
        f"{seed}\0{normalize_question(question)}".encode("utf-8"), digest_size=8
    ).digest()
    value = int.from_bytes(digest, "big") / float(1 << 64)
    return value < test_fraction


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("train_output", type=Path)
    parser.add_argument("test_output", type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", default="single-equation-v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.test_fraction < 1.0:
        parser.error("--test-fraction must be between 0 and 1")
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    for output in (args.train_output, args.test_output):
        if output.exists() and not args.overwrite:
            parser.error(f"output exists (pass --overwrite): {output}")

    rows, stats = select_clean_rows(args.input)
    train_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    for row in rows:
        destination = test_rows if is_test(
            row["question"], args.test_fraction, args.seed
        ) else train_rows
        destination.append(row)

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.test_output, test_rows)

    print(f"input: {args.input}")
    for name in sorted(stats):
        print(f"{name}: {stats[name]:,}")
    print(f"train: {len(train_rows):,} -> {args.train_output}")
    print(f"test: {len(test_rows):,} -> {args.test_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
