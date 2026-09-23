"""Convert a SVAMP JSON array to candidate-attraction evaluation JSONL."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path


OPS = {
    ast.Add: ("+", lambda a, b: a + b),
    ast.Sub: ("-", lambda a, b: a - b),
    ast.Mult: ("*", lambda a, b: a * b),
    ast.Div: ("/", lambda a, b: a / b),
}


def format_number(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def render_and_evaluate(node: ast.AST) -> tuple[str, Decimal, int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        value = Decimal(str(node.value))
        return format_number(value), value, 0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        text, value, count = render_and_evaluate(node.operand)
        if isinstance(node.op, ast.USub):
            return f"-{text}", -value, count
        return text, value, count
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left_text, left, left_count = render_and_evaluate(node.left)
        right_text, right, right_count = render_and_evaluate(node.right)
        symbol, operation = OPS[type(node.op)]
        with localcontext() as context:
            context.prec = 50
            value = operation(left, right)
        text = f"({left_text}{symbol}{right_text})"
        return text, value, left_count + right_count + 1
    raise ValueError(f"unsupported equation node: {ast.dump(node)}")


def convert(path: Path):
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError("SVAMP input must be a JSON array")

    full: list[dict[str, str]] = []
    single: list[dict[str, str]] = []
    two_step: list[dict[str, str]] = []
    stats: Counter[str] = Counter()
    for item in source:
        stats["input"] += 1
        body = str(item.get("Body", "")).strip()
        question_tail = str(item.get("Question", "")).strip()
        question = " ".join(part for part in (body, question_tail) if part)
        equation = str(item.get("Equation", "")).strip()
        if not question or not equation or "Answer" not in item:
            stats["missing_field"] += 1
            continue

        tree = ast.parse(equation, mode="eval")
        expression, result, operation_count = render_and_evaluate(tree.body)
        expected = Decimal(str(item["Answer"]))
        if result != expected:
            stats["answer_mismatch"] += 1
            continue

        if expression.startswith("(") and expression.endswith(")"):
            expression = expression[1:-1]
        answer = format_number(expected)
        row = {
            "question": question,
            "answer": f"<<{expression}={answer}>>\n#### {answer}",
        }
        full.append(row)
        stats[f"operations_{operation_count}"] += 1
        if operation_count == 1:
            single.append(row)
        elif operation_count == 2:
            two_step.append(row)

    stats["full_output"] = len(full)
    stats["single_output"] = len(single)
    stats["two_step_output"] = len(two_step)
    return full, single, two_step, stats


def write_jsonl(path: Path, rows: list[dict[str, str]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("full_output", type=Path)
    parser.add_argument("single_output", type=Path)
    parser.add_argument("--two-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    full, single, two_step, stats = convert(args.input)
    write_jsonl(args.full_output, full, args.overwrite)
    write_jsonl(args.single_output, single, args.overwrite)
    if args.two_output:
        write_jsonl(args.two_output, two_step, args.overwrite)
    for name in sorted(stats):
        print(f"{name}: {stats[name]:,}")
    print(f"full: {args.full_output}")
    print(f"single: {args.single_output}")
    if args.two_output:
        print(f"two-step: {args.two_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
