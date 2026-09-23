"""Convert common SVAMP/MAWPS-style records to candidate JSONL."""
import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


def rows_from(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("data", "questions", "items"):
            if isinstance(value.get(key), list):
                return value[key]
    raise ValueError("expected a JSON array or an object containing data/questions/items")


def convert(row):
    if "Body" in row:
        question = " ".join(str(row.get(key, "")).strip()
                            for key in ("Body", "Question") if row.get(key))
    else:
        question = row.get("question") or row.get("Question") or row.get("sQuestion")
    answer = row.get("answer", row.get("Answer"))
    if answer is None:
        for key in ("lSolutions", "solutions", "final_ans"):
            if row.get(key) is not None:
                answer = row[key]
                break
    if isinstance(answer, list):
        if len(answer) != 1:
            raise ValueError("expected exactly one final numeric answer")
        answer = answer[0]
    if not question or answer is None:
        raise ValueError(f"cannot find question/answer fields in {sorted(row)}")
    text = str(answer).strip()
    marker = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    number = marker.group(1) if marker else text
    try:
        value = Decimal(number.replace(",", "").replace("$", ""))
    except InvalidOperation as error:
        raise ValueError("expected a numeric answer or an explicit #### numeric marker") from error
    if not value.is_finite():
        raise ValueError("answer must be finite")
    # Evaluation uses final-answer supervision only; do not invent a solution.
    return {"question": str(question).strip(), "answer": f"#### {format(value, 'f')}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("output exists; use --overwrite to replace it")
    text = args.input.read_text(encoding="utf-8-sig")
    try:
        source = json.loads(text)
    except json.JSONDecodeError:
        source = [json.loads(line) for line in text.splitlines() if line.strip()]
    converted = [convert(row) for row in rows_from(source)]
    if not converted:
        raise ValueError("input dataset is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(converted)} records to {args.output}")


if __name__ == "__main__":
    main()
