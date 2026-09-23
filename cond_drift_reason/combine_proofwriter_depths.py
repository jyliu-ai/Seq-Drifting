"""Merge D0-D5 ProofWriter files while preserving valid proof candidates."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = p.parse_args()
    root = Path(args.data_dir)

    for split in args.splits:
        rows = []
        seen = set()
        labels = defaultdict(set)
        for depth in range(6):
            path = root / f"proofwriter_d{depth}_{split}.jsonl"
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    key = (row["question"], row["label"], row["answer"])
                    labels[row["question"]].add(row["label"])
                    if key not in seen:
                        seen.add(key)
                        rows.append(row)
        conflicts = {q: sorted(v) for q, v in labels.items() if len(v) > 1}
        if conflicts:
            raise ValueError(f"same prompt has conflicting labels in {split}: "
                             f"{list(conflicts.items())[:3]}")
        output = root / f"proofwriter_all_{split}.jsonl"
        candidate_total = Counter()
        for row in rows:
            candidate_total[(row["question"], row["label"])] += 1
        candidate_index = Counter()
        with output.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                row = dict(row)
                row["merged_index"] = index
                group = (row["question"], row["label"])
                row["candidate_index"] = candidate_index[group]
                row["n_candidates"] = candidate_total[group]
                candidate_index[group] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        groups = len(labels)
        candidate_counts = Counter()
        for row in rows:
            candidate_counts[(row["question"], row["label"])] += 1
        multi = sum(count > 1 for count in candidate_counts.values())
        print(json.dumps({
            "split": split,
            "rows": len(rows),
            "groups": groups,
            "multi_candidate_groups": multi,
            "max_candidates": max(candidate_counts.values(), default=0),
            "output": str(output),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
