"""Convert ProofWriter OWA source parquet into proof-generation JSONL.

The OWA-depth-5 source contains questions at every actual proof depth.  This
script emits separate exact-depth files, retaining all shortest proof variants
for each query (up to --max-candidates).  False queries use a real proof of the
opposite statement.  Unknown queries are written separately because no proof
exists for either polarity.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import pyarrow.parquet as pq


SIGN_RE = re.compile(r'"([+-])"\)$')
TRIPLE_RE = re.compile(r'\b(triple\d+)\b')
RULE_FOR_INT_RE = re.compile(r'\((rule\d+)\s+%\s+(int\d+)\)')
INT_RE = re.compile(r'int(\d+)$')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-candidates", type=int, default=8,
                   help="maximum shortest proofs per query; 0 keeps all")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    return p.parse_args()


def decode_json(value):
    if value is None:
        return {}
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def label_of(answer):
    if answer is True:
        return "true"
    if answer is False:
        return "false"
    value = str(answer).strip().lower()
    if value not in {"true", "false", "unknown"}:
        raise ValueError(f"unexpected ProofWriter answer: {answer!r}")
    return value


def flip_representation(rep):
    match = SIGN_RE.search(rep)
    if not match:
        raise ValueError(f"cannot find polarity in representation: {rep!r}")
    flipped = "-" if match.group(1) == "+" else "+"
    return rep[:match.start(1)] + flipped + rep[match.end(1):]


def unique_in_order(items):
    return list(dict.fromkeys(items))


def proof_target(proof, detail, triples, rules, label):
    representation = proof.get("representation", "")
    intermediates = decode_json(proof.get("intermediates"))
    triple_ids = unique_in_order(TRIPLE_RE.findall(representation))
    rule_for_int = {int_id: rule_id
                    for rule_id, int_id in RULE_FOR_INT_RE.findall(representation)}

    lines = []
    for triple_id in triple_ids:
        item = triples.get(triple_id)
        if item and item.get("text"):
            lines.append(f"Fact: {item['text'].strip()}")

    ordered_ints = sorted(
        intermediates,
        key=lambda key: int(INT_RE.search(key).group(1)) if INT_RE.search(key) else -1,
        reverse=True,
    )
    for int_id in ordered_ints:
        rule_id = rule_for_int.get(int_id)
        rule = rules.get(rule_id, {}) if rule_id else {}
        if rule.get("text"):
            lines.append(f"Rule: {rule['text'].strip()}")
        text = intermediates[int_id].get("text", "").strip()
        if text:
            lines.append(f"Therefore: {text}")

    conclusion = detail.get("text", "").strip()
    if not lines and conclusion:
        lines.append(f"Fact: {conclusion}")
    elif conclusion and not any(
            line == f"Therefore: {conclusion}" or line == f"Fact: {conclusion}"
            for line in lines):
        lines.append(f"Therefore: {conclusion}")
    lines.append(f"#### {label}")
    return "\n".join(lines), len(intermediates)


def make_prompt(theory, question):
    return (f"Theory:\n{theory.strip()}\n\nQuery: {question.strip()}\n"
            "Give the proof in order and finish with #### true or #### false.")


def iter_parquet(path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=32):
        yield from batch.to_pylist()


def convert_split(path, output_dir, split, max_candidates):
    handles = {depth: (output_dir / f"proofwriter_d{depth}_{split}.jsonl").open(
        "w", encoding="utf-8") for depth in range(6)}
    unknown_handle = (output_dir / f"proofwriter_unknown_{split}.jsonl").open(
        "w", encoding="utf-8")
    stats = Counter()
    seen_group_keys = set()

    try:
        for world in iter_parquet(path):
            triples = decode_json(world["triples"])
            rules = decode_json(world["rules"])
            questions = decode_json(world["questions"])
            details = world.get("proofDetails") or []
            by_rep = {detail["representation"]: detail for detail in details}

            for query_id, question in questions.items():
                label = label_of(question["answer"])
                depth = int(question.get("QDep", -1))
                prompt = make_prompt(world["theory"], question["question"])
                group_key = (prompt, label)
                if group_key in seen_group_keys:
                    raise ValueError(f"duplicate query/label across worlds: {group_key!r}")
                seen_group_keys.add(group_key)
                stats[f"queries_{label}"] += 1
                stats[f"queries_{label}_d{depth}"] += 1

                base = {
                    "question": prompt,
                    "world_id": world["id"],
                    "query_id": query_id,
                    "depth": depth,
                    "label": label,
                }
                if label == "unknown":
                    row = dict(base, answer="#### unknown")
                    unknown_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats["rows_unknown"] += 1
                    continue

                target_rep = question["representation"]
                if label == "false":
                    target_rep = flip_representation(target_rep)
                detail = by_rep.get(target_rep)
                if detail is None:
                    raise ValueError(
                        f"missing proof detail for {world['id']} {query_id}: {target_rep}")
                candidates = []
                for proof in detail.get("proofsWithIntermediates") or []:
                    target, n_intermediates = proof_target(
                        proof, detail, triples, rules, label)
                    candidates.append((n_intermediates, len(target), target))
                if not candidates:
                    raise ValueError(f"no proofs for {world['id']} {query_id}")

                shortest_count = min(item[0] for item in candidates)
                shortest = sorted({item[2] for item in candidates
                                   if item[0] == shortest_count}, key=lambda x: (len(x), x))
                if max_candidates > 0:
                    shortest = shortest[:max_candidates]
                if depth not in handles:
                    raise ValueError(f"proved query has unexpected depth {depth}")
                for candidate_index, target in enumerate(shortest):
                    row = dict(base, answer=target, candidate_index=candidate_index,
                               n_candidates=len(shortest))
                    handles[depth].write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats[f"rows_{label}_d{depth}"] += 1
                stats[f"candidate_groups_d{depth}"] += 1
                stats[f"candidates_d{depth}"] += len(shortest)
    finally:
        for handle in handles.values():
            handle.close()
        unknown_handle.close()
    return dict(sorted(stats.items()))


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    split_prompts = {}
    for split in args.splits:
        path = input_dir / f"owa_depth5_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        all_stats[split] = convert_split(
            path, output_dir, split, args.max_candidates)
        split_prompts[split] = set()
        for depth in range(6):
            out = output_dir / f"proofwriter_d{depth}_{split}.jsonl"
            with out.open(encoding="utf-8") as f:
                for line in f:
                    split_prompts[split].add(json.loads(line)["question"])

    overlaps = {}
    for i, left in enumerate(args.splits):
        for right in args.splits[i + 1:]:
            overlaps[f"{left}_{right}"] = len(
                split_prompts[left] & split_prompts[right])
    all_stats["prompt_overlap"] = overlaps
    (output_dir / "stats.json").write_text(
        json.dumps(all_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(all_stats, indent=2, ensure_ascii=False))
    if any(overlaps.values()):
        raise ValueError(f"train/dev/test prompt overlap detected: {overlaps}")


if __name__ == "__main__":
    main()
