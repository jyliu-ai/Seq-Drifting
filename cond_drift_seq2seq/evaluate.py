"""ELF-style task evaluation for WMT14 De-En and XSum."""

import math
import statistics
from collections import Counter

import torch


def _token_stats(id_lists):
    entropies = []
    all_ids = []
    for ids in id_lists:
        if not ids:
            continue
        all_ids.extend(ids)
        counts = Counter(ids)
        n = len(ids)
        entropies.append(-sum((count / n) * math.log(count / n)
                              for count in counts.values()))
    entropy = sum(entropies) / max(len(entropies), 1)
    unique = len(set(all_ids)) / max(len(all_ids), 1)
    return entropy, unique


def _task_metrics(predictions, references, dataset_name):
    if dataset_name == "wmt14_de_en":
        try:
            import sacrebleu
            score = sacrebleu.corpus_bleu(
                predictions, [references], lowercase=True,
                use_effective_order=True)
            out = {"bleu": float(score.score)}
            # bp/len_ratio/p1 separate the two ways this model loses BLEU: n-gram precision
            # and the brevity penalty. A non-autoregressive model that commits to EOS early
            # gives away several points to bp alone while its precision is unchanged, and the
            # aggregate score cannot tell those apart. Read defensively: these are internals
            # of sacrebleu's score object, and an eval that raises would kill the whole run.
            try:
                out["bp"] = float(score.bp)
                out["len_ratio"] = float(score.sys_len) / max(float(score.ref_len), 1.0)
                # All four orders, not just p1: degenerate repetition RAISES unigram
                # precision (every stray ' the' can still match one in the reference) while
                # collapsing p2-p4, so p1 alone moves the wrong way and hides the failure.
                for i, prec in enumerate(score.precisions[:4], start=1):
                    out[f"p{i}"] = float(prec)
            except (AttributeError, IndexError, TypeError):
                pass
            return out
        except ImportError:
            print("  [eval] BLEU skipped: install sacrebleu")
            return {}

    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("  [eval] ROUGE skipped: install rouge-score")
        return {}
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    values = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in values:
            values[key].append(100.0 * scores[key].fmeasure)
    result = {}
    for key, rows in values.items():
        n = len(rows)
        mean = sum(rows) / max(n, 1)
        std = statistics.pstdev(rows) if n > 1 else 0.0
        result[key] = mean
        result[f"{key}_sem"] = std / math.sqrt(n) if n > 1 else 0.0
    return result


def _generator_forward(gen, query_emb, query_mask, z, cfg, device):
    amp = bool(cfg.use_bf16 and device.type == "cuda")
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        return gen(query_emb, query_mask, z).float()


@torch.no_grad()
def run_eval(gen, embedder, dataset, cfg, tokenizer, device,
             n_queries, n_show=6):
    was_training = gen.training
    gen.eval()
    n = min(n_queries, len(dataset))
    all_indices = torch.arange(n)
    sources, references = dataset.texts(all_indices, tokenizer)
    predictions, id_lists = [], []
    eos_hits = 0
    batch_size = max(1, cfg.eval_batch_size)

    gold_len_total = 0
    for start in range(0, n, batch_size):
        indices = all_indices[start:start + batch_size]
        query_ids, query_mask, _, gold_mask = dataset.rows(indices)
        # Gold content length excludes the one supervised EOS, matching predictions (which
        # are truncated AT their EOS below). Needed to read BLEU's brevity penalty, which
        # a non-autoregressive model can lose several points to without any loss of n-gram
        # precision -- mean_len alone cannot tell those two apart.
        gold_len_total += int(gold_mask.sum()) - int(gold_mask.shape[0])
        query_ids = query_ids.to(device)
        query_mask = query_mask.to(device)
        query_emb = embedder.query_embeds(query_ids)
        z = gen.sample_z(query_ids.shape[0], cfg.noise_dim, cfg.temp, device)
        emb = _generator_forward(gen, query_emb, query_mask, z, cfg, device)
        if cfg.sphere_norm:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        tokens = embedder.decode(emb)
        for row in tokens:
            ids = row.tolist()
            eos = tokenizer.eos_token_id
            if eos in ids:
                eos_hits += 1
                ids = ids[:ids.index(eos)]
            id_lists.append(ids)
            predictions.append(tokenizer.decode(ids, skip_special_tokens=True))

    metrics = {"n_queries": float(n)}
    metrics.update(_task_metrics(predictions, references, cfg.dataset_name))
    entropy, unique = _token_stats(id_lists)
    metrics["entropy"] = entropy
    metrics["uniq_tok"] = unique
    metrics["eos_rate"] = eos_hits / max(n, 1)
    metrics["mean_len"] = sum(map(len, id_lists)) / max(n, 1)
    metrics["gold_len"] = gold_len_total / max(n, 1)

    def repeated_fourgram(ids):
        grams = [tuple(ids[i:i + 4]) for i in range(max(0, len(ids) - 3))]
        return len(grams) != len(set(grams))

    metrics["repeat_rate"] = sum(repeated_fourgram(ids) for ids in id_lists) / max(n, 1)
    metrics["q_distinct"] = len(set(predictions)) / max(n, 1)

    shown = []
    for i in range(min(n_show, n)):
        source = sources[i]
        if len(source) > 600:
            source = source[:600] + "..."
        shown.append(
            f"SOURCE: {source}\nREFERENCE: {references[i]}\nPREDICTION: {predictions[i]}")
    if was_training:
        gen.train()
    return metrics, shown
