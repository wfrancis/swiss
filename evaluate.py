"""
Evaluation script for Swiss Legal Citation Retrieval.
Computes citation-level Macro F1 score.
"""
import csv
import sys
from typing import Dict, List, Set


def parse_citations(citation_str: str) -> Set[str]:
    """Parse semicolon-separated citation string into a set."""
    if not citation_str or citation_str.strip() == "":
        return set()
    return {c.strip() for c in citation_str.split(";") if c.strip()}


def citation_f1(predicted: Set[str], gold: Set[str]) -> float:
    """Compute F1 score for a single query's citations."""
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(submission_path: str, gold_path: str) -> float:
    """Compute macro-averaged F1 across all queries."""
    # Load gold
    gold = {}
    with open(gold_path, "r") as f:
        for row in csv.DictReader(f):
            gold[row["query_id"]] = parse_citations(row["gold_citations"])

    # Load predictions
    preds = {}
    with open(submission_path, "r") as f:
        for row in csv.DictReader(f):
            preds[row["query_id"]] = parse_citations(row["predicted_citations"])

    # Compute per-query F1
    f1_scores = []
    for qid in gold:
        pred_set = preds.get(qid, set())
        gold_set = gold[qid]
        f1 = citation_f1(pred_set, gold_set)
        f1_scores.append(f1)
        print(f"  {qid}: F1={f1:.4f} (pred={len(pred_set)}, gold={len(gold_set)}, "
              f"tp={len(pred_set & gold_set)})")

    macro = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    print(f"\nMacro F1: {macro:.4f}")
    return macro


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <submission.csv> <gold.csv>")
        sys.exit(1)
    macro_f1(sys.argv[1], sys.argv[2])
