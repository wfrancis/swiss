#!/usr/bin/env python3
"""
Report a small reliability scorecard for a candidate submission.

This is meant to complement raw val macro F1 with uncertainty and output-shape
checks so we do not promote a val-only overfit just because it wins on 10 rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--gold-csv", type=Path, default=Path("data/val.csv"))
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument(
        "--ref-test",
        action="append",
        default=[],
        help="Reference test submission in label=path form. May be repeated.",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def parse_citations(value: str) -> set[str]:
    if not value or not value.strip():
        return set()
    return {piece.strip() for piece in value.split(";") if piece.strip()}


def load_gold(path: Path) -> dict[str, set[str]]:
    with path.open() as f:
        return {
            row["query_id"]: parse_citations(row["gold_citations"])
            for row in csv.DictReader(f)
        }


def load_predictions(path: Path) -> dict[str, set[str]]:
    with path.open() as f:
        return {
            row["query_id"]: parse_citations(row["predicted_citations"])
            for row in csv.DictReader(f)
        }


def citation_f1(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def bootstrap_lower_bound(
    by_query_scores: dict[str, float],
    iterations: int,
    seed: int,
    alpha: float,
) -> float:
    rng = random.Random(seed)
    query_ids = list(by_query_scores)
    sampled_means = []
    for _ in range(iterations):
        sample = [by_query_scores[rng.choice(query_ids)] for _ in query_ids]
        sampled_means.append(sum(sample) / len(sample))
    sampled_means.sort()
    index = max(0, min(len(sampled_means) - 1, int(alpha * len(sampled_means))))
    return sampled_means[index]


def average_prediction_count(predictions: dict[str, set[str]]) -> float:
    counts = [len(citations) for citations in predictions.values()]
    return sum(counts) / len(counts) if counts else 0.0


def average_court_fraction(predictions: dict[str, set[str]]) -> float:
    fractions = []
    for citations in predictions.values():
        if not citations:
            fractions.append(0.0)
            continue
        courts = sum(1 for citation in citations if not citation.startswith("Art."))
        fractions.append(courts / len(citations))
    return sum(fractions) / len(fractions) if fractions else 0.0


def average_jaccard(
    left: dict[str, set[str]],
    right: dict[str, set[str]],
) -> float:
    scores = []
    for query_id, left_set in left.items():
        right_set = right.get(query_id, set())
        union = left_set | right_set
        scores.append(len(left_set & right_set) / len(union) if union else 1.0)
    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    args = parse_args()
    gold = load_gold(args.gold_csv)
    val_predictions = load_predictions(args.val_csv)

    by_query_scores = {
        query_id: citation_f1(val_predictions.get(query_id, set()), gold_citations)
        for query_id, gold_citations in gold.items()
    }
    macro_f1 = sum(by_query_scores.values()) / len(by_query_scores) if by_query_scores else 0.0
    lower_90 = bootstrap_lower_bound(by_query_scores, args.bootstrap, args.seed, 0.10)
    lower_95 = bootstrap_lower_bound(by_query_scores, args.bootstrap, args.seed, 0.05)

    report: dict[str, object] = {
        "val_macro_f1": macro_f1,
        "val_bootstrap_lb90": lower_90,
        "val_bootstrap_lb95": lower_95,
        "val_query_std": statistics.pstdev(by_query_scores.values()) if by_query_scores else 0.0,
        "val_query_min": min(by_query_scores.values()) if by_query_scores else 0.0,
        "val_avg_predictions": average_prediction_count(val_predictions),
        "val_avg_court_fraction": average_court_fraction(val_predictions),
        "val_by_query": by_query_scores,
    }

    if args.test_csv:
        test_predictions = load_predictions(args.test_csv)
        report["test_avg_predictions"] = average_prediction_count(test_predictions)
        report["test_avg_court_fraction"] = average_court_fraction(test_predictions)
        ref_scores = {}
        for item in args.ref_test:
            label, raw_path = item.split("=", 1)
            ref_scores[label] = average_jaccard(test_predictions, load_predictions(Path(raw_path)))
        report["test_reference_jaccard"] = ref_scores

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"val_macro_f1={report['val_macro_f1']:.6f}")
    print(f"val_bootstrap_lb90={report['val_bootstrap_lb90']:.6f}")
    print(f"val_bootstrap_lb95={report['val_bootstrap_lb95']:.6f}")
    print(f"val_query_std={report['val_query_std']:.6f}")
    print(f"val_query_min={report['val_query_min']:.6f}")
    print(f"val_avg_predictions={report['val_avg_predictions']:.2f}")
    print(f"val_avg_court_fraction={report['val_avg_court_fraction']:.4f}")
    if args.test_csv:
        print(f"test_avg_predictions={report['test_avg_predictions']:.2f}")
        print(f"test_avg_court_fraction={report['test_avg_court_fraction']:.4f}")
        for label, score in sorted(report["test_reference_jaccard"].items()):
            print(f"test_jaccard_{label}={score:.6f}")
    print("val_by_query:")
    for query_id, score in sorted(by_query_scores.items()):
        print(f"  {query_id}: {score:.6f}")


if __name__ == "__main__":
    main()
