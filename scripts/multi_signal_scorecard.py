#!/usr/bin/env python3
"""
Compute the multi-signal scorecard for variant submissions.

Signals:
  1. Raw val macro F1
  2. Bootstrap LB90 on val (lower 10th percentile of macro F1 over 2000 query-resampled bootstraps)
  3. Per-query F1 floor (worst single-query F1) and std
  4. Test output shape vs reference baseline (mean/median cite count, law/court mix)
  5. Test-prediction Jaccard overlap vs reference baseline (mean per-query Jaccard)

Usage:
  python scripts/multi_signal_scorecard.py \\
      --val-gold data/val.csv \\
      --reference-test submissions/test_submission_baseline_public_best_30257.csv \\
      --reference-val submissions/val_pred_baseline_public_best_30257.csv \\
      --variant NAME=submissions/val_pred_X.csv,submissions/test_submission_X.csv \\
      --variant NAME2=submissions/val_pred_Y.csv,submissions/test_submission_Y.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np


LAW_PATTERNS = (
    re.compile(r"^Art\. "),
    re.compile(r"^Art\.\s"),
)
COURT_PATTERNS = (
    re.compile(r"^BGE "),
    re.compile(r"^\d+[A-Z]_\d+/\d+"),
)


def is_law(citation: str) -> bool:
    return any(p.match(citation) for p in LAW_PATTERNS)


def is_court(citation: str) -> bool:
    return any(p.match(citation) for p in COURT_PATTERNS)


def load_gold(path: Path) -> dict[str, set[str]]:
    with path.open() as f:
        return {row["query_id"]: set(row["gold_citations"].split(";")) for row in csv.DictReader(f)}


def load_predictions(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["query_id"]
            cites = row.get("predicted_citations") or ""
            out[qid] = set(c for c in cites.split(";") if c)
    return out


def f1(pred: set[str], gold: set[str]) -> float:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def per_query_f1(predictions: dict[str, set[str]], gold: dict[str, set[str]]) -> dict[str, float]:
    return {qid: f1(predictions.get(qid, set()), gold[qid]) for qid in gold}


def bootstrap_lb(scores: list[float], n_iter: int = 2000, lb_pct: int = 10, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    arr = np.array(scores, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0
    means = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        means.append(arr[idx].mean())
    return float(np.percentile(means, lb_pct))


def shape_stats(predictions: dict[str, set[str]]) -> dict[str, float]:
    counts = [len(p) for p in predictions.values()]
    if not counts:
        return {"n_queries": 0, "mean_cites": 0.0, "median_cites": 0.0, "law_pct": 0.0, "court_pct": 0.0, "law_mean": 0.0, "court_mean": 0.0}
    law_counts = []
    court_counts = []
    other_counts = []
    for p in predictions.values():
        law_counts.append(sum(1 for c in p if is_law(c)))
        court_counts.append(sum(1 for c in p if is_court(c)))
        other_counts.append(sum(1 for c in p if not is_law(c) and not is_court(c)))
    total = sum(counts)
    return {
        "n_queries": len(counts),
        "mean_cites": mean(counts),
        "median_cites": median(counts),
        "law_pct": 100.0 * sum(law_counts) / total if total else 0.0,
        "court_pct": 100.0 * sum(court_counts) / total if total else 0.0,
        "law_mean": mean(law_counts),
        "court_mean": mean(court_counts),
        "other_mean": mean(other_counts),
    }


def baseline_overlap(variant: dict[str, set[str]], reference: dict[str, set[str]]) -> dict[str, float]:
    qids = sorted(set(variant) & set(reference))
    if not qids:
        return {"mean_jaccard": 0.0, "median_jaccard": 0.0, "min_jaccard": 0.0, "n_queries": 0}
    j = [jaccard(variant[qid], reference[qid]) for qid in qids]
    return {
        "mean_jaccard": mean(j),
        "median_jaccard": median(j),
        "min_jaccard": min(j),
        "n_queries": len(j),
    }


def evaluate_variant(
    name: str,
    val_pred_path: Path | None,
    test_pred_path: Path | None,
    val_gold: dict[str, set[str]],
    reference_val: dict[str, set[str]] | None,
    reference_test: dict[str, set[str]] | None,
) -> dict[str, object]:
    out: dict[str, object] = {"name": name}

    if val_pred_path is not None and val_pred_path.exists():
        val_pred = load_predictions(val_pred_path)
        per_q = per_query_f1(val_pred, val_gold)
        scores = [per_q[qid] for qid in sorted(val_gold)]
        out["val_macro_f1"] = mean(scores) if scores else 0.0
        out["val_lb90"] = bootstrap_lb(scores, n_iter=2000, lb_pct=10, seed=0)
        out["val_floor"] = min(scores) if scores else 0.0
        out["val_std"] = pstdev(scores) if scores else 0.0
        out["val_per_query"] = per_q
        if reference_val is not None:
            out["val_overlap_vs_ref"] = baseline_overlap(val_pred, reference_val)
    else:
        out["val_macro_f1"] = None

    if test_pred_path is not None and test_pred_path.exists():
        test_pred = load_predictions(test_pred_path)
        out["test_shape"] = shape_stats(test_pred)
        if reference_test is not None:
            out["test_overlap_vs_ref"] = baseline_overlap(test_pred, reference_test)
    else:
        out["test_shape"] = None

    return out


def fmt_pct(value: float | None) -> str:
    return "  --  " if value is None else f"{value*100:6.2f}"


def fmt_num(value: float | None, width: int = 6, decimals: int = 2) -> str:
    return " " * (width - 2) + "--" if value is None else f"{value:{width}.{decimals}f}"


def print_table(rows: list[dict[str, object]]) -> None:
    print()
    print("=== VAL SIGNALS (10 queries) ===")
    print(f"{'variant':45} {'F1':>7} {'LB90':>7} {'floor':>7} {'std':>7} {'over_ref':>9}")
    for row in rows:
        f1_v = row.get("val_macro_f1")
        lb = row.get("val_lb90")
        floor = row.get("val_floor")
        std = row.get("val_std")
        over_ref = row.get("val_overlap_vs_ref")
        over_str = "  --  " if over_ref is None else f"{over_ref['mean_jaccard']*100:7.2f}"
        print(
            f"{row['name'][:45]:45} "
            f"{fmt_pct(f1_v):>7} "
            f"{fmt_pct(lb):>7} "
            f"{fmt_pct(floor):>7} "
            f"{fmt_pct(std):>7} "
            f"{over_str:>9}"
        )
    print()
    print("=== TEST SHAPE & OVERLAP (40 queries) ===")
    print(
        f"{'variant':45} {'#cites':>7} {'med':>5} {'law%':>6} {'court%':>7} "
        f"{'mean_J':>7} {'min_J':>6}"
    )
    for row in rows:
        shape = row.get("test_shape")
        over = row.get("test_overlap_vs_ref")
        if shape is None:
            print(f"{row['name'][:45]:45} (no test predictions)")
            continue
        mean_j = "  --  " if over is None else f"{over['mean_jaccard']*100:7.2f}"
        min_j = "  --  " if over is None else f"{over['min_jaccard']*100:6.2f}"
        print(
            f"{row['name'][:45]:45} "
            f"{shape['mean_cites']:7.2f} "
            f"{shape['median_cites']:5.0f} "
            f"{shape['law_pct']:6.2f} "
            f"{shape['court_pct']:7.2f} "
            f"{mean_j:>7} "
            f"{min_j:>6}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-gold", type=Path, required=True)
    parser.add_argument("--reference-name", type=str, required=True)
    parser.add_argument("--reference-val", type=Path, required=True)
    parser.add_argument("--reference-test", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="NAME=val_pred.csv[,test_pred.csv]",
    )
    args = parser.parse_args()

    val_gold = load_gold(args.val_gold)
    reference_val = load_predictions(args.reference_val)
    reference_test = load_predictions(args.reference_test)

    rows: list[dict[str, object]] = []

    rows.append(
        evaluate_variant(
            f"{args.reference_name} (REFERENCE)",
            args.reference_val,
            args.reference_test,
            val_gold,
            reference_val=None,  # don't compare baseline to itself
            reference_test=None,
        )
    )

    for spec in args.variant:
        name, _, paths = spec.partition("=")
        parts = paths.split(",")
        val_path = Path(parts[0]) if parts and parts[0] else None
        test_path = Path(parts[1]) if len(parts) > 1 and parts[1] else None
        rows.append(
            evaluate_variant(
                name,
                val_path,
                test_path,
                val_gold,
                reference_val=reference_val,
                reference_test=reference_test,
            )
        )

    print_table(rows)


if __name__ == "__main__":
    main()
