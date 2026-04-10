#!/usr/bin/env python3
"""
Majority-vote ensemble across all existing test submissions.

For each query, count how many CSVs include each citation. Include citations
that appear in at least K of N variants. Sweep K to find best on val.

Usage:
    .venv/bin/python scripts/overnight_ensemble_vote.py
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def load_predictions(path: Path) -> dict[str, set[str]]:
    with open(path) as f:
        reader = csv.DictReader(f)
        key = "predicted_citations" if "predicted_citations" in reader.fieldnames else "citations"
        return {row["query_id"]: set(c.strip() for c in row[key].split(";") if c.strip()) for row in reader}


def load_gold(path: Path) -> dict[str, set[str]]:
    with open(path) as f:
        return {row["query_id"]: set(row["gold_citations"].split(";")) for row in csv.DictReader(f)}


def macro_f1(predictions: dict[str, set[str]], gold: dict[str, set[str]]) -> float:
    f1s = []
    for qid in gold:
        pred = predictions.get(qid, set())
        g = gold[qid]
        if not pred and not g:
            f1s.append(1.0)
            continue
        if not pred or not g:
            f1s.append(0.0)
            continue
        tp = len(pred & g)
        p = tp / len(pred)
        r = tp / len(g)
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def find_csv_pairs(submissions_dir: Path) -> list[tuple[Path, Path]]:
    """Find test/val CSV pairs. Returns (test_csv, val_csv) tuples."""
    pairs = []
    for test_csv in sorted(submissions_dir.glob("test_submission_*.csv")):
        # Derive val name: test_submission_X.csv -> val_pred_X.csv
        suffix = test_csv.name.replace("test_submission_", "")
        val_csv = submissions_dir / f"val_pred_{suffix}"
        if val_csv.exists():
            pairs.append((test_csv, val_csv))
    return pairs


def build_vote_map(predictions_list: list[dict[str, set[str]]]) -> dict[str, Counter]:
    """For each query, count how many times each citation appears across all variants."""
    vote_map: dict[str, Counter] = {}
    for preds in predictions_list:
        for qid, cites in preds.items():
            if qid not in vote_map:
                vote_map[qid] = Counter()
            for cite in cites:
                vote_map[qid][cite] += 1
    return vote_map


def threshold_vote(vote_map: dict[str, Counter], min_votes: int) -> dict[str, set[str]]:
    result = {}
    for qid, counter in vote_map.items():
        result[qid] = {cite for cite, count in counter.items() if count >= min_votes}
    return result


def write_csv_out(predictions: dict[str, set[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions):
            writer.writerow([qid, ";".join(sorted(predictions[qid]))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submissions-dir", type=Path, default=BASE / "submissions")
    parser.add_argument("--gold-csv", type=Path, default=BASE / "data/val.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE / "submissions")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    gold = load_gold(args.gold_csv)
    pairs = find_csv_pairs(args.submissions_dir)
    print(f"Found {len(pairs)} test/val CSV pairs", flush=True)

    if len(pairs) < 3:
        print("Too few pairs to ensemble. Exiting.")
        return

    # Load all val and test predictions
    val_preds_list = []
    test_preds_list = []
    names = []
    for test_csv, val_csv in pairs:
        try:
            val_p = load_predictions(val_csv)
            test_p = load_predictions(test_csv)
            if val_p and test_p:
                val_preds_list.append(val_p)
                test_preds_list.append(test_p)
                names.append(test_csv.stem)
        except Exception as e:
            print(f"  Skipping {test_csv.name}: {e}")

    N = len(val_preds_list)
    print(f"Loaded {N} valid pairs for ensembling", flush=True)

    val_vote_map = build_vote_map(val_preds_list)
    test_vote_map = build_vote_map(test_preds_list)

    # Sweep threshold K
    results = []
    for K in range(1, max(N // 2 + 1, 4)):
        val_ensemble = threshold_vote(val_vote_map, K)
        f1 = macro_f1(val_ensemble, gold)
        avg_preds = sum(len(p) for p in val_ensemble.values()) / len(val_ensemble) if val_ensemble else 0
        results.append({"K": K, "val_f1": f1, "avg_preds": avg_preds})

    results.sort(key=lambda r: r["val_f1"], reverse=True)

    print(f"\n{'='*70}")
    print(f"ENSEMBLE VOTE RESULTS (N={N} variants)")
    print(f"{'='*70}")
    baseline_f1 = 0.282430
    for r in results:
        delta = (r["val_f1"] - baseline_f1) * 100
        print(f"  K={r['K']:2d}: val_F1={r['val_f1']:.6f} ({delta:+.2f}pp) avg_preds={r['avg_preds']:.1f}")

    # Write top-K pairs
    for i, r in enumerate(results[:args.top_k], 1):
        tag = f"overnight_ensemble_K{r['K']}_top{i}"
        val_ensemble = threshold_vote(val_vote_map, r["K"])
        test_ensemble = threshold_vote(test_vote_map, r["K"])
        val_path = args.output_dir / f"val_pred_{tag}.csv"
        test_path = args.output_dir / f"test_submission_{tag}.csv"
        write_csv_out(val_ensemble, val_path)
        write_csv_out(test_ensemble, test_path)
        print(f"  Wrote {val_path.name} + {test_path.name}")


if __name__ == "__main__":
    main()
