#!/usr/bin/env python3
"""
Additively inject court citations using raw FAISS court_dense_rank.

Instead of training a classifier, this takes the winner baseline and adds
top-N court citations per query by raw FAISS similarity rank. The raw score
outperforms trained classifiers per CODEX_MEMORY findings.

Usage:
    .venv/bin/python scripts/overnight_faiss_inject.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
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


def load_judged_bundles(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def is_court(citation: str) -> bool:
    return bool(re.match(r"^BGE ", citation) or re.match(r"^\d+[A-Z]_\d+/\d+", citation))


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


def extract_court_candidates(bundles_data: dict) -> dict[str, list[dict]]:
    """Extract court candidates per query, sorted by court_dense_rank (ascending = best)."""
    result = {}
    for bundle in bundles_data.get("bundles", []):
        qid = bundle["query_id"]
        courts = []
        for c in bundle.get("candidates", []):
            rank = c.get("court_dense_rank")
            if c.get("kind") == "court" and rank is not None and rank < 9999:
                courts.append({
                    "citation": c["citation"],
                    "court_dense_rank": rank,
                    "raw_score": c.get("raw_score", 0.0),
                    "judge_label": c.get("judge_label", ""),
                    "judge_confidence": c.get("judge_confidence", 0.0),
                })
        courts.sort(key=lambda x: x["court_dense_rank"])
        result[qid] = courts
    return result


def inject_courts(
    baseline: dict[str, set[str]],
    court_candidates: dict[str, list[dict]],
    max_add: int,
    rank_threshold: int,
    require_judge_positive: bool = False,
) -> dict[str, set[str]]:
    """Add top-N court citations not already in baseline."""
    result = {}
    for qid, preds in baseline.items():
        new_preds = set(preds)
        candidates = court_candidates.get(qid, [])
        added = 0
        for c in candidates:
            if added >= max_add:
                break
            if c["court_dense_rank"] > rank_threshold:
                break
            if c["citation"] in new_preds:
                continue
            if require_judge_positive and c["judge_label"] == "reject":
                continue
            new_preds.add(c["citation"])
            added += 1
        result[qid] = new_preds
    return result


def write_csv(predictions: dict[str, set[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions):
            writer.writerow([qid, ";".join(sorted(predictions[qid]))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-val", type=Path,
                        default=BASE / "submissions/val_pred_baseline_public_best_30257.csv")
    parser.add_argument("--baseline-test", type=Path,
                        default=BASE / "submissions/test_submission_baseline_public_best_30257.csv")
    parser.add_argument("--val-judged", type=Path,
                        default=BASE / "artifacts/v11/val_v11_strict_v1/judged_bundles.json")
    parser.add_argument("--test-judged", type=Path,
                        default=BASE / "artifacts/v11/test_v11_strict_v1/judged_bundles.json")
    parser.add_argument("--gold-csv", type=Path, default=BASE / "data/val.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE / "submissions")
    args = parser.parse_args()

    gold = load_gold(args.gold_csv)
    baseline_val = load_predictions(args.baseline_val)
    baseline_test = load_predictions(args.baseline_test)

    print("Loading judged bundles for court candidates...", flush=True)
    val_bundles = load_judged_bundles(args.val_judged)
    test_bundles = load_judged_bundles(args.test_judged)
    val_courts = extract_court_candidates(val_bundles)
    test_courts = extract_court_candidates(test_bundles)

    baseline_f1 = macro_f1(baseline_val, gold)
    print(f"Baseline val F1: {baseline_f1:.6f}")

    # Grid search
    max_adds = [1, 2, 3, 4, 5]
    rank_thresholds = [5, 10, 20, 40, 80, 160]
    judge_filters = [False, True]  # require judge non-reject?

    results = []
    for max_add, rank_thresh, judge_filt in [(m, r, j)
            for m in max_adds for r in rank_thresholds for j in judge_filters]:
        injected = inject_courts(baseline_val, val_courts, max_add, rank_thresh, judge_filt)
        f1 = macro_f1(injected, gold)
        avg_preds = sum(len(p) for p in injected.values()) / len(injected)
        results.append({
            "max_add": max_add,
            "rank_threshold": rank_thresh,
            "judge_filter": judge_filt,
            "val_f1": f1,
            "delta_pp": (f1 - baseline_f1) * 100,
            "avg_preds": avg_preds,
        })

    results.sort(key=lambda r: r["val_f1"], reverse=True)

    print(f"\n{'='*70}")
    print(f"TOP 10 FAISS INJECTION CONFIGS")
    print(f"{'='*70}")
    for r in results[:10]:
        print(f"  val_F1={r['val_f1']:.6f} ({r['delta_pp']:+.2f}pp) "
              f"max_add={r['max_add']} rank_thresh={r['rank_threshold']} "
              f"judge_filt={r['judge_filter']} avg_preds={r['avg_preds']:.1f}")

    # Write top-5 val+test pairs
    for i, r in enumerate(results[:5], 1):
        tag = f"overnight_faiss_inject_top{i}"
        injected_val = inject_courts(baseline_val, val_courts,
                                     r["max_add"], r["rank_threshold"], r["judge_filter"])
        injected_test = inject_courts(baseline_test, test_courts,
                                      r["max_add"], r["rank_threshold"], r["judge_filter"])
        val_path = args.output_dir / f"val_pred_{tag}.csv"
        test_path = args.output_dir / f"test_submission_{tag}.csv"
        write_csv(injected_val, val_path)
        write_csv(injected_test, test_path)
        print(f"  Wrote {val_path.name} + {test_path.name}")


if __name__ == "__main__":
    main()
