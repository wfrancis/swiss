#!/usr/bin/env python3
"""
Diagnostic: treat court-dense classifier as an ADDITIVE recall booster
on top of the 0.30257 baseline.

For each query:
  1. Start with the baseline's prediction set
  2. Take the court-dense classifier's top-ranked candidates that are
     NOT already in the baseline ("additions")
  3. Add the top-N additions to the baseline
  4. Compute val F1 of the combined set

Sweep N from 0 to 20 and report:
  - precision of the additions (fraction of additions that are gold)
  - new macro F1
  - delta vs baseline
  - test shape vs baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from run_v11_meta_selector import (  # noqa: E402
    candidate_features,
    collect_source_names,
    fit_model,
    group_rows_by_query,
    load_bundles,
    load_gold,
    predict_rows,
)


def build_rows(bundles, gold_map, source_names):
    """Same as run_v11_meta_selector.build_rows but also keeps raw_score, final_score, judge_label."""
    rows = []
    for bundle in bundles:
        gold = gold_map.get(bundle["query_id"], set()) if gold_map is not None else set()
        for c in bundle["candidates"]:
            rows.append(
                {
                    "query_id": bundle["query_id"],
                    "citation": c["citation"],
                    "kind": c["kind"],
                    "estimated_count": bundle["estimated_count"],
                    "features": candidate_features(c, bundle["estimated_count"], source_names),
                    "label": int(c["citation"] in gold) if gold_map is not None else 0,
                    "raw_score": float(c.get("raw_score", 0.0)),
                    "final_score": float(c.get("final_score", 0.0)),
                    "judge_label": c.get("judge_label", "reject"),
                }
            )
    return rows


def f1(pred: set[str], gold: set[str]) -> float:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def load_predictions(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["query_id"]] = set(c for c in (row.get("predicted_citations") or "").split(";") if c)
    return out


def write_predictions(path: Path, preds: dict[str, set[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(["query_id", "predicted_citations"])
        for qid in sorted(preds):
            w.writerow([qid, ";".join(sorted(preds[qid]))])


def additions_for_query(
    rows: list[dict],
    baseline_set: set[str],
    top_n: int,
    sort_key: str = "prob",
) -> list[dict]:
    """Return the top_n picks NOT already in baseline_set, ordered by sort_key descending.

    sort_key is one of "prob" (classifier probability) or "raw_score" or "final_score".
    """
    if top_n <= 0:
        return []
    ranked = sorted(rows, key=lambda r: (-r[sort_key], r["citation"]))
    out: list[dict] = []
    for r in ranked:
        if r["citation"] in baseline_set:
            continue
        out.append(r)
        if len(out) >= top_n:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-judged", type=Path, required=True)
    parser.add_argument("--val-judged", type=Path, required=True)
    parser.add_argument("--test-judged", type=Path, required=True)
    parser.add_argument("--train-gold", type=Path, required=True)
    parser.add_argument("--val-gold", type=Path, required=True)
    parser.add_argument("--baseline-val", type=Path, required=True)
    parser.add_argument("--baseline-test", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=None,
                        help="If set, writes blended val/test CSVs at the best N for promotion")
    parser.add_argument("--n-min", type=int, default=0)
    parser.add_argument("--n-max", type=int, default=20)
    args = parser.parse_args()

    train_bundles = load_bundles([args.train_judged])
    val_bundles = load_bundles([args.val_judged])
    test_bundles = load_bundles([args.test_judged])
    train_gold = load_gold(args.train_gold)
    val_gold = load_gold(args.val_gold)
    baseline_val = load_predictions(args.baseline_val)
    baseline_test = load_predictions(args.baseline_test)
    source_names = collect_source_names(train_bundles, val_bundles, test_bundles)

    print(f"Train: {len(train_bundles)} q   Val: {len(val_bundles)} q   Test: {len(test_bundles)} q", flush=True)
    print("Fitting court-dense classifier on train...", flush=True)
    train_rows = build_rows(train_bundles, train_gold, source_names)
    model = fit_model(train_rows, seed=0)

    val_rows = build_rows(val_bundles, val_gold, source_names)
    test_rows = build_rows(test_bundles, None, source_names)
    predict_rows(model, val_rows)
    predict_rows(model, test_rows)
    val_grouped = group_rows_by_query(val_rows)
    test_grouped = group_rows_by_query(test_rows)

    # Baseline F1 baseline
    baseline_f1 = mean(f1(baseline_val[qid], val_gold[qid]) for qid in val_gold)
    print(f"\nBaseline val F1: {baseline_f1*100:.2f}%", flush=True)

    def sweep(sort_key: str, label: str):
        print(f"\n=== ADDITIVE BLEND on val (rank by {label}) ===")
        print(f"{'N':>3} {'val F1':>8} {'delta':>7} {'add prec':>10} {'gold added':>11} {'mean cites':>11}")
        results = []
        for n in range(args.n_min, args.n_max + 1):
            scores = []
            added_total = 0
            added_correct = 0
            sizes = []
            for qid, qrows in val_grouped.items():
                base_set = set(baseline_val.get(qid, set()))
                additions = additions_for_query(qrows, base_set, n, sort_key=sort_key)
                add_set = {r["citation"] for r in additions}
                blended = base_set | add_set
                sizes.append(len(blended))
                scores.append(f1(blended, val_gold[qid]))
                added_total += len(add_set)
                added_correct += len(add_set & val_gold[qid])
            macro = mean(scores)
            prec = added_correct / added_total if added_total else 0.0
            gold_added_per_query = added_correct / len(val_grouped)
            results.append((n, macro, prec, gold_added_per_query, mean(sizes)))
            print(
                f"{n:>3} "
                f"{macro*100:7.2f}% "
                f"{(macro - baseline_f1)*100:+6.2f} "
                f"{prec*100:9.2f}% "
                f"{gold_added_per_query:11.3f} "
                f"{mean(sizes):11.2f}"
            )
        return results

    val_results_prob = sweep("prob", "classifier prob")
    val_results_raw = sweep("raw_score", "raw_score")
    val_results_final = sweep("final_score", "final_score")

    val_results = val_results_prob
    best_n, best_f1, *_ = max(val_results, key=lambda r: r[1])
    print(f"\nBest N on val (prob): {best_n} (F1 {best_f1*100:.2f}%, +{(best_f1-baseline_f1)*100:.2f}pp)")

    best_n_raw, best_f1_raw, *_ = max(val_results_raw, key=lambda r: r[1])
    print(f"Best N on val (raw_score): {best_n_raw} (F1 {best_f1_raw*100:.2f}%, +{(best_f1_raw-baseline_f1)*100:.2f}pp)")

    best_n_final, best_f1_final, *_ = max(val_results_final, key=lambda r: r[1])
    print(f"Best N on val (final_score): {best_n_final} (F1 {best_f1_final*100:.2f}%, +{(best_f1_final-baseline_f1)*100:.2f}pp)")

    # Test shape at best N
    print(f"\n=== TEST shape at N={best_n} ===")
    test_blended: dict[str, set[str]] = {}
    for qid, qrows in test_grouped.items():
        base_set = set(baseline_test.get(qid, set()))
        additions = additions_for_query(qrows, base_set, best_n)
        test_blended[qid] = base_set | {r["citation"] for r in additions}

    sizes = [len(p) for p in test_blended.values()]
    print(f"  Mean cites:   {mean(sizes):.2f}  (baseline {mean(len(p) for p in baseline_test.values()):.2f})")

    # Jaccard with baseline
    qids = sorted(set(test_blended) & set(baseline_test))
    jacc = []
    for qid in qids:
        a, b = test_blended[qid], baseline_test[qid]
        u = a | b
        jacc.append(len(a & b) / len(u) if u else 1.0)
    print(f"  Mean Jaccard with baseline: {mean(jacc)*100:.2f}%")

    if args.output_prefix:
        for sort_key, label in [("prob", "prob"), ("raw_score", "raw"), ("final_score", "final")]:
            for n in [1, 2, 3]:
                val_blended: dict[str, set[str]] = {}
                for qid, qrows in val_grouped.items():
                    base_set = set(baseline_val.get(qid, set()))
                    additions = additions_for_query(qrows, base_set, n, sort_key=sort_key)
                    val_blended[qid] = base_set | {r["citation"] for r in additions}

                test_blended_out: dict[str, set[str]] = {}
                for qid, qrows in test_grouped.items():
                    base_set = set(baseline_test.get(qid, set()))
                    additions = additions_for_query(qrows, base_set, n, sort_key=sort_key)
                    test_blended_out[qid] = base_set | {r["citation"] for r in additions}

                val_out = Path(f"{args.output_prefix}_val_{label}_n{n}.csv")
                test_out = Path(f"{args.output_prefix}_test_{label}_n{n}.csv")
                write_predictions(val_out, val_blended)
                write_predictions(test_out, test_blended_out)
                print(f"Wrote {val_out} and {test_out}")


if __name__ == "__main__":
    main()
