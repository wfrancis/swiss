#!/usr/bin/env python3
"""
C1 diagnostic: retrain the court-dense classifier without judge-label
features (judge_label one-hot + judge_confidence) and compare its
top-K val/test curves to the full-feature classifier trained on the
same data.

Hypothesis: DeepSeek's 99.7% court-reject rate is teaching the classifier
that 'court' implies 'reject', destroying its ability to surface court
citations. Removing the judge label features should test this.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from run_v11_meta_selector import (  # noqa: E402
    AUTO_BUCKET_NAMES,
    LABEL_NAMES,
    collect_source_names,
    group_rows_by_query,
    load_bundles,
    load_gold,
)


LAW_RX = re.compile(r"^Art\.\s")
COURT_BGE_RX = re.compile(r"^BGE ")
COURT_CASE_RX = re.compile(r"^\d+[A-Z]_\d+/\d+")


def is_law(c: str) -> bool:
    return bool(LAW_RX.match(c))


def is_court(c: str) -> bool:
    return bool(COURT_BGE_RX.match(c) or COURT_CASE_RX.match(c))


def candidate_features(
    candidate: dict[str, Any],
    estimated_count: int,
    source_names: list[str],
    *,
    drop_judge: bool,
) -> np.ndarray:
    feats: list[float] = [
        1.0 if candidate["kind"] == "law" else 0.0,
        1.0 if candidate["kind"] == "court" else 0.0,
    ]
    if not drop_judge:
        feats.extend(1.0 if candidate["judge_label"] == label else 0.0 for label in LABEL_NAMES)

    auto_bucket = candidate["auto_bucket"] or "other"
    feats.extend(1.0 if auto_bucket == name else 0.0 for name in AUTO_BUCKET_NAMES)

    if not drop_judge:
        feats.append(float(candidate["judge_confidence"]))
    feats.extend(
        [
            float(candidate["final_score"]),
            float(candidate["raw_score"]),
            float(candidate["gpt_full_freq"]),
            float(candidate["is_explicit"]),
            float(candidate["is_query_case"]),
        ]
    )
    for key in ["dense_rank", "bm25_rank", "court_dense_rank"]:
        rank = candidate.get(key)
        feats.append(0.0 if rank is None else 1.0 / (1.0 + float(rank)))

    source_set = set(candidate["sources"])
    feats.extend(1.0 if source in source_set else 0.0 for source in source_names)
    feats.append(float(len(source_set)))
    feats.append(float(estimated_count))
    return np.array(feats, dtype=float)


def build_rows(
    bundles: list[dict[str, Any]],
    gold_map: dict[str, set[str]] | None,
    source_names: list[str],
    *,
    drop_judge: bool,
) -> list[dict[str, Any]]:
    rows = []
    for bundle in bundles:
        gold = gold_map.get(bundle["query_id"], set()) if gold_map else set()
        for c in bundle["candidates"]:
            rows.append(
                {
                    "query_id": bundle["query_id"],
                    "citation": c["citation"],
                    "kind": c["kind"],
                    "estimated_count": bundle["estimated_count"],
                    "features": candidate_features(c, bundle["estimated_count"], source_names, drop_judge=drop_judge),
                    "label": int(c["citation"] in gold) if gold_map is not None else 0,
                }
            )
    return rows


def fit_model(rows: list[dict[str, Any]], seed: int = 0) -> ExtraTreesClassifier:
    x = np.stack([r["features"] for r in rows])
    y = np.array([r["label"] for r in rows])
    model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=4,
        n_jobs=-1,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(x, y)
    return model


def predict_rows(model: ExtraTreesClassifier, rows: list[dict[str, Any]]) -> None:
    x = np.stack([r["features"] for r in rows])
    probs = model.predict_proba(x)[:, 1]
    for r, p in zip(rows, probs):
        r["prob"] = float(p)


def f1(pred: set[str], gold: set[str]) -> float:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def topk_pred(rows: list[dict], k: int) -> set[str]:
    ranked = sorted(rows, key=lambda r: (-r["prob"], r["citation"]))
    out: set[str] = set()
    for r in ranked:
        if len(out) >= k:
            break
        out.add(r["citation"])
    return out


def evaluate(
    rows: list[dict],
    gold: dict[str, set[str]] | None,
    baseline: dict[str, set[str]] | None,
    k_values: list[int],
) -> list[dict]:
    grouped = group_rows_by_query(rows)
    results = []
    for k in k_values:
        preds = {qid: topk_pred(qrows, k) for qid, qrows in grouped.items()}
        sizes = [len(p) for p in preds.values()]
        law_counts = [sum(1 for c in p if is_law(c)) for p in preds.values()]
        court_counts = [sum(1 for c in p if is_court(c)) for p in preds.values()]
        total = sum(sizes)
        rec = {
            "k": k,
            "mean_cites": mean(sizes),
            "law_pct": 100.0 * sum(law_counts) / total if total else 0.0,
            "court_pct": 100.0 * sum(court_counts) / total if total else 0.0,
        }
        if gold is not None:
            scores = [f1(preds[qid], gold[qid]) for qid in gold if qid in preds]
            rec["macro_f1"] = mean(scores) if scores else 0.0
            rec["floor_f1"] = min(scores) if scores else 0.0
        if baseline is not None:
            qids = sorted(set(preds) & set(baseline))
            j = [jaccard(preds[qid], baseline[qid]) for qid in qids]
            rec["mean_j"] = mean(j) if j else 0.0
            rec["min_j"] = min(j) if j else 0.0
        results.append(rec)
    return results


def load_predictions(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["query_id"]] = set(c for c in (row.get("predicted_citations") or "").split(";") if c)
    return out


def print_table(label: str, val_rows: list[dict], test_rows: list[dict]) -> None:
    print(f"\n=== {label} ===")
    print(f"{'K':>3} | {'val F1':>7} {'val flr':>8} {'val cites':>9} {'val law%':>9} {'val ct%':>8} | {'tst cites':>9} {'tst law%':>9} {'tst ct%':>8} {'tst meanJ':>10}")
    for v, t in zip(val_rows, test_rows):
        print(
            f"{v['k']:>3} | "
            f"{v['macro_f1']*100:6.2f}% "
            f"{v['floor_f1']*100:7.2f}% "
            f"{v['mean_cites']:9.2f} "
            f"{v['law_pct']:8.2f}% "
            f"{v['court_pct']:7.2f}% | "
            f"{t['mean_cites']:9.2f} "
            f"{t['law_pct']:8.2f}% "
            f"{t['court_pct']:7.2f}% "
            f"{t['mean_j']*100:9.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-judged", type=Path, required=True)
    parser.add_argument("--val-judged", type=Path, required=True)
    parser.add_argument("--test-judged", type=Path, required=True)
    parser.add_argument("--train-gold", type=Path, required=True)
    parser.add_argument("--val-gold", type=Path, required=True)
    parser.add_argument("--baseline-test", type=Path, required=True)
    parser.add_argument("--baseline-val", type=Path, required=True)
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=30)
    parser.add_argument("--k-step", type=int, default=2)
    args = parser.parse_args()

    train_bundles = load_bundles([args.train_judged])
    val_bundles = load_bundles([args.val_judged])
    test_bundles = load_bundles([args.test_judged])
    train_gold = load_gold(args.train_gold)
    val_gold = load_gold(args.val_gold)
    baseline_val = load_predictions(args.baseline_val)
    baseline_test = load_predictions(args.baseline_test)
    source_names = collect_source_names(train_bundles, val_bundles, test_bundles)
    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))

    print(f"Train queries: {len(train_bundles)}  Val: {len(val_bundles)}  Test: {len(test_bundles)}", flush=True)
    print(f"Source-name features: {len(source_names)}", flush=True)

    for label, drop_judge in [("FULL features (control)", False), ("NO judge_label / judge_confidence", True)]:
        print(f"\nFitting {label}...", flush=True)
        train_rows = build_rows(train_bundles, train_gold, source_names, drop_judge=drop_judge)
        val_rows = build_rows(val_bundles, val_gold, source_names, drop_judge=drop_judge)
        test_rows = build_rows(test_bundles, None, source_names, drop_judge=drop_judge)
        model = fit_model(train_rows, seed=0)
        predict_rows(model, val_rows)
        predict_rows(model, test_rows)
        val_results = evaluate(val_rows, val_gold, baseline_val, k_values)
        test_results = evaluate(test_rows, None, baseline_test, k_values)
        print_table(label, val_results, test_results)


if __name__ == "__main__":
    main()
