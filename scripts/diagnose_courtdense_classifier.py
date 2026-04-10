#!/usr/bin/env python3
"""
Diagnose whether the court-dense meta-selector classifier has useful signal
beyond its current K=6 truncation.

Sweeps K (top-K-by-classifier-prob, ignoring threshold and court cap) from
6 to 25 on val and test, reporting:
  - val macro F1
  - test mean cites
  - test law / court mix
  - test Jaccard vs the 0.30257 baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from statistics import mean, median

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from run_v11_meta_selector import (  # noqa: E402
    build_rows,
    group_rows_by_query,
    predict_rows,
)

import re

LAW_RX = re.compile(r"^Art\.\s")
COURT_BGE_RX = re.compile(r"^BGE ")
COURT_CASE_RX = re.compile(r"^\d+[A-Z]_\d+/\d+")


def is_law(c: str) -> bool:
    return bool(LAW_RX.match(c))


def is_court(c: str) -> bool:
    return bool(COURT_BGE_RX.match(c) or COURT_CASE_RX.match(c))


def load_gold(path: Path) -> dict[str, set[str]]:
    with path.open() as f:
        return {row["query_id"]: set(row["gold_citations"].split(";")) for row in csv.DictReader(f)}


def load_predictions(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            qid = row["query_id"]
            out[qid] = set(c for c in (row.get("predicted_citations") or "").split(";") if c)
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
    return len(a & b) / len(union) if union else 1.0


def topk_pred(rows: list[dict], k: int) -> set[str]:
    ranked = sorted(rows, key=lambda r: (-r["prob"], r["citation"]))
    out: set[str] = set()
    for r in ranked:
        if len(out) >= k:
            break
        out.add(r["citation"])
    return out


def evaluate_split(
    bundles: list[dict],
    model,
    source_names: list[str],
    gold: dict[str, set[str]] | None,
    baseline: dict[str, set[str]] | None,
    k_values: list[int],
) -> list[dict]:
    rows = build_rows(bundles, gold_map=None, source_names=source_names)
    predict_rows(model, rows)
    grouped = group_rows_by_query(rows)
    out = []
    for k in k_values:
        preds = {qid: topk_pred(qrows, k) for qid, qrows in grouped.items()}
        sizes = [len(p) for p in preds.values()]
        law_means = [sum(1 for c in p if is_law(c)) for p in preds.values()]
        court_means = [sum(1 for c in p if is_court(c)) for p in preds.values()]
        total_cites = sum(sizes)
        record = {
            "k": k,
            "mean_cites": mean(sizes) if sizes else 0.0,
            "median_cites": median(sizes) if sizes else 0.0,
            "law_pct": 100.0 * sum(law_means) / total_cites if total_cites else 0.0,
            "court_pct": 100.0 * sum(court_means) / total_cites if total_cites else 0.0,
        }
        if gold is not None:
            scores = [f1(preds[qid], gold[qid]) for qid in gold if qid in preds]
            record["macro_f1"] = mean(scores) if scores else 0.0
            record["floor_f1"] = min(scores) if scores else 0.0
        if baseline is not None:
            qids = sorted(set(preds) & set(baseline))
            j = [jaccard(preds[qid], baseline[qid]) for qid in qids]
            record["mean_jaccard"] = mean(j) if j else 0.0
            record["min_jaccard"] = min(j) if j else 0.0
        out.append(record)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pkl", type=Path, required=True)
    parser.add_argument("--val-judged", type=Path, required=True)
    parser.add_argument("--test-judged", type=Path, required=True)
    parser.add_argument("--val-gold", type=Path, required=True)
    parser.add_argument("--baseline-test", type=Path, required=True)
    parser.add_argument("--baseline-val", type=Path, required=True)
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=30)
    parser.add_argument("--k-step", type=int, default=2)
    args = parser.parse_args()

    with args.model_pkl.open("rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    source_names = bundle["source_names"]
    print(f"Loaded model. Source-name features: {len(source_names)}", flush=True)

    val_payload = json.loads(args.val_judged.read_text())["bundles"]
    test_payload = json.loads(args.test_judged.read_text())["bundles"]
    val_gold = load_gold(args.val_gold)
    baseline_val = load_predictions(args.baseline_val)
    baseline_test = load_predictions(args.baseline_test)

    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))

    print(f"\nVal candidates per query (avg): {sum(len(b['candidates']) for b in val_payload) / len(val_payload):.1f}", flush=True)
    print(f"Test candidates per query (avg): {sum(len(b['candidates']) for b in test_payload) / len(test_payload):.1f}", flush=True)

    print("\n=== VAL — top-K by classifier prob (no threshold, no court cap) ===")
    print(f"{'K':>3} {'macro F1':>9} {'floor':>7} {'#cites':>7} {'law%':>6} {'court%':>7} {'meanJ':>7}")
    val_results = evaluate_split(val_payload, model, source_names, val_gold, baseline_val, k_values)
    for r in val_results:
        print(
            f"{r['k']:>3} "
            f"{r['macro_f1']*100:8.2f}% "
            f"{r['floor_f1']*100:6.2f}% "
            f"{r['mean_cites']:7.2f} "
            f"{r['law_pct']:6.2f} "
            f"{r['court_pct']:7.2f} "
            f"{r['mean_jaccard']*100:6.2f}%"
        )

    print("\n=== TEST — top-K by classifier prob (no gold) ===")
    print(f"{'K':>3} {'#cites':>7} {'law%':>6} {'court%':>7} {'meanJ':>7} {'minJ':>6}")
    test_results = evaluate_split(test_payload, model, source_names, None, baseline_test, k_values)
    for r in test_results:
        print(
            f"{r['k']:>3} "
            f"{r['mean_cites']:7.2f} "
            f"{r['law_pct']:6.2f} "
            f"{r['court_pct']:7.2f} "
            f"{r['mean_jaccard']*100:6.2f}% "
            f"{r['min_jaccard']*100:5.2f}%"
        )


if __name__ == "__main__":
    main()
