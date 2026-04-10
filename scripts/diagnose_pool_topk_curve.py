#!/usr/bin/env python3
"""For each val query, compute the gold-recall ceiling at various top-K values
of the candidate pool, plus the baseline-floor F1 if we just trust must_include.

Helps decide:
  - what top-K to use for the LLM selector pool
  - whether "auto-include must_include + LLM adds from plausible/reject" is safe
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from llm_selector_ds_v1 import filter_top_k  # noqa: E402

VAL_BUNDLES = REPO / "artifacts/v11/val_v11_courtdense_ds_val1/judged_bundles.json"
VAL_GOLD = REPO / "data/val.csv"
BASELINE = REPO / "submissions/val_pred_baseline_public_best_30257.csv"


def f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(gold)
    return 2 * p * r / (p + r)


def main():
    bundles = json.loads(VAL_BUNDLES.read_text())["bundles"]
    by_qid = {b["query_id"]: b for b in bundles}

    gold = {}
    with VAL_GOLD.open() as f_:
        for row in csv.DictReader(f_):
            gold[row["query_id"]] = {c for c in row["gold_citations"].split(";") if c}

    baseline = {}
    with BASELINE.open() as f_:
        for row in csv.DictReader(f_):
            baseline[row["query_id"]] = {c for c in row["predicted_citations"].split(";") if c}

    qids = sorted(by_qid)

    print("="*80)
    print("BASELINE (0.30257) per-query F1 reference")
    print("="*80)
    f1s = [f1(baseline[q], gold[q]) for q in qids]
    print(f"  baseline macro F1: {mean(f1s):.4f}  per-query: {[f'{x:.3f}' for x in f1s]}")

    print()
    print("="*80)
    print("FLOOR: F1 if we just predict ALL must_include cites")
    print("="*80)
    floor_f1s = []
    for qid in qids:
        b = by_qid[qid]
        must = {c["citation"] for c in b["candidates"] if c.get("judge_label") == "must_include"}
        f = f1(must, gold[qid])
        floor_f1s.append(f)
        n_must_in_gold = len(must & gold[qid])
        print(f"  {qid}: must={len(must):3d} gold={len(gold[qid]):3d} tp={n_must_in_gold:3d} F1={f:.3f}  (baseline F1={f1(baseline[qid], gold[qid]):.3f})")
    print(f"  must_include floor macro F1: {mean(floor_f1s):.4f}")

    print()
    print("="*80)
    print("CEILING: gold recall in top-K pool (sorted by candidate_signal)")
    print("="*80)
    print(f"{'qid':<10} {'gold':>5} {'full':>5} | " + " ".join(f'K={k:<4}' for k in [50, 80, 120, 200, 300, 500, 1000, 9999]))
    for qid in qids:
        b = by_qid[qid]
        g = gold[qid]
        full_pool = {c["citation"] for c in b["candidates"]}
        full_recall = len(g & full_pool) / len(g)
        recalls = []
        for k in [50, 80, 120, 200, 300, 500, 1000, 9999]:
            top = filter_top_k(b["candidates"], k)
            top_cites = {c["citation"] for c in top}
            recalls.append(len(g & top_cites) / len(g))
        print(f"{qid:<10} {len(g):>5} {full_recall:>4.0%} | " + " ".join(f"{r:>4.0%} " for r in recalls))

    # Mean ceiling at each K
    print()
    print("MEAN ceiling across all 10 val queries:")
    for k in [50, 80, 120, 200, 300, 500, 1000, 9999]:
        rs = []
        for qid in qids:
            b = by_qid[qid]
            g = gold[qid]
            top = filter_top_k(b["candidates"], k)
            top_cites = {c["citation"] for c in top}
            rs.append(len(g & top_cites) / len(g))
        print(f"  K={k:<5}: mean recall = {mean(rs):.1%}")

    print()
    print("="*80)
    print("ORACLE F1: best possible if we picked perfect cites from top-K pool")
    print("="*80)
    print(f"{'K':<6} {'mean F1':<10} {'mean recall':<12}")
    for k in [50, 80, 120, 200, 300, 500, 1000, 9999]:
        f1s_k = []
        for qid in qids:
            b = by_qid[qid]
            g = gold[qid]
            top = filter_top_k(b["candidates"], k)
            top_cites = {c["citation"] for c in top}
            oracle_pred = g & top_cites  # perfect picks
            f1s_k.append(f1(oracle_pred, g))
        print(f"K={k:<5} {mean(f1s_k):<10.4f}")


if __name__ == "__main__":
    main()
