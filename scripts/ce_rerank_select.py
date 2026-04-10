#!/usr/bin/env python3
"""Turn cross-encoder scores into predicted-citation CSVs at various K cutoffs.

Reads artifacts/ce_rerank/{split}_scores.json and writes, for each K in the
sweep, submissions/ce_rerank_v1_{split}_k{K}.csv (top-K cites by CE score).

Also reports a quick val macro F1 per K for live feedback (the proper decision
comes from the multi-signal scorecard run separately).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent

GOLD_PATH = REPO / "data/val.csv"
SCORES_DIR = REPO / "artifacts/ce_rerank"
SUBS_DIR = REPO / "submissions"


def load_gold() -> dict[str, set[str]]:
    with GOLD_PATH.open() as f:
        return {
            row["query_id"]: {c for c in row["gold_citations"].split(";") if c}
            for row in csv.DictReader(f)
        }


def f1(pred: set[str], gold: set[str]) -> float:
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


def top_k(scores: dict[str, float], k: int) -> set[str]:
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return {c for c, _ in ranked[:k]}


def write_csv(path: Path, preds: dict[str, set[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "predicted_citations"])
        for qid in sorted(preds):
            w.writerow([qid, ";".join(sorted(preds[qid]))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--scores", type=Path, default=None)
    ap.add_argument(
        "--ks",
        type=str,
        default="15,18,20,22,25,28,30,33,35,40",
        help="comma-separated K values",
    )
    ap.add_argument("--tag", type=str, default="ce_rerank_v1")
    args = ap.parse_args()

    scores_path = args.scores or (SCORES_DIR / f"{args.split}_scores.json")
    print(f"Loading scores: {scores_path}")
    scores_by_qid: dict[str, dict[str, float]] = json.loads(scores_path.read_text())
    print(f"  {len(scores_by_qid)} queries")

    ks = [int(k) for k in args.ks.split(",")]
    gold = load_gold() if args.split == "val" else None

    print()
    print(f"{'K':>5} {'macro F1':>10} {'min':>7} {'max':>7} {'mean preds':>11}")
    for k in ks:
        preds: dict[str, set[str]] = {}
        for qid, scores in scores_by_qid.items():
            preds[qid] = top_k(scores, k)
        out_path = SUBS_DIR / f"{args.tag}_{args.split}_k{k}.csv"
        write_csv(out_path, preds)
        if gold:
            f1s = [f1(preds[qid], gold[qid]) for qid in sorted(gold) if qid in preds]
            mean_preds = mean(len(p) for p in preds.values())
            print(f"{k:>5} {mean(f1s):>10.4f} {min(f1s):>7.3f} {max(f1s):>7.3f} {mean_preds:>11.1f}")
        else:
            mean_preds = mean(len(p) for p in preds.values())
            print(f"{k:>5} {'--':>10} {'--':>7} {'--':>7} {mean_preds:>11.1f}")

    print(f"\nWrote {len(ks)} CSVs to {SUBS_DIR}/")
    print(f"Tag prefix: {args.tag}_{args.split}_k*")


if __name__ == "__main__":
    main()
