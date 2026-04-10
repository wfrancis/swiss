#!/usr/bin/env python3
"""
Compute candidate-pool recall on val: how many gold citations are even
present in the court-dense candidate pool? This is the absolute ceiling
for any selector trained on this data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean

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


def load_baseline(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["query_id"]] = set(c for c in (row.get("predicted_citations") or "").split(";") if c)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--judged", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--label", type=str, default="court-dense val")
    args = p.parse_args()

    bundles = json.loads(args.judged.read_text())["bundles"]
    gold_map = load_gold(args.gold)
    baseline = load_baseline(args.baseline) if args.baseline else None

    print(f"\n=== {args.label} ===")
    print(f"{'qid':>10} {'#gold':>6} {'#pool':>6} {'pool∩gold':>10} {'recall%':>8} | {'#law∩':>7} {'#court∩':>9} | {'#bsl':>5} {'bsl∩gold':>9}")

    overall_recall = []
    overall_law_recall = []
    overall_court_recall = []
    bsl_recall = []
    pool_minus_bsl_recall = []

    for bundle in bundles:
        qid = bundle["query_id"]
        gold = gold_map.get(qid, set())
        if not gold:
            continue
        pool_cites = {c["citation"] for c in bundle["candidates"]}
        hit = pool_cites & gold
        recall = len(hit) / len(gold)
        overall_recall.append(recall)

        gold_law = {g for g in gold if is_law(g)}
        gold_court = {g for g in gold if is_court(g)}
        law_hit = pool_cites & gold_law
        court_hit = pool_cites & gold_court
        if gold_law:
            overall_law_recall.append(len(law_hit) / len(gold_law))
        if gold_court:
            overall_court_recall.append(len(court_hit) / len(gold_court))

        bsl_str = "  --  "
        bsl_hit_str = "   --  "
        if baseline:
            bsl = baseline.get(qid, set())
            bsl_hit = bsl & gold
            bsl_recall.append(len(bsl_hit) / len(gold))
            bsl_str = f"{len(bsl):5d}"
            bsl_hit_str = f"{len(bsl_hit):8d}"
            # how much does pool add beyond baseline?
            pool_minus_bsl = (pool_cites - bsl) & gold
            pool_minus_bsl_recall.append(len(pool_minus_bsl) / len(gold))

        print(
            f"{qid:>10} "
            f"{len(gold):6d} "
            f"{len(pool_cites):6d} "
            f"{len(hit):10d} "
            f"{recall*100:7.2f}% | "
            f"{len(law_hit):7d} "
            f"{len(court_hit):9d} | "
            f"{bsl_str} "
            f"{bsl_hit_str}"
        )

    print(f"\nMean pool recall:            {mean(overall_recall)*100:.2f}%")
    print(f"Mean law recall:             {mean(overall_law_recall)*100:.2f}%")
    print(f"Mean court recall:           {mean(overall_court_recall)*100:.2f}%")
    if bsl_recall:
        print(f"Mean baseline recall:        {mean(bsl_recall)*100:.2f}%  (for reference)")
        print(f"Mean pool-only recall (gold in court-dense pool but NOT in baseline preds): {mean(pool_minus_bsl_recall)*100:.2f}%")


if __name__ == "__main__":
    main()
