#!/usr/bin/env python3
"""Diagnose why llm_selector_ds_v1 lost on the smoke test.

For val_001 and val_002, compute:
  - gold size
  - gold recall in full pool (~2200 cands)
  - gold recall in top-80 pool
  - gold recall in must_include subset
  - baseline (0.30257 sub) recall
  - llm-smoke-pick recall + which gold cites it caught vs missed
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from llm_selector_ds_v1 import filter_top_k  # noqa: E402

VAL_BUNDLES = REPO / "artifacts/v11/val_v11_courtdense_ds_val1/judged_bundles.json"
VAL_GOLD = REPO / "data/val.csv"
BASELINE = REPO / "submissions/val_pred_baseline_public_best_30257.csv"
SMOKE_PRED = REPO / "submissions/llm_selector_ds_v1_val_smoke.csv"


def load_csv_predictions(path: Path, col: str) -> dict[str, set[str]]:
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["query_id"]] = {c for c in row[col].split(";") if c}
    return out


def main():
    bundles = json.loads(VAL_BUNDLES.read_text())["bundles"]
    by_qid = {b["query_id"]: b for b in bundles}

    gold = {}
    with VAL_GOLD.open() as f:
        for row in csv.DictReader(f):
            gold[row["query_id"]] = {c for c in row["gold_citations"].split(";") if c}

    baseline = load_csv_predictions(BASELINE, "predicted_citations")
    smoke = load_csv_predictions(SMOKE_PRED, "gold_citations")

    for qid in ["val_001", "val_002"]:
        b = by_qid[qid]
        g = gold[qid]
        full_pool = {c["citation"] for c in b["candidates"]}
        top80 = filter_top_k(b["candidates"], 80)
        top80_cites = {c["citation"] for c in top80}
        must_include = {c["citation"] for c in b["candidates"] if c.get("judge_label") == "must_include"}
        bl = baseline.get(qid, set())
        sm = smoke.get(qid, set())

        print(f"\n{'='*70}")
        print(f"{qid}  gold={len(g)}")
        print(f"{'='*70}")
        print(f"  full pool size:   {len(full_pool)}    gold ∩ pool:   {len(g & full_pool):3d}  recall={len(g & full_pool)/len(g):.1%}")
        print(f"  top-80 pool:      {len(top80_cites)}    gold ∩ top80:  {len(g & top80_cites):3d}  recall={len(g & top80_cites)/len(g):.1%}")
        print(f"  must_include set: {len(must_include)}    gold ∩ must:   {len(g & must_include):3d}  recall={len(g & must_include)/len(g):.1%}")
        print(f"  baseline pred:    {len(bl)}    gold ∩ base:   {len(g & bl):3d}  recall={len(g & bl)/len(g):.1%}")
        print(f"  llm smoke pred:   {len(sm)}    gold ∩ llm:    {len(g & sm):3d}  recall={len(g & sm)/len(g):.1%}")
        print()
        print(f"  CEILING from top-80 pool: gold ∩ top80 / gold = {len(g & top80_cites)/len(g):.1%}")
        print(f"  baseline recovers:        {len(g & bl)} of {len(g)} gold")
        print(f"  llm recovers:             {len(g & sm)} of {len(g)} gold")
        print(f"  llm caught NOT in baseline: {len((g & sm) - bl)}")
        print(f"  llm MISSED that baseline got: {len((g & bl) - sm)}")
        print(f"  baseline gold ∩ top-80 pool: {len(g & bl & top80_cites)}")

        # What labels did the LLM pick from?
        cite_to_label = {c["citation"]: c.get("judge_label") for c in b["candidates"]}
        from collections import Counter
        llm_label_dist = Counter(cite_to_label.get(c, "OUT_OF_POOL") for c in sm)
        print(f"  llm pick label dist: {dict(llm_label_dist)}")

        # What's in baseline that the LLM missed?
        baseline_missed_by_llm = bl - sm
        print(f"  cites in baseline but NOT in llm: {len(baseline_missed_by_llm)}")
        for c in sorted(baseline_missed_by_llm):
            in_top80 = "✓" if c in top80_cites else "✗"
            label = cite_to_label.get(c, "?")
            in_gold = "GOLD" if c in g else "    "
            print(f"    [{in_top80} top80] [{label:13s}] {in_gold} {c}")


if __name__ == "__main__":
    main()
