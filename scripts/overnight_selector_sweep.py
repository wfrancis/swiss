#!/usr/bin/env python3
"""
Sweep selector parameters on EXISTING judged artifacts — zero API calls.

The V11 pipeline has 5 selector knobs (court_fraction, must_keep_confidence,
max_output, min_output, min_courts_if_any) that only affect the SELECT stage.
Re-judging is not needed. This script sweeps all combinations, evaluates on
val, and writes the top-K val+test CSV pairs.

Usage:
    .venv/bin/python scripts/overnight_selector_sweep.py \
        --top-k 5 \
        --output-dir submissions/overnight_selector_sweep
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import pickle
import sys
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from pipeline_v11 import V11Config, evaluate_predictions, select_candidates


def load_pickle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def sweep_select(bundles: list, rows: list, grid: list[dict], gold_map: dict) -> list[dict]:
    results = []
    for idx, params in enumerate(grid):
        config = V11Config(
            split="val",
            use_judge=True,
            judge_model="sweep",
            prompt_version="sweep",
            law_judge_topk=60,
            court_judge_topk=36,
            law_batch_size=20,
            court_batch_size=12,
            use_court_dense=True,
            court_dense_query_limit=8,
            court_dense_topk=160,
            max_output=params["max_output"],
            min_output=params["min_output"],
            court_fraction=params["court_fraction"],
            min_courts_if_any=params["min_courts_if_any"],
            must_keep_confidence=params["must_keep_confidence"],
            cache_path=Path("/dev/null"),
            court_text_cache_path=Path("/dev/null"),
            court_dense_cache_path=Path("/dev/null"),
            query_offset=0,
            max_queries=None,
        )
        predictions: dict[str, set[str]] = {}
        for bundle in bundles:
            selected = select_candidates(bundle, config)
            predictions[bundle.query_id] = {c.citation for c in selected}

        macro_f1, per_query = evaluate_predictions(predictions, gold_map)
        avg_preds = sum(len(p) for p in predictions.values()) / len(predictions)
        results.append({
            "rank": 0,
            "params": params,
            "val_f1": macro_f1,
            "avg_preds": avg_preds,
            "predictions": predictions,
        })
        if (idx + 1) % 100 == 0:
            print(f"  swept {idx + 1}/{len(grid)}...", flush=True)

    results.sort(key=lambda r: r["val_f1"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results


def write_csv(predictions: dict[str, set[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions):
            writer.writerow([qid, ";".join(sorted(predictions[qid]))])


def apply_params_to_split(bundles: list, params: dict) -> dict[str, set[str]]:
    config = V11Config(
        split="test",
        use_judge=True,
        judge_model="sweep",
        prompt_version="sweep",
        law_judge_topk=60,
        court_judge_topk=36,
        law_batch_size=20,
        court_batch_size=12,
        use_court_dense=True,
        court_dense_query_limit=8,
        court_dense_topk=160,
        max_output=params["max_output"],
        min_output=params["min_output"],
        court_fraction=params["court_fraction"],
        min_courts_if_any=params["min_courts_if_any"],
        must_keep_confidence=params["must_keep_confidence"],
        cache_path=Path("/dev/null"),
        court_text_cache_path=Path("/dev/null"),
        court_dense_cache_path=Path("/dev/null"),
        query_offset=0,
        max_queries=None,
    )
    predictions: dict[str, set[str]] = {}
    for bundle in bundles:
        selected = select_candidates(bundle, config)
        predictions[bundle.query_id] = {c.citation for c in selected}
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-judged", type=Path,
                        default=BASE / "artifacts/v11/val_v11_strict_v1/judged_bundles.pkl")
    parser.add_argument("--test-judged", type=Path,
                        default=BASE / "artifacts/v11/test_v11_strict_v1/judged_bundles.pkl")
    parser.add_argument("--gold-csv", type=Path, default=BASE / "data/val.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=BASE / "submissions")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("Loading judged artifacts...", flush=True)
    val_payload = load_pickle(args.val_judged)
    test_payload = load_pickle(args.test_judged)
    val_bundles = val_payload["bundles"]
    test_bundles = test_payload["bundles"]

    gold_map = {}
    with open(args.gold_csv) as f:
        for row in csv.DictReader(f):
            gold_map[row["query_id"]] = set(row["gold_citations"].split(";"))

    # Build parameter grid
    court_fractions = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    must_keep_confs = [0.65, 0.70, 0.75, 0.80, 0.86, 0.90, 0.95]
    max_outputs = [25, 30, 35, 40, 45, 50]
    min_outputs = [6, 8, 10, 12, 15]
    min_courts = [2, 3, 4, 5, 6]

    grid = []
    for cf, mkc, maxo, mino, mc in itertools.product(
        court_fractions, must_keep_confs, max_outputs, min_outputs, min_courts
    ):
        if mino >= maxo:
            continue
        grid.append({
            "court_fraction": cf,
            "must_keep_confidence": mkc,
            "max_output": maxo,
            "min_output": mino,
            "min_courts_if_any": mc,
        })

    print(f"Sweeping {len(grid)} selector configurations on val...", flush=True)
    results = sweep_select(val_bundles, val_payload.get("rows", []), grid, gold_map)

    # Report top results
    print(f"\n{'='*70}")
    print(f"TOP {args.top_k} SELECTOR CONFIGURATIONS")
    print(f"{'='*70}")
    baseline_f1 = 0.282430
    for r in results[:args.top_k]:
        delta = (r["val_f1"] - baseline_f1) * 100
        print(f"  #{r['rank']}: val_F1={r['val_f1']:.6f} ({delta:+.2f}pp) "
              f"avg_preds={r['avg_preds']:.1f} | {r['params']}")

    # Write top-K val+test CSV pairs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for r in results[:args.top_k]:
        tag = f"overnight_selector_sweep_top{r['rank']}"
        val_path = args.output_dir / f"val_pred_{tag}.csv"
        test_path = args.output_dir / f"test_submission_{tag}.csv"

        write_csv(r["predictions"], val_path)
        test_preds = apply_params_to_split(test_bundles, r["params"])
        write_csv(test_preds, test_path)
        print(f"  Wrote {val_path.name} + {test_path.name}")

    # Also write the full results summary
    summary_path = args.output_dir / "overnight_selector_sweep_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Swept {len(grid)} configurations\n")
        f.write(f"Baseline val F1: {baseline_f1:.6f}\n\n")
        for r in results[:20]:
            delta = (r["val_f1"] - baseline_f1) * 100
            f.write(f"#{r['rank']}: val_F1={r['val_f1']:.6f} ({delta:+.2f}pp) "
                    f"avg_preds={r['avg_preds']:.1f} | {r['params']}\n")
    print(f"\nFull summary: {summary_path}")


if __name__ == "__main__":
    main()
