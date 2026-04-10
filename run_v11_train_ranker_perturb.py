#!/usr/bin/env python3
"""
Train a local ranker on V11 candidate bundles, then use it only for tiny
winner-anchored law additions.

This keeps the proven Kaggle-winning branch as the anchor and uses the
train-fitted model as a conservative perturbation signal instead of a full
replacement selector.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

from run_v11_train_selector import (
    build_priors,
    build_rows,
    collect_source_names,
    f1_score,
    fit_model,
    group_rows_by_query,
    load_bundle_payloads,
    load_gold,
    predict_rows,
    top_statute_codes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--train-gold", type=Path, required=True)
    parser.add_argument("--val-candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--val-gold", type=Path, required=True)
    parser.add_argument("--base-val-csv", type=Path, required=True)
    parser.add_argument("--test-candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--base-test-csv", type=Path, required=True)
    parser.add_argument("--output-val-csv", type=Path, required=True)
    parser.add_argument("--output-test-csv", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, set[str]]:
    with path.open() as f:
        return {
            row["query_id"]: {c for c in row["predicted_citations"].split(";") if c}
            for row in csv.DictReader(f)
        }


def write_predictions(path: Path, predictions: dict[str, set[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for query_id in sorted(predictions):
            writer.writerow([query_id, ";".join(sorted(predictions[query_id]))])


def apply_additions(
    base_predictions: dict[str, set[str]],
    rows_by_query: dict[str, list[dict]],
    *,
    prob_thresh: float,
    final_thresh: float,
    max_add: int,
    rank_cap: int,
) -> dict[str, set[str]]:
    out = {query_id: set(preds) for query_id, preds in base_predictions.items()}
    for query_id, rows in rows_by_query.items():
        ranked = sorted(
            rows,
            key=lambda row: (
                -row["prob"],
                -row["final_score"],
                row["baseline_rank"],
                row["citation"],
            ),
        )
        adds = 0
        for row in ranked:
            if adds >= max_add:
                break
            if row["citation"] in out[query_id]:
                continue
            if row["kind"] != "law":
                continue
            if row["prob"] < prob_thresh:
                continue
            if row["final_score"] < final_thresh:
                continue
            if row["baseline_rank"] > rank_cap:
                continue
            out[query_id].add(row["citation"])
            adds += 1
    return out


def main() -> None:
    args = parse_args()

    train_gold = load_gold(args.train_gold)
    val_gold = load_gold(args.val_gold)

    train_payloads = load_bundle_payloads(args.train_candidates)
    val_payloads = load_bundle_payloads(args.val_candidates)
    test_payloads = load_bundle_payloads(args.test_candidates)
    source_names = collect_source_names(train_payloads, val_payloads, test_payloads)

    train_query_ids = {
        bundle.query_id
        for payload in train_payloads
        for bundle in payload["bundles"]
    }
    priors = build_priors(train_gold, train_query_ids)
    known_codes = top_statute_codes(priors)

    train_rows = build_rows(train_payloads, train_gold, source_names, priors, known_codes)
    model = fit_model(train_rows, args.seed)

    val_rows = build_rows(val_payloads, val_gold, source_names, priors, known_codes)
    test_rows = build_rows(test_payloads, None, source_names, priors, known_codes)
    predict_rows(model, val_rows)
    predict_rows(model, test_rows)
    val_rows_by_query = group_rows_by_query(val_rows)
    test_rows_by_query = group_rows_by_query(test_rows)

    base_val = load_predictions(args.base_val_csv)
    base_test = load_predictions(args.base_test_csv)

    search_space = product(
        [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3],
        [0.3, 0.4, 0.5, 0.6, 0.7],
        [0, 1, 2, 3],
        [20, 30, 40, 60, 80, 120],
    )

    best = None
    for prob_thresh, final_thresh, max_add, rank_cap in search_space:
        candidate = apply_additions(
            base_val,
            val_rows_by_query,
            prob_thresh=prob_thresh,
            final_thresh=final_thresh,
            max_add=max_add,
            rank_cap=rank_cap,
        )
        macro = sum(f1_score(candidate[qid], val_gold[qid]) for qid in sorted(val_gold)) / len(val_gold)
        added = sum(len(candidate[qid] - base_val[qid]) for qid in candidate)
        result = {
            "val_macro_f1": macro,
            "prob_thresh": prob_thresh,
            "final_thresh": final_thresh,
            "max_add": max_add,
            "rank_cap": rank_cap,
            "added_vs_base_val": added,
        }
        if best is None or result["val_macro_f1"] > best["val_macro_f1"]:
            best = result

    if best is None:
        raise RuntimeError("perturb search produced no result")

    val_predictions = apply_additions(
        base_val,
        val_rows_by_query,
        prob_thresh=best["prob_thresh"],
        final_thresh=best["final_thresh"],
        max_add=best["max_add"],
        rank_cap=best["rank_cap"],
    )
    test_predictions = apply_additions(
        base_test,
        test_rows_by_query,
        prob_thresh=best["prob_thresh"],
        final_thresh=best["final_thresh"],
        max_add=best["max_add"],
        rank_cap=best["rank_cap"],
    )

    best["added_vs_base_test"] = sum(
        len(test_predictions[qid] - base_test[qid]) for qid in test_predictions
    )
    best["changed_test_queries"] = sum(
        1 for qid in test_predictions if test_predictions[qid] != base_test[qid]
    )

    write_predictions(args.output_val_csv, val_predictions)
    write_predictions(args.output_test_csv, test_predictions)

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(json.dumps(best, ensure_ascii=False, indent=2))
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
