#!/usr/bin/env python3
"""Train a deterministic JSON offline selector from notebook feature dumps.

This consumes rows emitted by:

  OFFLINE_CANDIDATE_FEATURES_PATH=... python3 notebooks/swiss_prize_offline_retriever.py

Rows should contain query_id, citation, target, domains, features, and either a
label field or gold labels derivable from --gold-csv files. The output is a
plain JSON artifact consumed by notebooks/swiss_prize_offline_retriever.py.
No API calls, no pickle/joblib dependency, no hidden labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "precompute" / "offline_selector.json"


def split_citations(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def load_gold(paths: list[Path]) -> dict[str, set[str]]:
    gold: dict[str, set[str]] = {}
    csv.field_size_limit(sys.maxsize)
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qid = str(row.get("query_id", "")).strip()
                if not qid:
                    continue
                cites = set(split_citations(row.get("gold_citations", "")))
                if cites:
                    gold[qid] = cites
    return gold


def load_rows(
    feature_paths: list[Path],
    gold: dict[str, set[str]],
    *,
    max_negatives_per_query: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    negative_counts: dict[str, int] = defaultdict(int)
    raw_rows = 0
    for path in feature_paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw_rows += 1
                row = json.loads(line)
                qid = str(row.get("query_id", ""))
                citation = str(row.get("citation", ""))
                if "label" not in row and qid in gold:
                    row["label"] = 1 if citation in gold[qid] else 0
                if "label" in row:
                    row["label"] = int(row["label"])
                    if row["label"] == 0 and max_negatives_per_query and max_negatives_per_query > 0:
                        if negative_counts[qid] >= max_negatives_per_query:
                            continue
                        negative_counts[qid] += 1
                    rows.append(row)
    return rows, raw_rows


def downsample_rows(rows: list[dict[str, Any]], max_negatives_per_query: int | None) -> list[dict[str, Any]]:
    if not max_negatives_per_query or max_negatives_per_query <= 0:
        return rows
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("query_id", ""))].append(row)
    kept: list[dict[str, Any]] = []
    for qid in sorted(grouped):
        positives = [row for row in grouped[qid] if int(row.get("label", 0)) == 1]
        negatives = [row for row in grouped[qid] if int(row.get("label", 0)) == 0]
        negatives.sort(key=lambda row: (int(row.get("rank", 999999)), -float(row.get("base_score", 0.0)), str(row.get("citation", ""))))
        kept.extend(positives)
        kept.extend(negatives[:max_negatives_per_query])
    return kept


def domain_key(row: dict[str, Any]) -> str:
    domains = set(str(d) for d in row.get("domains", []))
    for name in ["detention", "criminal", "social", "family", "inheritance", "obligations"]:
        if name in domains:
            return name
    return "default"


def macro_f1_by_query(groups: dict[str, list[dict[str, Any]]], predictions: dict[str, set[str]]) -> float:
    scores = []
    for qid, rows in groups.items():
        pred = predictions.get(qid, set())
        gold = {str(r["citation"]) for r in rows if int(r.get("label", 0)) == 1}
        if not gold:
            continue
        tp = len(pred & gold)
        if not pred:
            scores.append(0.0)
        else:
            scores.append(2 * tp / (len(pred) + len(gold)))
    return float(sum(scores) / len(scores)) if scores else 0.0


def select_for_group(
    rows: list[dict[str, Any]],
    probabilities: dict[int, float],
    threshold: float,
    target_multiplier: float,
    min_keep: int,
    max_keep: int,
) -> set[str]:
    if not rows:
        return set()
    target = int(round(float(rows[0].get("target", 22)) * target_multiplier))
    target = max(min_keep, min(max_keep, target))
    ranked = sorted(
        rows,
        key=lambda r: (
            -probabilities[id(r)],
            -float(r.get("base_score", 0.0)),
            str(r.get("citation", "")),
        ),
    )
    selected: list[str] = []
    for row in ranked:
        prob = probabilities[id(row)]
        reasons = [str(x) for x in row.get("reasons", [])]
        explicit = any(r.split(":", 1)[0].startswith("explicit") for r in reasons)
        if explicit or prob >= threshold or len(selected) < max(min_keep, target // 2):
            selected.append(str(row["citation"]))
        if len(selected) >= target:
            break
    return set(selected)


def tune_thresholds(
    rows: list[dict[str, Any]],
    probabilities: dict[int, float],
    min_keep: int,
    max_keep: int,
) -> tuple[dict[str, float], dict[str, float], float]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(row)

    thresholds: dict[str, float] = {}
    multipliers: dict[str, float] = {}
    domain_order = ["default", "detention", "criminal", "social", "family", "inheritance", "obligations"]
    grid_t = [round(x, 2) for x in np.linspace(0.12, 0.82, 15)]
    grid_m = [0.80, 0.90, 1.00, 1.10, 1.20]

    for domain in domain_order:
        domain_qids = [
            qid for qid, qrows in by_query.items()
            if (domain_key(qrows[0]) == domain if domain != "default" else True)
        ]
        if not domain_qids:
            continue
        domain_groups = {qid: by_query[qid] for qid in domain_qids}
        best = (-1.0, 0.50, 1.0)
        for threshold in grid_t:
            for mult in grid_m:
                preds = {
                    qid: select_for_group(qrows, probabilities, threshold, mult, min_keep, max_keep)
                    for qid, qrows in domain_groups.items()
                }
                score = macro_f1_by_query(domain_groups, preds)
                if score > best[0]:
                    best = (score, threshold, mult)
        thresholds[domain] = best[1]
        multipliers[domain] = best[2]

    all_preds = {}
    for qid, qrows in by_query.items():
        d = domain_key(qrows[0])
        all_preds[qid] = select_for_group(
            qrows,
            probabilities,
            thresholds.get(d, thresholds.get("default", 0.50)),
            multipliers.get(d, multipliers.get("default", 1.0)),
            min_keep,
            max_keep,
        )
    return thresholds, multipliers, macro_f1_by_query(by_query, all_preds)


def train_sklearn(rows: list[dict[str, Any]], *, c_value: float) -> tuple[dict[str, float], float, dict[int, float]]:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover - fallback path
        raise RuntimeError(f"sklearn unavailable: {exc}") from exc

    feature_names = sorted({k for row in rows for k in row.get("features", {}).keys() if k != "bias"})
    x = np.asarray(
        [[float(row.get("features", {}).get(name, 0.0)) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        fit_intercept=True,
        max_iter=2000,
        random_state=20260520,
        solver="liblinear",
    )
    model.fit(x, y)
    weights = {name: float(coef) for name, coef in zip(feature_names, model.coef_[0])}
    bias = float(model.intercept_[0])
    probs = model.predict_proba(x)[:, 1]
    return weights, bias, {id(row): float(prob) for row, prob in zip(rows, probs)}


def fallback_weights(rows: list[dict[str, Any]]) -> tuple[dict[str, float], float, dict[int, float]]:
    weights = {
        "base_score": 0.55,
        "log_score": 0.65,
        "src_explicit": 4.0,
        "src_tfidf": 0.35,
        "src_dense_law": 0.40,
        "src_dense_law_chunk": 0.45,
        "src_dense_court": 0.28,
        "src_memory": 0.75,
        "src_statute": 0.20,
        "src_kit": 0.10,
        "src_graph": 0.18,
        "src_rerank": 0.45,
        "statute_match": 0.35,
        "inv_tfidf_rank": 0.80,
        "inv_dense_law_rank": 0.65,
        "inv_dense_law_chunk_rank": 0.70,
        "inv_dense_court_rank": 0.45,
        "inv_rerank_rank": 0.55,
    }
    bias = -2.0
    probs = {}
    for row in rows:
        feats = row.get("features", {})
        raw = bias + sum(weights.get(k, 0.0) * float(feats.get(k, 0.0)) for k in weights)
        raw = max(-50.0, min(50.0, raw))
        probs[id(row)] = 1.0 / (1.0 + math.exp(-raw))
    return weights, bias, probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    env_features = [
        Path(part).expanduser()
        for chunk in os.getenv("OFFLINE_CANDIDATE_FEATURES_PATH", "").split(os.pathsep)
        for part in chunk.split(",")
        if part.strip()
    ]
    parser.add_argument("--features-jsonl", type=Path, nargs="+", default=env_features)
    parser.add_argument("--gold-csv", type=Path, nargs="*", default=[REPO / "data" / "train.csv", REPO / "data" / "val.csv"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-keep", type=int, default=8)
    parser.add_argument("--max-keep", type=int, default=45)
    parser.add_argument("--max-negatives-per-query", type=int, default=300)
    parser.add_argument("--logreg-c", type=float, default=0.35)
    parser.add_argument("--no-sklearn", action="store_true")
    args = parser.parse_args()
    if not args.features_jsonl:
        raise SystemExit("Provide --features-jsonl or set OFFLINE_CANDIDATE_FEATURES_PATH.")

    gold = load_gold(args.gold_csv)
    rows, raw_rows = load_rows(
        args.features_jsonl,
        gold,
        max_negatives_per_query=args.max_negatives_per_query,
    )
    if not rows:
        raise SystemExit("No labeled candidate rows loaded.")
    if not args.max_negatives_per_query or args.max_negatives_per_query <= 0:
        raw_rows = len(rows)
    else:
        rows = downsample_rows(rows, None)
    positives = sum(int(row["label"]) for row in rows)
    if positives == 0:
        raise SystemExit("No positive labels found in candidate rows.")

    if args.no_sklearn:
        weights, bias, probabilities = fallback_weights(rows)
        model_type = "fallback_linear"
    else:
        try:
            weights, bias, probabilities = train_sklearn(rows, c_value=args.logreg_c)
            model_type = "sklearn_logistic_regression"
        except Exception as exc:
            print(f"[warn] sklearn training failed ({type(exc).__name__}: {exc}); using fallback weights", flush=True)
            weights, bias, probabilities = fallback_weights(rows)
            model_type = "fallback_linear"

    thresholds, target_multipliers, macro = tune_thresholds(rows, probabilities, args.min_keep, args.max_keep)
    payload = {
        "version": 1,
        "model_type": model_type,
        "created_by": "scripts/train_offline_selector.py",
        "rows": len(rows),
        "raw_rows": raw_rows,
        "positives": positives,
        "query_count": len({str(row["query_id"]) for row in rows}),
        "bias": bias,
        "weights": weights,
        "thresholds": thresholds,
        "target_multipliers": target_multipliers,
        "min_keep": args.min_keep,
        "max_keep": args.max_keep,
        "train_macro_f1": macro,
        "notes": "Trained only from public train/val feature dumps; no API calls and no hidden labels.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[selector] wrote {args.out} rows={len(rows):,} positives={positives:,} "
        f"queries={payload['query_count']} raw_rows={raw_rows:,} train_macro_f1={macro:.6f} model={model_type}",
        flush=True,
    )


if __name__ == "__main__":
    main()
