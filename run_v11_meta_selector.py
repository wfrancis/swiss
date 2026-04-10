#!/usr/bin/env python3
"""
Train and apply a lightweight meta-selector on top of judged V11 bundles.

This keeps the stable V11 retrieval/judge pipeline untouched while letting us
experiment quickly with learned selection policies from cached judged artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold


BASE = Path(__file__).resolve().parent

LABEL_NAMES = ["reject", "plausible", "must_include"]
AUTO_BUCKET_NAMES = ["auto_drop", "auto_keep", "other"]


@dataclass
class SelectorConfig:
    target_mult: float
    bias: int
    min_out: int
    max_out: int
    thresh: float
    court_cap_frac: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_mult": self.target_mult,
            "bias": self.bias,
            "min_out": self.min_out,
            "max_out": self.max_out,
            "thresh": self.thresh,
            "court_cap_frac": self.court_cap_frac,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-judged", type=Path, nargs="+", required=True)
    parser.add_argument("--train-gold", type=Path, required=True)
    parser.add_argument("--apply-judged", type=Path, nargs="+")
    parser.add_argument("--apply-gold", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--config-out", type=Path)
    parser.add_argument("--random-search", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--evaluate-loo", action="store_true")
    return parser.parse_args()


def load_gold(gold_path: Path) -> dict[str, set[str]]:
    with gold_path.open() as f:
        return {
            row["query_id"]: set(row["gold_citations"].split(";"))
            for row in csv.DictReader(f)
        }


def load_bundles(paths: list[Path]) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text())
        for bundle in payload["bundles"]:
            query_id = bundle["query_id"]
            if query_id in seen_query_ids:
                raise ValueError(f"duplicate query_id across judged artifacts: {query_id}")
            seen_query_ids.add(query_id)
            bundles.append(bundle)
    return bundles


def collect_source_names(*bundle_sets: list[dict[str, Any]]) -> list[str]:
    names = set()
    for bundles in bundle_sets:
        for bundle in bundles:
            for candidate in bundle["candidates"]:
                names.update(candidate["sources"])
    return sorted(names)


def candidate_features(
    candidate: dict[str, Any],
    estimated_count: int,
    source_names: list[str],
) -> np.ndarray:
    features: list[float] = []
    features.extend(
        [
            1.0 if candidate["kind"] == "law" else 0.0,
            1.0 if candidate["kind"] == "court" else 0.0,
        ]
    )
    features.extend(1.0 if candidate["judge_label"] == label else 0.0 for label in LABEL_NAMES)

    auto_bucket = candidate["auto_bucket"] or "other"
    features.extend(1.0 if auto_bucket == name else 0.0 for name in AUTO_BUCKET_NAMES)

    features.extend(
        [
            float(candidate["judge_confidence"]),
            float(candidate["final_score"]),
            float(candidate["raw_score"]),
            float(candidate["gpt_full_freq"]),
            float(candidate["is_explicit"]),
            float(candidate["is_query_case"]),
        ]
    )

    for key in ["dense_rank", "bm25_rank", "court_dense_rank"]:
        rank = candidate.get(key)
        features.append(0.0 if rank is None else 1.0 / (1.0 + float(rank)))

    source_set = set(candidate["sources"])
    features.extend(1.0 if source in source_set else 0.0 for source in source_names)
    features.append(float(len(source_set)))
    features.append(float(estimated_count))
    return np.array(features, dtype=float)


def build_rows(
    bundles: list[dict[str, Any]],
    gold_map: dict[str, set[str]] | None,
    source_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for bundle in bundles:
        gold = gold_map.get(bundle["query_id"], set()) if gold_map is not None else set()
        for candidate in bundle["candidates"]:
            rows.append(
                {
                    "query_id": bundle["query_id"],
                    "citation": candidate["citation"],
                    "kind": candidate["kind"],
                    "estimated_count": bundle["estimated_count"],
                    "features": candidate_features(candidate, bundle["estimated_count"], source_names),
                    "label": int(candidate["citation"] in gold) if gold_map is not None else 0,
                }
            )
    return rows


def fit_model(rows: list[dict[str, Any]], seed: int = 0) -> ExtraTreesClassifier:
    x = np.stack([row["features"] for row in rows])
    y = np.array([row["label"] for row in rows])
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
    x = np.stack([row["features"] for row in rows])
    probs = model.predict_proba(x)[:, 1]
    for row, prob in zip(rows, probs):
        row["prob"] = float(prob)


def predict_rows_oof(rows: list[dict[str, Any]], folds: int, seed: int) -> None:
    query_ids = sorted({row["query_id"] for row in rows})
    if len(query_ids) < 2:
        model = fit_model(rows, seed=seed)
        predict_rows(model, rows)
        return

    x = np.stack([row["features"] for row in rows])
    y = np.array([row["label"] for row in rows])
    groups = np.array([row["query_id"] for row in rows])
    n_splits = max(2, min(folds, len(query_ids)))
    splitter = GroupKFold(n_splits=n_splits)
    probs = np.zeros(len(rows), dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=4,
            n_jobs=-1,
            class_weight="balanced",
            random_state=seed + fold_idx,
        )
        model.fit(x[train_idx], y[train_idx])
        fold_probs = model.predict_proba(x[test_idx])[:, 1]
        probs[test_idx] = fold_probs

    for row, prob in zip(rows, probs):
        row["prob"] = float(prob)


def f1_score(pred: set[str], gold: set[str]) -> float:
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def group_rows_by_query(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(row)
    return grouped


def select_predictions(rows: list[dict[str, Any]], cfg: SelectorConfig) -> set[str]:
    ranked = sorted(rows, key=lambda row: (-row["prob"], row["citation"]))
    target = round(rows[0]["estimated_count"] * cfg.target_mult + cfg.bias)
    target = max(cfg.min_out, target)
    target = min(cfg.max_out, target)

    selected: list[str] = []
    selected_set: set[str] = set()
    court_cap = max(0, round(target * cfg.court_cap_frac))
    courts = 0

    for row in ranked:
        if len(selected) >= target:
            break
        if row["prob"] < cfg.thresh and len(selected) >= cfg.min_out:
            break
        if row["citation"] in selected_set:
            continue
        if row["kind"] == "court" and courts >= court_cap:
            continue
        selected.append(row["citation"])
        selected_set.add(row["citation"])
        if row["kind"] == "court":
            courts += 1

    return selected_set


def evaluate_rows(
    rows: list[dict[str, Any]],
    gold_map: dict[str, set[str]],
    cfg: SelectorConfig,
) -> tuple[float, dict[str, float], dict[str, set[str]]]:
    grouped = group_rows_by_query(rows)
    by_query_scores: dict[str, float] = {}
    predictions: dict[str, set[str]] = {}
    for query_id, query_rows in grouped.items():
        pred = select_predictions(query_rows, cfg)
        predictions[query_id] = pred
        by_query_scores[query_id] = f1_score(pred, gold_map[query_id])
    macro_f1 = sum(by_query_scores.values()) / len(by_query_scores) if by_query_scores else 0.0
    return macro_f1, by_query_scores, predictions


def random_search(
    rows: list[dict[str, Any]],
    gold_map: dict[str, set[str]],
    iterations: int,
    seed: int,
) -> SelectorConfig:
    rng = random.Random(seed)
    best_score = -1.0
    best_cfg: SelectorConfig | None = None

    target_mults = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
    biases = [-8, -6, -4, -2, 0, 2]
    min_outs = [2, 4, 6, 8]
    max_outs = [8, 10, 12, 14, 16, 18, 20]
    thresholds = [0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    court_caps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

    for _ in range(iterations):
        cfg = SelectorConfig(
            target_mult=rng.choice(target_mults),
            bias=rng.choice(biases),
            min_out=rng.choice(min_outs),
            max_out=rng.choice(max_outs),
            thresh=rng.choice(thresholds),
            court_cap_frac=rng.choice(court_caps),
        )
        if cfg.max_out < cfg.min_out:
            continue
        score, _, _ = evaluate_rows(rows, gold_map, cfg)
        if score > best_score:
            best_score = score
            best_cfg = cfg
            print(f"New best macro F1: {score:.4%} with {cfg.to_dict()}", flush=True)

    if best_cfg is None:
        raise RuntimeError("random_search failed to find a config")
    return best_cfg


def evaluate_leave_one_query_out(
    train_bundles: list[dict[str, Any]],
    gold_map: dict[str, set[str]],
    source_names: list[str],
    cfg: SelectorConfig,
) -> float:
    query_ids = sorted(bundle["query_id"] for bundle in train_bundles)
    by_query = {bundle["query_id"]: bundle for bundle in train_bundles}

    scores = []
    for held_query_id in query_ids:
        fit_bundles = [by_query[qid] for qid in query_ids if qid != held_query_id]
        held_bundles = [by_query[held_query_id]]
        fit_rows = build_rows(fit_bundles, gold_map, source_names)
        held_rows = build_rows(held_bundles, gold_map, source_names)
        model = fit_model(fit_rows)
        predict_rows(model, held_rows)
        score, _, _ = evaluate_rows(held_rows, gold_map, cfg)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def write_predictions(output_csv: Path, predictions: dict[str, set[str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\r\n")
        writer.writerow(["query_id", "predicted_citations"])
        for query_id in sorted(predictions):
            writer.writerow([query_id, ";".join(sorted(predictions[query_id]))])


def main() -> None:
    args = parse_args()
    apply_judged = args.apply_judged or args.train_judged
    apply_gold = args.apply_gold

    train_bundles = load_bundles(args.train_judged)
    apply_bundles = load_bundles(apply_judged)
    train_gold = load_gold(args.train_gold)
    apply_gold_map = load_gold(apply_gold) if apply_gold else None

    source_names = collect_source_names(train_bundles, apply_bundles)
    train_rows = build_rows(train_bundles, train_gold, source_names)
    predict_rows_oof(train_rows, folds=args.folds, seed=args.seed)

    print(
        f"Training rows: {len(train_rows):,} candidates across {len(train_bundles)} queries",
        flush=True,
    )

    best_cfg = random_search(train_rows, train_gold, iterations=args.random_search, seed=args.seed)
    train_macro, train_scores, _ = evaluate_rows(train_rows, train_gold, best_cfg)
    print(f"OOF train macro F1: {train_macro:.4%}", flush=True)
    for query_id in sorted(train_scores):
        print(f"  {query_id}: {train_scores[query_id]:.4%}", flush=True)

    if args.evaluate_loo:
        loo_macro = evaluate_leave_one_query_out(train_bundles, train_gold, source_names, best_cfg)
        print(f"Leave-one-query-out macro F1: {loo_macro:.4%}", flush=True)

    model = fit_model(train_rows, seed=args.seed)
    apply_rows = build_rows(apply_bundles, apply_gold_map, source_names)
    predict_rows(model, apply_rows)

    if apply_gold_map is not None:
        macro_f1, by_query_scores, predictions = evaluate_rows(apply_rows, apply_gold_map, best_cfg)
        print(f"Apply macro F1: {macro_f1:.4%}", flush=True)
        for query_id in sorted(by_query_scores):
            print(f"  {query_id}: {by_query_scores[query_id]:.4%}", flush=True)
    else:
        grouped = group_rows_by_query(apply_rows)
        predictions = {
            query_id: select_predictions(query_rows, best_cfg)
            for query_id, query_rows in grouped.items()
        }

    write_predictions(args.output_csv, predictions)
    print(f"Wrote predictions to {args.output_csv}", flush=True)

    if args.model_out:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        with args.model_out.open("wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "source_names": source_names,
                    "selector_config": best_cfg.to_dict(),
                },
                f,
            )
        print(f"Saved model bundle to {args.model_out}", flush=True)

    if args.config_out:
        args.config_out.parent.mkdir(parents=True, exist_ok=True)
        args.config_out.write_text(json.dumps(best_cfg.to_dict(), indent=2))
        print(f"Saved selector config to {args.config_out}", flush=True)


if __name__ == "__main__":
    main()
