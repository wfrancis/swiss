#!/usr/bin/env python3
"""
Judge-softening ablation over the existing court-dense judged bundles.

Uses cached artifacts — NO new API calls. Tests whether the 9.01% val F1 from
the overnight court-dense sweep can be recovered by changing how the selector
uses judge labels, the selection budget, or the classifier itself.

Variants:
  1. repro           : reproduces overnight config as a control (should match ~9.0%)
  2. wider_budget    : same features, wider (min_out, max_out, target_mult) search
  3. no_judge_feats  : zero out judge_label one-hot + judge_confidence features
  4. force_must      : auto-include all must_include; fill remaining by classifier prob
  5. pure_final_score: ignore classifier entirely, rank by final_score with wide budget

Each variant is trained on train_v11_trainfit_local333_courtdense_restart1/judged_bundles.json
and evaluated on val_v11_courtdense_ds_val1/judged_bundles.json.

Outputs per variant:
  - prints train OOF F1, val apply F1, selected config
  - writes submissions/val_pred_ablation_<variant>.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# Reuse helpers from the main selector so we inherit any fixes.
from run_v11_meta_selector import (  # noqa: E402
    SelectorConfig,
    LABEL_NAMES,
    AUTO_BUCKET_NAMES,
    collect_source_names,
    f1_score,
    group_rows_by_query,
    load_bundles,
    load_gold,
    select_predictions,
    write_predictions,
)


# ---------- feature builders (masked variants) ----------------------------


def candidate_features_full(
    candidate: dict[str, Any],
    estimated_count: int,
    source_names: list[str],
) -> np.ndarray:
    features: list[float] = [
        1.0 if candidate["kind"] == "law" else 0.0,
        1.0 if candidate["kind"] == "court" else 0.0,
    ]
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


def candidate_features_no_judge(
    candidate: dict[str, Any],
    estimated_count: int,
    source_names: list[str],
) -> np.ndarray:
    """Judge label one-hot zeroed out, judge_confidence zeroed out."""
    features: list[float] = [
        1.0 if candidate["kind"] == "law" else 0.0,
        1.0 if candidate["kind"] == "court" else 0.0,
    ]
    features.extend(0.0 for _ in LABEL_NAMES)  # MASK judge labels

    auto_bucket = candidate["auto_bucket"] or "other"
    features.extend(1.0 if auto_bucket == name else 0.0 for name in AUTO_BUCKET_NAMES)

    features.extend(
        [
            0.0,  # MASK judge confidence
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


# ---------- row building --------------------------------------------------


def build_rows(
    bundles: list[dict[str, Any]],
    gold_map: dict[str, set[str]] | None,
    source_names: list[str],
    feature_fn: Callable[..., np.ndarray],
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
                    "features": feature_fn(candidate, bundle["estimated_count"], source_names),
                    "label": int(candidate["citation"] in gold) if gold_map is not None else 0,
                    "judge_label": candidate.get("judge_label"),
                    "final_score": float(candidate.get("final_score", 0.0)),
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


def predict_rows_oof(rows: list[dict[str, Any]], folds: int, seed: int) -> None:
    query_ids = sorted({row["query_id"] for row in rows})
    if len(query_ids) < 2:
        model = fit_model(rows, seed=seed)
        probs = model.predict_proba(np.stack([row["features"] for row in rows]))[:, 1]
        for row, p in zip(rows, probs):
            row["prob"] = float(p)
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
        probs[test_idx] = model.predict_proba(x[test_idx])[:, 1]
    for row, p in zip(rows, probs):
        row["prob"] = float(p)


def predict_rows_final(rows: list[dict[str, Any]], model: ExtraTreesClassifier) -> None:
    x = np.stack([row["features"] for row in rows])
    probs = model.predict_proba(x)[:, 1]
    for row, p in zip(rows, probs):
        row["prob"] = float(p)


# ---------- selection policies -------------------------------------------


def select_force_must(rows: list[dict[str, Any]], cfg: SelectorConfig) -> set[str]:
    """Auto-include every must_include, then fill to target by classifier prob."""
    target = round(rows[0]["estimated_count"] * cfg.target_mult + cfg.bias)
    target = max(cfg.min_out, min(cfg.max_out, target))

    must_rows = [r for r in rows if r["judge_label"] == "must_include"]
    rest_rows = [r for r in rows if r["judge_label"] != "must_include"]

    selected: list[str] = []
    selected_set: set[str] = set()
    courts = 0
    court_cap = max(0, round(target * cfg.court_cap_frac))

    # First pass: seed with must_include sorted by classifier prob (still cap courts)
    must_sorted = sorted(must_rows, key=lambda r: (-r["prob"], r["citation"]))
    for row in must_sorted:
        if row["citation"] in selected_set:
            continue
        if row["kind"] == "court" and courts >= court_cap:
            continue
        selected.append(row["citation"])
        selected_set.add(row["citation"])
        if row["kind"] == "court":
            courts += 1
        if len(selected) >= target:
            break

    # Second pass: fill with non-must by classifier prob
    if len(selected) < target:
        rest_sorted = sorted(rest_rows, key=lambda r: (-r["prob"], r["citation"]))
        for row in rest_sorted:
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


def select_pure_final_score(rows: list[dict[str, Any]], cfg: SelectorConfig) -> set[str]:
    """Ignore classifier; rank by final_score; respect target/cap/threshold on score."""
    target = round(rows[0]["estimated_count"] * cfg.target_mult + cfg.bias)
    target = max(cfg.min_out, min(cfg.max_out, target))

    ranked = sorted(rows, key=lambda r: (-r["final_score"], r["citation"]))
    selected: list[str] = []
    selected_set: set[str] = set()
    courts = 0
    court_cap = max(0, round(target * cfg.court_cap_frac))

    for row in ranked:
        if len(selected) >= target:
            break
        if row["final_score"] < cfg.thresh and len(selected) >= cfg.min_out:
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


SELECTION_POLICIES: dict[str, Callable[..., set[str]]] = {
    "prob": select_predictions,
    "force_must": select_force_must,
    "final_score": select_pure_final_score,
}


# ---------- random search over wider budgets ----------------------------


def random_search(
    rows: list[dict[str, Any]],
    gold_map: dict[str, set[str]],
    iterations: int,
    seed: int,
    target_mults: list[float],
    biases: list[int],
    min_outs: list[int],
    max_outs: list[int],
    thresholds: list[float],
    court_caps: list[float],
    policy: str,
) -> SelectorConfig:
    rng = random.Random(seed)
    policy_fn = SELECTION_POLICIES[policy]
    best = -1.0
    best_cfg: SelectorConfig | None = None
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
        grouped = group_rows_by_query(rows)
        scores = [
            f1_score(policy_fn(qrows, cfg), gold_map[qid])
            for qid, qrows in grouped.items()
            if qid in gold_map
        ]
        if not scores:
            continue
        macro = sum(scores) / len(scores)
        if macro > best:
            best = macro
            best_cfg = cfg
    if best_cfg is None:
        raise RuntimeError(f"random_search[{policy}] found no config")
    return best_cfg


def evaluate(
    rows: list[dict[str, Any]],
    gold_map: dict[str, set[str]],
    cfg: SelectorConfig,
    policy: str,
) -> tuple[float, dict[str, float], dict[str, set[str]]]:
    policy_fn = SELECTION_POLICIES[policy]
    grouped = group_rows_by_query(rows)
    scores: dict[str, float] = {}
    preds: dict[str, set[str]] = {}
    for qid, qrows in grouped.items():
        pred = policy_fn(qrows, cfg)
        preds[qid] = pred
        scores[qid] = f1_score(pred, gold_map.get(qid, set()))
    macro = sum(scores.values()) / len(scores) if scores else 0.0
    return macro, scores, preds


# ---------- variants ------------------------------------------------------


def variant_repro(train_rows, train_gold, val_rows, val_gold, iters, seed):
    # Current overnight ranges
    cfg = random_search(
        train_rows, train_gold, iters, seed,
        target_mults=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7],
        biases=[-8, -6, -4, -2, 0, 2],
        min_outs=[2, 4, 6, 8],
        max_outs=[8, 10, 12, 14, 16, 18, 20],
        thresholds=[0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        court_caps=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
        policy="prob",
    )
    return cfg, "prob"


def variant_wider(train_rows, train_gold, val_rows, val_gold, iters, seed):
    cfg = random_search(
        train_rows, train_gold, iters, seed,
        target_mults=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        biases=[-6, -4, -2, 0, 2, 4, 6],
        min_outs=[4, 6, 8, 10, 12, 14],
        max_outs=[14, 16, 18, 20, 22, 25, 28, 30, 35],
        thresholds=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25],
        court_caps=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        policy="prob",
    )
    return cfg, "prob"


def variant_force_must(train_rows, train_gold, val_rows, val_gold, iters, seed):
    cfg = random_search(
        train_rows, train_gold, iters, seed,
        target_mults=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        biases=[-6, -4, -2, 0, 2, 4, 6],
        min_outs=[4, 6, 8, 10, 12, 14],
        max_outs=[14, 16, 18, 20, 22, 25, 28, 30, 35],
        thresholds=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25],
        court_caps=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        policy="force_must",
    )
    return cfg, "force_must"


def variant_pure_final_score(train_rows, train_gold, val_rows, val_gold, iters, seed):
    cfg = random_search(
        train_rows, train_gold, iters, seed,
        target_mults=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        biases=[-4, -2, 0, 2, 4, 6],
        min_outs=[4, 6, 8, 10, 12],
        max_outs=[14, 16, 18, 20, 22, 25, 28, 30],
        # final_score ranges ~0..1; use score-space thresholds
        thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        court_caps=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        policy="final_score",
    )
    return cfg, "final_score"


VARIANTS = {
    "repro": (candidate_features_full, variant_repro),
    "wider_budget": (candidate_features_full, variant_wider),
    "no_judge_feats": (candidate_features_no_judge, variant_wider),
    "force_must": (candidate_features_full, variant_force_must),
    "pure_final_score": (candidate_features_full, variant_pure_final_score),
}


# ---------- runner --------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-judged", type=Path, required=True)
    ap.add_argument("--train-gold", type=Path, required=True)
    ap.add_argument("--val-judged", type=Path, required=True)
    ap.add_argument("--val-gold", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=BASE / "submissions")
    ap.add_argument("--random-search", type=int, default=400)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
    )
    args = ap.parse_args()

    train_bundles = load_bundles([args.train_judged])
    val_bundles = load_bundles([args.val_judged])
    train_gold = load_gold(args.train_gold)
    val_gold = load_gold(args.val_gold)

    source_names = collect_source_names(train_bundles, val_bundles)

    results: list[dict[str, Any]] = []
    for variant in args.variants:
        feature_fn, run_fn = VARIANTS[variant]
        print(f"\n=== Variant: {variant} ===", flush=True)

        train_rows = build_rows(train_bundles, train_gold, source_names, feature_fn)
        val_rows = build_rows(val_bundles, val_gold, source_names, feature_fn)

        # pure_final_score doesn't need classifier training/prediction
        policy_only = variant == "pure_final_score"
        if policy_only:
            # still fill a 'prob' field so downstream code doesn't choke
            for r in train_rows + val_rows:
                r["prob"] = r["final_score"]
        else:
            predict_rows_oof(train_rows, folds=args.folds, seed=args.seed)
            model = fit_model(train_rows, seed=args.seed)
            predict_rows_final(val_rows, model)

        cfg, policy = run_fn(train_rows, train_gold, val_rows, val_gold, args.random_search, args.seed)
        train_macro, _, _ = evaluate(train_rows, train_gold, cfg, policy)
        val_macro, val_scores, val_preds = evaluate(val_rows, val_gold, cfg, policy)

        print(f"  best cfg: {cfg.to_dict()}  policy={policy}", flush=True)
        print(f"  train OOF F1: {train_macro:.4%}", flush=True)
        print(f"  val F1:       {val_macro:.4%}", flush=True)
        for qid in sorted(val_scores):
            print(f"    {qid}: {val_scores[qid]:.2%}", flush=True)

        # save predictions
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_csv = args.output_dir / f"val_pred_ablation_{variant}.csv"
        write_predictions(out_csv, val_preds)

        results.append({
            "variant": variant,
            "train_f1": train_macro,
            "val_f1": val_macro,
            "policy": policy,
            "config": cfg.to_dict(),
            "output": str(out_csv),
        })

    print("\n=== ABLATION SUMMARY ===", flush=True)
    print(f"{'variant':<22} {'train_f1':>10} {'val_f1':>10}  policy", flush=True)
    for r in results:
        print(
            f"{r['variant']:<22} {r['train_f1']:>10.2%} {r['val_f1']:>10.2%}  {r['policy']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
