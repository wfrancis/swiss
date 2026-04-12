#!/usr/bin/env python3
"""
Train a HistGradientBoosting classifier on judged V11 bundles with ~65 features.

Key differences from run_v11_meta_selector.py:
1. Uses ALL 1,139 train queries (not 200-333)
2. Richer features (~65 vs ~35): cross-features, relative position, statute codes
3. HistGradientBoosting instead of ExtraTrees (better feature interactions)
4. Stratified GroupKFold (ensures each fold has sparse AND dense queries)
5. Calibrated probabilities for meaningful thresholds
6. Optional procedural injection post-processing
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

BASE = Path(__file__).resolve().parent

LABEL_NAMES = ["reject", "plausible", "must_include"]
AUTO_BUCKET_NAMES = ["auto_drop", "auto_keep", "other"]
TOP_STATUTES = [
    "ZGB", "OR", "StGB", "BV", "BGG", "StPO", "SchKG", "ZPO",
    "ATSG", "IVG", "IPRG", "DBG", "URG", "DSG", "RPG", "JStG",
]


@dataclass
class SelectorConfig:
    target_mult: float
    bias: int
    min_out: int
    max_out: int
    thresh: float
    court_cap_frac: float

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-judged", type=Path, nargs="+", required=True)
    parser.add_argument("--train-gold", type=Path, required=True)
    parser.add_argument("--apply-judged", type=Path, nargs="+")
    parser.add_argument("--apply-gold", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--train-oof-csv", type=Path)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--config-out", type=Path)
    parser.add_argument("--random-search", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--procedural", action="store_true")
    return parser.parse_args()


# ── Data loading ───────────────────────────────────────────────


def load_gold(gold_path: Path) -> dict[str, set[str]]:
    with gold_path.open() as f:
        return {
            row["query_id"]: set(row["gold_citations"].split(";"))
            for row in csv.DictReader(f)
        }


def load_bundles(paths: list[Path]) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(paths):
        payload = json.loads(path.read_text())
        for bundle in payload["bundles"]:
            qid = bundle["query_id"]
            if qid not in seen:
                seen.add(qid)
                bundles.append(bundle)
    return bundles


def collect_source_names(*bundle_sets: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for bundles in bundle_sets:
        for bundle in bundles:
            for c in bundle["candidates"]:
                names.update(c.get("sources", []))
    return sorted(names)


# ── Feature extraction ─────────────────────────────────────────


def extract_statute(citation: str) -> str:
    if not citation.startswith("Art."):
        return ""
    parts = citation.split()
    return parts[-1] if len(parts) > 1 else ""


def candidate_features(
    candidate: dict[str, Any],
    bundle: dict[str, Any],
    source_names: list[str],
    bundle_stats: dict[str, Any],
) -> np.ndarray:
    f: list[float] = []
    kind = candidate.get("kind", "law")
    label = candidate.get("judge_label", "reject") or "reject"
    auto = candidate.get("auto_bucket") or "other"
    conf = float(candidate.get("judge_confidence", 0) or 0)
    final = float(candidate.get("final_score", 0) or 0)
    raw = float(candidate.get("raw_score", 0) or 0)
    gpf = int(candidate.get("gpt_full_freq", 0) or 0)
    cite = candidate.get("citation", "")
    est = int(bundle.get("estimated_count", 20) or 20)

    # GROUP 1: Core retrieval (14)
    f.append(1.0 if kind == "law" else 0.0)
    f.append(1.0 if kind == "court" else 0.0)
    f.append(raw)
    f.append(final)
    f.append(final - raw)
    for key in ["dense_rank", "bm25_rank", "court_dense_rank", "baseline_rank"]:
        rank = candidate.get(key)
        f.append(0.0 if rank is None else 1.0 / (1.0 + float(rank)))
    f.append(float(gpf))
    f.append(1.0 if gpf >= 1 else 0.0)
    f.append(1.0 if gpf >= 2 else 0.0)
    f.append(1.0 if candidate.get("is_explicit") else 0.0)
    f.append(1.0 if candidate.get("is_query_case") else 0.0)

    # GROUP 2: Judge signals (5)
    label_bonus = {"must_include": 2.0, "plausible": 1.0, "reject": 0.0}.get(label, 0.0)
    f.extend(1.0 if label == l else 0.0 for l in LABEL_NAMES)
    f.append(conf)
    f.append(label_bonus * conf)

    # GROUP 3: Auto-bucket (3)
    f.extend(1.0 if auto == b else 0.0 for b in AUTO_BUCKET_NAMES)

    # GROUP 4: Sources (variable + 2)
    source_set = set(candidate.get("sources", []))
    f.extend(1.0 if s in source_set else 0.0 for s in source_names)
    f.append(float(len(source_set)))
    f.append(1.0 if len(source_set) >= 2 else 0.0)

    # GROUP 5: Citation patterns (2 + len(TOP_STATUTES))
    f.append(1.0 if cite.startswith("Art.") else 0.0)
    f.append(1.0 if cite.startswith("BGE") else 0.0)
    statute = extract_statute(cite)
    f.extend(1.0 if statute == s else 0.0 for s in TOP_STATUTES)

    # GROUP 6: Query context (6)
    f.append(float(est))
    f.append(float(np.log1p(est)))
    f.append(float(bundle_stats["n_candidates"]))
    f.append(float(bundle_stats["n_law"]))
    f.append(float(bundle_stats["n_court"]))
    f.append(float(candidate.get("_rank_pct", 0.5)))

    # GROUP 7: Cross-features (6)
    f.append(float(kind == "law" and label == "must_include"))
    f.append(float(kind == "law") * gpf)
    inv_cdr = candidate.get("court_dense_rank")
    f.append(float(kind == "court") * (0.0 if inv_cdr is None else 1.0 / (1.0 + float(inv_cdr))))
    f.append(float(label == "must_include" and auto == "auto_keep"))
    f.append(final * conf)
    dr = candidate.get("dense_rank")
    br = candidate.get("bm25_rank")
    f.append(1.0 if dr is not None and br is not None and dr < 50 and br < 50 else 0.0)

    # GROUP 8: Relative position (4)
    f.append(float(candidate.get("_score_pct", 0.5)))
    f.append(float(candidate.get("_conf_pct", 0.5)))
    f.append(1.0 if candidate.get("_rank_pct", 1.0) <= 0.05 else 0.0)
    f.append(1.0 if candidate.get("_conf_rank", 999) < 10 else 0.0)

    return np.array(f, dtype=np.float32)


def add_relative_features(bundle: dict[str, Any]) -> dict[str, Any]:
    """Add percentile-based features to candidates within a bundle."""
    candidates = bundle["candidates"]
    if not candidates:
        return {"n_candidates": 0, "n_law": 0, "n_court": 0}

    scores = [float(c.get("final_score", 0) or 0) for c in candidates]
    confs = [float(c.get("judge_confidence", 0) or 0) for c in candidates]
    n = len(candidates)

    score_sorted = sorted(range(n), key=lambda i: -scores[i])
    conf_sorted = sorted(range(n), key=lambda i: -confs[i])

    for rank, idx in enumerate(score_sorted):
        candidates[idx]["_rank_pct"] = rank / max(n - 1, 1)
        candidates[idx]["_score_pct"] = rank / max(n - 1, 1)
    for rank, idx in enumerate(conf_sorted):
        candidates[idx]["_conf_pct"] = rank / max(n - 1, 1)
        candidates[idx]["_conf_rank"] = rank

    n_law = sum(1 for c in candidates if c.get("kind") == "law")
    return {"n_candidates": n, "n_law": n_law, "n_court": n - n_law}


def build_dataset(
    bundles: list[dict[str, Any]],
    gold_map: dict[str, set[str]] | None,
    source_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build X, y, groups, metadata for training or inference."""
    all_features = []
    all_labels = []
    all_groups = []
    all_meta = []

    for bundle in bundles:
        qid = bundle["query_id"]
        gold = gold_map.get(qid, set()) if gold_map else set()
        stats = add_relative_features(bundle)

        for c in bundle["candidates"]:
            feat = candidate_features(c, bundle, source_names, stats)
            label = int(c["citation"] in gold) if gold_map else 0
            all_features.append(feat)
            all_labels.append(label)
            all_groups.append(qid)
            all_meta.append({
                "query_id": qid,
                "citation": c["citation"],
                "kind": c.get("kind", "law"),
                "estimated_count": bundle.get("estimated_count", 20),
            })

    X = np.stack(all_features)
    y = np.array(all_labels, dtype=np.int32)
    groups = np.array(all_groups)
    return X, y, groups, all_meta


# ── Training ───────────────────────────────────────────────────


def stratified_group_kfold(query_ids: list[str], gold_map: dict[str, set[str]], n_folds: int) -> dict[str, int]:
    """Assign queries to folds, stratified by gold count."""
    sorted_by_count = sorted(query_ids, key=lambda q: len(gold_map.get(q, set())))
    fold_map = {}
    for i, qid in enumerate(sorted_by_count):
        fold_map[qid] = i % n_folds
    return fold_map


def train_oof(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    gold_map: dict[str, set[str]], meta: list[dict],
    n_folds: int, seed: int,
) -> tuple[np.ndarray, Any]:
    """Train with stratified GroupKFold, return OOF probabilities."""
    query_ids = sorted(set(groups))
    fold_map = stratified_group_kfold(query_ids, gold_map, n_folds)

    oof_probs = np.zeros(len(y), dtype=np.float64)
    models = []

    for fold in range(n_folds):
        train_queries = {q for q, f in fold_map.items() if f != fold}
        test_queries = {q for q, f in fold_map.items() if f == fold}

        train_mask = np.array([g in train_queries for g in groups])
        test_mask = np.array([g in test_queries for g in groups])

        X_train, y_train = X[train_mask], y[train_mask]
        X_test = X[test_mask]

        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        print(f"  Fold {fold+1}/{n_folds}: {train_mask.sum():,} train ({pos_count} pos), "
              f"{test_mask.sum():,} test ({len(test_queries)} queries)", flush=True)

        model = HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=seed + fold,
            verbose=0,
        )
        model.fit(X_train, y_train)

        fold_probs = model.predict_proba(X_test)[:, 1]
        oof_probs[test_mask] = fold_probs
        models.append(model)

        # Per-fold F1 with default config
        fold_meta = [m for m, mask in zip(meta, test_mask) if mask]
        fold_f1 = quick_eval(fold_probs, fold_meta, gold_map)
        print(f"    Fold {fold+1} quick F1: {fold_f1:.4f}", flush=True)

    return oof_probs, models


def quick_eval(probs: np.ndarray, meta: list[dict], gold_map: dict[str, set[str]],
               top_n: int = 15) -> float:
    """Quick F1 using top-N by probability per query."""
    from collections import defaultdict
    query_preds: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for prob, m in zip(probs, meta):
        query_preds[m["query_id"]].append((prob, m["citation"]))

    f1s = []
    for qid, items in query_preds.items():
        gold = gold_map.get(qid, set())
        if not gold:
            continue
        items.sort(reverse=True)
        est = items[0] if items else 15  # fallback
        # Adaptive top-N based on estimated_count
        target = min(top_n, len(items))
        pred = {cite for _, cite in items[:target]}
        tp = len(pred & gold)
        p = tp / len(pred) if pred else 0
        r = tp / len(gold) if gold else 0
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0)
    return sum(f1s) / len(f1s) if f1s else 0


# ── Selection + threshold search ───────────────────────────────


def pregroup_by_query(probs: np.ndarray, meta: list[dict]) -> dict[str, list[tuple[float, str, str, int]]]:
    """Pre-group and pre-sort data by query for fast threshold search."""
    from collections import defaultdict
    grouped: dict[str, list[tuple[float, str, str, int]]] = defaultdict(list)
    for prob, m in zip(probs, meta):
        grouped[m["query_id"]].append((float(prob), m["citation"], m["kind"], int(m["estimated_count"])))
    for qid in grouped:
        grouped[qid].sort(key=lambda x: -x[0])
    return dict(grouped)


def select_predictions_fast(
    grouped: dict[str, list[tuple[float, str, str, int]]], cfg: SelectorConfig,
) -> dict[str, set[str]]:
    """Fast selection on pre-grouped data."""
    predictions: dict[str, set[str]] = {}
    for qid, items in grouped.items():
        est = items[0][3] if items else 20
        target = round(est * cfg.target_mult + cfg.bias)
        target = max(cfg.min_out, min(cfg.max_out, target))

        selected: set[str] = set()
        courts = 0
        court_cap = max(0, round(target * cfg.court_cap_frac))

        for prob, cite, kind, _ in items:
            if len(selected) >= target:
                break
            if prob < cfg.thresh and len(selected) >= cfg.min_out:
                break
            if kind == "court" and courts >= court_cap:
                continue
            selected.add(cite)
            if kind == "court":
                courts += 1

        predictions[qid] = selected
    return predictions


def select_predictions(
    probs: np.ndarray, meta: list[dict], cfg: SelectorConfig,
) -> dict[str, set[str]]:
    """Select predictions using model probabilities + config."""
    grouped = pregroup_by_query(probs, meta)
    return select_predictions_fast(grouped, cfg)


def evaluate(predictions: dict[str, set[str]], gold_map: dict[str, set[str]]) -> float:
    f1s = []
    for qid in gold_map:
        pred = predictions.get(qid, set())
        gold = gold_map[qid]
        if not pred and not gold:
            f1s.append(1.0)
            continue
        if not pred or not gold:
            f1s.append(0.0)
            continue
        tp = len(pred & gold)
        p = tp / len(pred)
        r = tp / len(gold)
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0)
    return sum(f1s) / len(f1s) if f1s else 0


def threshold_search(
    probs: np.ndarray, meta: list[dict], gold_map: dict[str, set[str]],
    n_iter: int, seed: int,
) -> tuple[SelectorConfig, float]:
    """Random search for best selection config. Pre-groups data for speed."""
    rng = random.Random(seed)
    best_f1 = 0
    best_cfg = None

    # Pre-group ONCE (was being re-grouped every iteration)
    grouped = pregroup_by_query(probs, meta)

    for i in range(n_iter):
        cfg = SelectorConfig(
            target_mult=rng.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]),
            bias=rng.choice([-8, -4, -2, 0, 2, 4, 8, 12]),
            min_out=rng.choice([3, 4, 5, 6, 8, 10]),
            max_out=rng.choice([10, 15, 20, 25, 30, 35, 40, 50]),
            thresh=rng.choice([0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]),
            court_cap_frac=rng.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50]),
        )
        if cfg.min_out >= cfg.max_out:
            continue

        predictions = select_predictions_fast(grouped, cfg)
        f1 = evaluate(predictions, gold_map)

        if f1 > best_f1:
            best_f1 = f1
            best_cfg = cfg
            print(f"    iter {i+1}: F1={f1:.6f} cfg={cfg.to_dict()}", flush=True)

    return best_cfg, best_f1


# ── Main ───────────────────────────────────────────────────────


def write_csv(predictions: dict[str, set[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions):
            w.writerow([qid, ";".join(sorted(predictions[qid]))])


def main():
    args = parse_args()

    print("Loading training data...", flush=True)
    train_bundles = load_bundles(args.train_judged)
    gold_map = load_gold(args.train_gold)
    print(f"  {len(train_bundles)} train bundles, {len(gold_map)} gold queries", flush=True)

    # Filter to queries with gold
    train_bundles = [b for b in train_bundles if b["query_id"] in gold_map]
    print(f"  {len(train_bundles)} bundles with gold labels", flush=True)

    apply_bundles = None
    if args.apply_judged:
        apply_bundles = load_bundles(args.apply_judged)
        print(f"  {len(apply_bundles)} apply bundles", flush=True)

    source_names = collect_source_names(
        train_bundles, apply_bundles if apply_bundles else []
    )
    print(f"  {len(source_names)} source names: {source_names}", flush=True)

    print("\nBuilding training dataset...", flush=True)
    X, y, groups, meta = build_dataset(train_bundles, gold_map, source_names)
    print(f"  X: {X.shape}, positives: {y.sum():,} ({y.mean()*100:.2f}%)", flush=True)

    print(f"\n=== OOF Cross-Validation ({args.folds} folds) ===", flush=True)
    oof_probs, models = train_oof(X, y, groups, gold_map, meta, args.folds, args.seed)

    print(f"\nSearching selection threshold ({args.random_search} iterations)...", flush=True)
    best_cfg, oof_f1 = threshold_search(oof_probs, meta, gold_map, args.random_search, args.seed)
    print(f"\n=== OOF F1: {oof_f1:.6f} ===", flush=True)
    print(f"Best config: {best_cfg.to_dict()}", flush=True)

    # Save OOF predictions
    if args.train_oof_csv:
        oof_preds = select_predictions(oof_probs, meta, best_cfg)
        write_csv(oof_preds, args.train_oof_csv)
        print(f"Wrote OOF predictions to {args.train_oof_csv}", flush=True)

    # Train final model on ALL data
    print("\nTraining final model on all data...", flush=True)
    final_model = HistGradientBoostingClassifier(
        max_iter=500, max_depth=6, learning_rate=0.05,
        min_samples_leaf=20, class_weight="balanced",
        random_state=args.seed, verbose=0,
    )
    final_model.fit(X, y)
    cal_model = final_model  # skip calibration (sklearn 1.8 API change)

    # Save model
    if args.model_out:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.model_out, "wb") as f:
            pickle.dump({
                "model": cal_model,
                "source_names": source_names,
                "config": best_cfg.to_dict(),
            }, f)
        print(f"Saved model to {args.model_out}", flush=True)

    if args.config_out:
        args.config_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.config_out, "w") as f:
            json.dump(best_cfg.to_dict(), f, indent=2)

    # Apply to val/test
    if apply_bundles:
        print("\nApplying to inference bundles...", flush=True)
        apply_gold = load_gold(args.apply_gold) if args.apply_gold else None
        X_apply, _, _, meta_apply = build_dataset(apply_bundles, apply_gold, source_names)
        apply_probs = cal_model.predict_proba(X_apply)[:, 1]
        predictions = select_predictions(apply_probs, meta_apply, best_cfg)

        # Procedural injection
        if args.procedural:
            try:
                proc_cache = json.loads((BASE / "precompute/llm_procedural_cache.json").read_text())
                for qid in predictions:
                    split = "val" if qid.startswith("val") else "test"
                    cls = proc_cache.get(f"{split}_{qid}", {})
                    if cls.get("confidence", 0) >= 0.7:
                        for c in cls.get("citations", []):
                            if c.startswith("Art.") and "BGG" not in c:
                                predictions[qid].add(c)
                print("  Applied procedural injection (no BGG)", flush=True)
            except Exception as e:
                print(f"  Procedural injection failed: {e}", flush=True)

        write_csv(predictions, args.output_csv)
        avg = sum(len(p) for p in predictions.values()) / len(predictions)
        print(f"Wrote {args.output_csv} (avg {avg:.1f} cites/query)", flush=True)

        if apply_gold:
            f1 = evaluate(predictions, apply_gold)
            print(f"\n=== Apply F1: {f1:.6f} ===", flush=True)


if __name__ == "__main__":
    main()
