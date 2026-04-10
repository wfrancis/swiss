#!/usr/bin/env python3
"""
Train and apply a local-only selector on top of V11 candidate bundles.

This path does not require judged train artifacts or new API calls. It fits on
train gold labels using the existing local retrieval features already present in
candidate bundles, searches selection hyperparameters on out-of-fold train
predictions, then applies the fitted model to another candidate artifact.
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold


BASE = Path(__file__).resolve().parent

AUTO_BUCKET_NAMES = ["auto_keep", "auto_drop", "other"]


@dataclass
class SelectorConfig:
    target_mult: float
    bias: int
    min_out: int
    max_out: int
    thresh: float
    court_cap_frac: float
    keep_explicit: bool
    keep_auto_keep: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_mult": self.target_mult,
            "bias": self.bias,
            "min_out": self.min_out,
            "max_out": self.max_out,
            "thresh": self.thresh,
            "court_cap_frac": self.court_cap_frac,
            "keep_explicit": self.keep_explicit,
            "keep_auto_keep": self.keep_auto_keep,
        }


@dataclass
class CitationPriors:
    exact_freq: dict[str, int]
    law_base_freq: dict[str, int]
    statute_code_freq: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--train-gold", type=Path, required=True)
    parser.add_argument("--apply-candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--apply-gold", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--train-oof-csv", type=Path)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--config-out", type=Path)
    parser.add_argument("--random-search", type=int, default=3000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_gold(gold_path: Path) -> dict[str, set[str]]:
    with gold_path.open() as f:
        return {
            row["query_id"]: {part for part in row["gold_citations"].split(";") if part}
            for row in csv.DictReader(f)
        }


def load_bundle_payloads(paths: list[Path]) -> list[dict[str, Any]]:
    payloads = []
    seen_query_ids: set[str] = set()
    for path in paths:
        with path.open("rb") as f:
            payload = pickle.load(f)
        for bundle in payload["bundles"]:
            query_id = bundle.query_id
            if query_id in seen_query_ids:
                raise ValueError(f"duplicate query_id across bundle artifacts: {query_id}")
            seen_query_ids.add(query_id)
        payloads.append(payload)
    return payloads


def collect_source_names(*payload_sets: list[dict[str, Any]]) -> list[str]:
    names = set()
    for payloads in payload_sets:
        for payload in payloads:
            for bundle in payload["bundles"]:
                for candidate in bundle.candidates:
                    names.update(candidate.sources)
    return sorted(names)


def inverse_rank(rank: int | None) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / (1.0 + float(rank))


def law_base(citation: str) -> str:
    if not citation.startswith("Art. "):
        return citation
    match = re.match(r"^(Art\.\s+\d+[a-z]?)", citation)
    if not match:
        return citation
    statute_match = re.search(r"\s+([A-ZÄÖÜ]{2,}[A-Za-zÄÖÜäöü]*)$", citation)
    if statute_match:
        return f"{match.group(1)} {statute_match.group(1)}"
    return match.group(1)


def statute_code(citation: str) -> str:
    match = re.search(r"\s+([A-ZÄÖÜ]{2,}[A-Za-zÄÖÜäöü]*)$", citation)
    return match.group(1) if match else ""


def build_priors(gold_map: dict[str, set[str]], query_ids: set[str]) -> CitationPriors:
    exact = Counter()
    law_bases = Counter()
    statute_codes = Counter()
    for query_id in query_ids:
        for citation in gold_map.get(query_id, set()):
            exact[citation] += 1
            base = law_base(citation)
            if base != citation or citation.startswith("Art. "):
                law_bases[base] += 1
            code = statute_code(citation)
            if code:
                statute_codes[code] += 1
    return CitationPriors(
        exact_freq=dict(exact),
        law_base_freq=dict(law_bases),
        statute_code_freq=dict(statute_codes),
    )


def top_statute_codes(priors: CitationPriors, limit: int = 24) -> list[str]:
    return [
        code
        for code, _ in Counter(priors.statute_code_freq).most_common(limit)
        if code
    ]


def bundle_candidate_counts(bundle: Any) -> tuple[int, int]:
    law_count = 0
    court_count = 0
    for candidate in bundle.candidates:
        if candidate.kind == "law":
            law_count += 1
        else:
            court_count += 1
    return law_count, court_count


def candidate_features(
    bundle: Any,
    candidate: Any,
    source_names: list[str],
    priors: CitationPriors,
    known_codes: list[str],
) -> np.ndarray:
    source_set = set(candidate.sources)
    auto_bucket = candidate.auto_bucket or "other"
    law_count, court_count = bundle_candidate_counts(bundle)
    base = law_base(candidate.citation)
    code = statute_code(candidate.citation)
    features: list[float] = [
        1.0 if candidate.kind == "law" else 0.0,
        1.0 if candidate.kind == "court" else 0.0,
    ]
    features.extend(1.0 if auto_bucket == name else 0.0 for name in AUTO_BUCKET_NAMES)
    flat_features = [
        float(candidate.raw_score),
        float(candidate.final_score),
        float(candidate.final_score - candidate.raw_score),
        float(candidate.gpt_full_freq),
        float(candidate.gpt_full_freq >= 1),
        float(candidate.gpt_full_freq >= 2),
        float(candidate.gpt_full_freq >= 3),
        float(candidate.is_explicit),
        float(candidate.is_query_case),
        float(candidate.citation.startswith("Art. ")),
        float(candidate.citation.startswith("BGE ")),
        float(candidate.citation.startswith("BGer ")),
        float(" E. " in candidate.citation),
        inverse_rank(candidate.baseline_rank),
        inverse_rank(candidate.dense_rank),
        inverse_rank(candidate.bm25_rank),
        inverse_rank(candidate.court_dense_rank),
        float(len(source_set)),
        float(bundle.estimated_count),
        float(len(bundle.candidates)),
        float(law_count),
        float(court_count),
        float(priors.exact_freq.get(candidate.citation, 0)),
        float(priors.law_base_freq.get(base, 0)),
        float(priors.statute_code_freq.get(code, 0)),
    ]
    features = list(features)
    features.extend(flat_features)
    features.extend(1.0 if source in source_set else 0.0 for source in source_names)
    features.extend(1.0 if code == known_code else 0.0 for known_code in known_codes)
    return np.array(features, dtype=float)


def build_rows(
    payloads: list[dict[str, Any]],
    gold_map: dict[str, set[str]] | None,
    source_names: list[str],
    priors: CitationPriors,
    known_codes: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for payload in payloads:
        for bundle in payload["bundles"]:
            gold = gold_map.get(bundle.query_id, set()) if gold_map is not None else set()
            for candidate in bundle.candidates:
                rows.append(
                    {
                        "query_id": bundle.query_id,
                        "citation": candidate.citation,
                        "kind": candidate.kind,
                        "estimated_count": bundle.estimated_count,
                        "is_explicit": bool(candidate.is_explicit),
                        "is_auto_keep": candidate.auto_bucket == "auto_keep",
                        "baseline_rank": candidate.baseline_rank,
                        "final_score": candidate.final_score,
                        "features": candidate_features(bundle, candidate, source_names, priors, known_codes),
                        "label": int(candidate.citation in gold) if gold_map is not None else 0,
                    }
                )
    return rows


def fit_model(rows: list[dict[str, Any]], seed: int) -> ExtraTreesClassifier:
    x = np.stack([row["features"] for row in rows])
    y = np.array([row["label"] for row in rows])
    model = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=seed,
    )
    model.fit(x, y)
    return model


def predict_rows(model: ExtraTreesClassifier, rows: list[dict[str, Any]]) -> None:
    x = np.stack([row["features"] for row in rows])
    probs = model.predict_proba(x)[:, 1]
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
    ordered = sorted(
        rows,
        key=lambda row: (
            -row["prob"],
            -row["is_explicit"],
            -row["is_auto_keep"],
            -row["final_score"],
            row["baseline_rank"],
            row["citation"],
        ),
    )
    target = round(rows[0]["estimated_count"] * cfg.target_mult + cfg.bias)
    target = max(cfg.min_out, target)
    target = min(cfg.max_out, target)

    selected: list[str] = []
    selected_set: set[str] = set()
    court_cap = max(0, round(target * cfg.court_cap_frac))
    courts = 0

    if cfg.keep_explicit:
        for row in ordered:
            if row["is_explicit"] and row["citation"] not in selected_set:
                selected.append(row["citation"])
                selected_set.add(row["citation"])
                if row["kind"] == "court":
                    courts += 1

    if cfg.keep_auto_keep:
        for row in ordered:
            if row["is_auto_keep"] and row["citation"] not in selected_set:
                if row["kind"] == "court" and courts >= court_cap:
                    continue
                selected.append(row["citation"])
                selected_set.add(row["citation"])
                if row["kind"] == "court":
                    courts += 1
                if len(selected) >= target:
                    return selected_set

    for row in ordered:
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
) -> tuple[SelectorConfig, float]:
    rng = random.Random(seed)
    best_score = -1.0
    best_cfg: SelectorConfig | None = None

    target_mults = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    biases = [-12, -10, -8, -6, -4, -2, 0, 2]
    min_outs = [4, 6, 8, 10]
    max_outs = [10, 12, 14, 16, 18, 20, 24, 28, 32, 40]
    thresholds = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35]
    court_caps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.33]

    for _ in range(iterations):
        cfg = SelectorConfig(
            target_mult=rng.choice(target_mults),
            bias=rng.choice(biases),
            min_out=rng.choice(min_outs),
            max_out=rng.choice(max_outs),
            thresh=rng.choice(thresholds),
            court_cap_frac=rng.choice(court_caps),
            keep_explicit=rng.choice([False, True]),
            keep_auto_keep=rng.choice([False, True]),
        )
        score, _, _ = evaluate_rows(rows, gold_map, cfg)
        if score > best_score:
            best_score = score
            best_cfg = cfg

    if best_cfg is None:
        raise RuntimeError("random search failed to produce a selector config")
    return best_cfg, best_score


def macro_candidate_recall(
    payloads: list[dict[str, Any]],
    gold_map: dict[str, set[str]],
) -> float:
    recalls = []
    for payload in payloads:
        for bundle in payload["bundles"]:
            gold = gold_map.get(bundle.query_id, set())
            if not gold:
                continue
            candidates = {candidate.citation for candidate in bundle.candidates}
            recalls.append(len(candidates & gold) / len(gold))
    return sum(recalls) / len(recalls) if recalls else 0.0


def write_predictions_csv(path: Path, predictions: dict[str, set[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for query_id in sorted(predictions):
            writer.writerow([query_id, ";".join(sorted(predictions[query_id]))])


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_gold = load_gold(args.train_gold)
    apply_gold = load_gold(args.apply_gold) if args.apply_gold else None

    train_payloads = load_bundle_payloads(args.train_candidates)
    apply_payloads = load_bundle_payloads(args.apply_candidates)
    source_names = collect_source_names(train_payloads, apply_payloads)

    train_query_ids = {
        bundle.query_id
        for payload in train_payloads
        for bundle in payload["bundles"]
    }
    priors = build_priors(train_gold, train_query_ids)
    known_codes = top_statute_codes(priors)

    query_ids = sorted(train_query_ids)
    if len(query_ids) < 2:
        raise ValueError("need at least two train queries to fit and evaluate selector")
    n_splits = min(args.folds, len(query_ids))
    if n_splits < 2:
        raise ValueError("need at least two folds for out-of-fold evaluation")

    by_query_payload = {
        bundle.query_id: {"bundles": [bundle]}
        for payload in train_payloads
        for bundle in payload["bundles"]
    }
    grouped_rows: list[dict[str, Any]] = []
    oof_by_query: dict[str, list[dict[str, Any]]] = {}
    splitter = GroupKFold(n_splits=n_splits)

    print(
        (
            f"Train selector: queries={len(query_ids)}, folds={n_splits}, "
            f"candidate_recall={macro_candidate_recall(train_payloads, train_gold):.4f}"
        ),
        flush=True,
    )

    for fold_idx, (fit_idx, hold_idx) in enumerate(
        splitter.split(query_ids, groups=query_ids),
        start=1,
    ):
        fit_query_ids = {query_ids[i] for i in fit_idx}
        hold_query_ids = {query_ids[i] for i in hold_idx}
        fold_priors = build_priors(train_gold, fit_query_ids)
        fold_codes = top_statute_codes(fold_priors)
        fit_payloads = [by_query_payload[qid] for qid in sorted(fit_query_ids)]
        hold_payloads = [by_query_payload[qid] for qid in sorted(hold_query_ids)]
        fit_rows = build_rows(fit_payloads, train_gold, source_names, fold_priors, fold_codes)
        hold_rows = build_rows(hold_payloads, train_gold, source_names, fold_priors, fold_codes)
        model = fit_model(fit_rows, args.seed + fold_idx)
        predict_rows(model, hold_rows)
        grouped_rows.extend(hold_rows)
        hold_grouped = group_rows_by_query(hold_rows)
        oof_by_query.update(hold_grouped)
        positives = sum(row["label"] for row in hold_rows)
        print(
            f"  fold {fold_idx}: fit_queries={len(fit_query_ids)}, hold_queries={len(hold_query_ids)}, "
            f"hold_rows={len(hold_rows)}, positives={positives}",
            flush=True,
        )

    selector_cfg, oof_score = random_search(grouped_rows, train_gold, args.random_search, args.seed)
    oof_macro, oof_by_query_scores, oof_predictions = evaluate_rows(grouped_rows, train_gold, selector_cfg)
    print(f"OOF selector macro F1: {oof_macro:.4f} ({oof_macro * 100:.2f}%)", flush=True)
    print(f"Best selector config: {selector_cfg.to_dict()}", flush=True)

    if args.train_oof_csv:
        write_predictions_csv(args.train_oof_csv, oof_predictions)

    final_rows = build_rows(train_payloads, train_gold, source_names, priors, known_codes)
    final_model = fit_model(final_rows, args.seed)

    apply_rows = build_rows(apply_payloads, apply_gold, source_names, priors, known_codes)
    predict_rows(final_model, apply_rows)
    apply_macro = None
    apply_predictions: dict[str, set[str]]
    apply_by_query_scores: dict[str, float] = {}
    if apply_gold is not None:
        apply_macro, apply_by_query_scores, apply_predictions = evaluate_rows(apply_rows, apply_gold, selector_cfg)
        print(f"Apply macro F1: {apply_macro:.4f} ({apply_macro * 100:.2f}%)", flush=True)
    else:
        apply_predictions = {}
        grouped = group_rows_by_query(apply_rows)
        for query_id, query_rows in grouped.items():
            apply_predictions[query_id] = select_predictions(query_rows, selector_cfg)

    write_predictions_csv(args.output_csv, apply_predictions)

    if args.model_out:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        with args.model_out.open("wb") as f:
            pickle.dump(final_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.config_out:
        args.config_out.parent.mkdir(parents=True, exist_ok=True)
        args.config_out.write_text(
            json.dumps(
                {
                    "train_query_count": len(query_ids),
                    "source_names": source_names,
                    "known_statute_codes": known_codes,
                    "selector_config": selector_cfg.to_dict(),
                    "oof_macro_f1": oof_macro,
                    "oof_by_query": oof_by_query_scores,
                    "apply_macro_f1": apply_macro,
                    "apply_by_query": apply_by_query_scores,
                    "candidate_recall_macro": macro_candidate_recall(train_payloads, train_gold),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
