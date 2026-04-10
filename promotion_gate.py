#!/usr/bin/env python3
"""
Predict whether a candidate is likely better or worse than the current best
public Kaggle submission using our real leaderboard history.

This intentionally focuses on winner-relative drift instead of raw val alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-val", type=Path, required=True)
    parser.add_argument("--candidate-test", type=Path, required=True)
    parser.add_argument(
        "--history-json",
        type=Path,
        default=Path("artifacts/v11_meta/kaggle_public_history.json"),
    )
    parser.add_argument("--gold-csv", type=Path, default=Path("data/val.csv"))
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def parse_citations(value: str) -> set[str]:
    if not value or not value.strip():
        return set()
    return {piece.strip() for piece in value.split(";") if piece.strip()}


def load_predictions(path: Path) -> dict[str, set[str]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        key = "predicted_citations" if "predicted_citations" in reader.fieldnames else "citations"
        return {row["query_id"]: parse_citations(row[key]) for row in reader}


def load_gold(path: Path) -> dict[str, set[str]]:
    with path.open() as f:
        return {
            row["query_id"]: parse_citations(row["gold_citations"])
            for row in csv.DictReader(f)
        }


def citation_f1(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def average_prediction_count(predictions: dict[str, set[str]]) -> float:
    counts = [len(citations) for citations in predictions.values()]
    return sum(counts) / len(counts) if counts else 0.0


def average_court_fraction(predictions: dict[str, set[str]]) -> float:
    fractions = []
    for citations in predictions.values():
        if not citations:
            fractions.append(0.0)
            continue
        courts = sum(1 for citation in citations if not citation.startswith("Art."))
        fractions.append(courts / len(citations))
    return sum(fractions) / len(fractions) if fractions else 0.0


def average_jaccard(left: dict[str, set[str]], right: dict[str, set[str]]) -> float:
    scores = []
    for query_id, left_set in left.items():
        right_set = right.get(query_id, set())
        union = left_set | right_set
        scores.append(len(left_set & right_set) / len(union) if union else 1.0)
    return sum(scores) / len(scores) if scores else 0.0


def diff_counts(
    candidate: dict[str, set[str]],
    anchor: dict[str, set[str]],
) -> dict[str, float]:
    adds = removes = add_courts = remove_courts = 0
    changed_queries = 0
    for query_id, candidate_set in candidate.items():
        anchor_set = anchor.get(query_id, set())
        added = candidate_set - anchor_set
        removed = anchor_set - candidate_set
        if added or removed:
            changed_queries += 1
        adds += len(added)
        removes += len(removed)
        add_courts += sum(1 for citation in added if not citation.startswith("Art."))
        remove_courts += sum(1 for citation in removed if not citation.startswith("Art."))
    total_queries = len(candidate) or 1
    return {
        "adds_vs_anchor": adds,
        "removes_vs_anchor": removes,
        "added_courts_vs_anchor": add_courts,
        "removed_courts_vs_anchor": remove_courts,
        "changed_query_fraction": changed_queries / total_queries,
    }


def score_features(
    val_predictions: dict[str, set[str]],
    test_predictions: dict[str, set[str]],
    gold: dict[str, set[str]],
    anchor_test: dict[str, set[str]],
    v11_test: dict[str, set[str]],
    v7_test: dict[str, set[str]],
) -> dict[str, float]:
    by_query = {
        query_id: citation_f1(val_predictions.get(query_id, set()), gold_citations)
        for query_id, gold_citations in gold.items()
    }
    report = {
        "val_macro_f1": sum(by_query.values()) / len(by_query) if by_query else 0.0,
        "val_query_std": statistics.pstdev(by_query.values()) if by_query else 0.0,
        "val_query_min": min(by_query.values()) if by_query else 0.0,
        "test_avg_predictions": average_prediction_count(test_predictions),
        "test_avg_court_fraction": average_court_fraction(test_predictions),
        "test_jaccard_anchor": average_jaccard(test_predictions, anchor_test),
        "test_jaccard_v11": average_jaccard(test_predictions, v11_test),
        "test_jaccard_v7": average_jaccard(test_predictions, v7_test),
    }
    report.update(diff_counts(test_predictions, anchor_test))
    return report


def robust_scale(values: list[float], value: float) -> float:
    median = statistics.median(values)
    deviations = [abs(piece - median) for piece in values]
    mad = statistics.median(deviations)
    if mad == 0:
        spread = max(values) - min(values)
        mad = spread if spread else 1.0
    return (value - median) / mad


def knn_predict_score(
    candidate_features: dict[str, float],
    history_rows: list[dict[str, object]],
) -> tuple[float, list[dict[str, float]]]:
    feature_names = [
        "val_macro_f1",
        "test_avg_predictions",
        "test_avg_court_fraction",
        "test_jaccard_anchor",
        "adds_vs_anchor",
        "removes_vs_anchor",
        "added_courts_vs_anchor",
    ]
    distances: list[tuple[float, dict[str, object]]] = []
    for row in history_rows:
        distance = 0.0
        for feature_name in feature_names:
            values = [float(piece[feature_name]) for piece in history_rows]
            scaled = robust_scale(values, float(candidate_features[feature_name])) - robust_scale(
                values, float(row[feature_name])
            )
            distance += scaled * scaled
        distances.append((math.sqrt(distance), row))
    distances.sort(key=lambda item: item[0])

    neighbors = distances[:3]
    weighted_score = 0.0
    total_weight = 0.0
    neighbor_payload = []
    for distance, row in neighbors:
        weight = 1.0 / max(distance, 1e-6)
        weighted_score += weight * float(row["public_score"])
        total_weight += weight
        neighbor_payload.append(
            {
                "name": str(row["name"]),
                "public_score": float(row["public_score"]),
                "distance": distance,
            }
        )
    predicted = weighted_score / total_weight if total_weight else 0.0
    return predicted, neighbor_payload


def evaluate_history(history_rows: list[dict[str, object]]) -> dict[str, float]:
    predictions: list[tuple[float, float]] = []
    for index, row in enumerate(history_rows):
        train_rows = [piece for i, piece in enumerate(history_rows) if i != index]
        predicted, _ = knn_predict_score(
            {key: float(value) for key, value in row.items() if isinstance(value, (float, int))},
            train_rows,
        )
        predictions.append((float(row["public_score"]), predicted))

    mae = sum(abs(actual - predicted) for actual, predicted in predictions) / len(predictions)
    pair_total = 0
    pair_correct = 0
    for index, (actual_i, pred_i) in enumerate(predictions):
        for actual_j, pred_j in predictions[index + 1 :]:
            if actual_i == actual_j:
                continue
            pair_total += 1
            if (actual_i - actual_j) * (pred_i - pred_j) > 0:
                pair_correct += 1
    return {
        "history_loo_mae": mae,
        "history_pairwise_ranking_accuracy": pair_correct / pair_total if pair_total else 0.0,
    }


def heuristic_verdict(candidate: dict[str, float], anchor: dict[str, float]) -> str:
    if (
        candidate["test_jaccard_anchor"] >= 0.92
        and candidate["adds_vs_anchor"] <= 60
        and candidate["removes_vs_anchor"] <= 10
        and candidate["added_courts_vs_anchor"] == 0
        and candidate["test_avg_predictions"] <= anchor["test_avg_predictions"] + 1.5
        and candidate["test_avg_court_fraction"] <= anchor["test_avg_court_fraction"] + 0.02
    ):
        return "likely_better_or_flat"
    if (
        candidate["test_jaccard_anchor"] < 0.85
        or candidate["adds_vs_anchor"] > 120
        or candidate["added_courts_vs_anchor"] > 30
        or candidate["test_avg_predictions"] > anchor["test_avg_predictions"] + 2.0
        or candidate["test_avg_court_fraction"] > anchor["test_avg_court_fraction"] + 0.03
    ):
        return "likely_worse"
    return "unclear"


def combined_verdict(
    predicted_score: float,
    anchor_score: float,
    heuristic: str,
) -> str:
    delta = predicted_score - anchor_score
    if heuristic == "likely_better_or_flat" and delta >= -0.001:
        return "likely_better_or_flat"
    if heuristic == "likely_worse" and delta <= 0.0:
        return "likely_worse"
    if delta >= 0.002:
        return "possibly_better"
    if delta <= -0.004:
        return "likely_worse"
    return "unclear"


def main() -> None:
    args = parse_args()
    repo_root = Path(".")
    history_data = json.loads(args.history_json.read_text())
    gold = load_gold(args.gold_csv)

    resolved_history = []
    for item in history_data:
        val_path = repo_root / item["val_csv"]
        test_path = repo_root / item["test_csv"]
        resolved_history.append(
            {
                **item,
                "val_predictions": load_predictions(val_path),
                "test_predictions": load_predictions(test_path),
            }
        )

    anchor_row = max(resolved_history, key=lambda item: float(item["public_score"]))
    anchor_test = anchor_row["test_predictions"]
    v11_test = load_predictions(repo_root / "submissions/test_submission_v11_staged.csv")
    v7_test = load_predictions(repo_root / "submissions/test_submission_v7.csv")

    history_rows = []
    for item in resolved_history:
        features = score_features(
            item["val_predictions"],
            item["test_predictions"],
            gold,
            anchor_test,
            v11_test,
            v7_test,
        )
        history_rows.append(
            {
                "name": item["name"],
                "description": item["description"],
                "submission_ref": item["submission_ref"],
                "public_score": float(item["public_score"]),
                **features,
            }
        )

    candidate_features = score_features(
        load_predictions(args.candidate_val),
        load_predictions(args.candidate_test),
        gold,
        anchor_test,
        v11_test,
        v7_test,
    )
    predicted_score, neighbors = knn_predict_score(candidate_features, history_rows)
    anchor_features = next(row for row in history_rows if row["name"] == anchor_row["name"])
    heuristic = heuristic_verdict(candidate_features, anchor_features)
    combined = combined_verdict(predicted_score, float(anchor_row["public_score"]), heuristic)
    history_eval = evaluate_history(history_rows)

    report = {
        "current_best_name": anchor_row["name"],
        "current_best_score": float(anchor_row["public_score"]),
        "candidate_features": candidate_features,
        "predicted_public_score": predicted_score,
        "predicted_delta_vs_best": predicted_score - float(anchor_row["public_score"]),
        "heuristic_verdict": heuristic,
        "combined_verdict": combined,
        "nearest_history_neighbors": neighbors,
        **history_eval,
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"current_best_name={report['current_best_name']}")
    print(f"current_best_score={report['current_best_score']:.5f}")
    print(f"predicted_public_score={report['predicted_public_score']:.5f}")
    print(f"predicted_delta_vs_best={report['predicted_delta_vs_best']:.5f}")
    print(f"heuristic_verdict={report['heuristic_verdict']}")
    print(f"combined_verdict={report['combined_verdict']}")
    print(f"history_loo_mae={report['history_loo_mae']:.6f}")
    print(
        "history_pairwise_ranking_accuracy="
        f"{report['history_pairwise_ranking_accuracy']:.4f}"
    )
    for key in [
        "val_macro_f1",
        "val_query_std",
        "val_query_min",
        "test_avg_predictions",
        "test_avg_court_fraction",
        "test_jaccard_anchor",
        "test_jaccard_v11",
        "test_jaccard_v7",
        "adds_vs_anchor",
        "removes_vs_anchor",
        "added_courts_vs_anchor",
        "removed_courts_vs_anchor",
        "changed_query_fraction",
    ]:
        value = report["candidate_features"][key]
        if isinstance(value, float):
            print(f"{key}={value:.6f}")
        else:
            print(f"{key}={value}")
    print("nearest_history_neighbors:")
    for neighbor in report["nearest_history_neighbors"]:
        print(
            f"  {neighbor['name']}: score={neighbor['public_score']:.5f} "
            f"distance={neighbor['distance']:.6f}"
        )


if __name__ == "__main__":
    main()
