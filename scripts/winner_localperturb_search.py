#!/usr/bin/env python3
"""
Recovered winner-anchored local perturbation search.

Recipe summary:
- Start from the public winner `test_submission_v11_consensus_loose_deepseekpriors.csv`
- Use the three nearest Rust neighbor variants as a tiny perturbation family
- Add only citations that appear in enough neighbor votes, preferring V11/V7 overlap
- Optionally remove citations absent from all neighbors, preferring courts absent from
  both V11 and V7
- The original 2026-04-08 inline run searched a small parameter grid and wrote the top
  3 outputs; this saved script preserves the recovered `build()` logic and reproduces
  the winning `top1` output directly from the recovered winning params instead of
  rerunning a search

Provenance:
- The original inline Python was recovered verbatim from Codex session history:
  `~/.codex/sessions/2026/04/07/rollout-2026-04-07T12-30-23-019d6935-bee7-7563-b911-844d429dd64e.jsonl`
  at line 6354 on 2026-04-09.
- This file is a recovered wrapper around that exact recovered core. The candidate
  build logic, tie-break ordering, and scoring helpers are preserved; only argparse,
  safe output paths, and profile/verification helpers were added.

Winning params:
- add_vote_min     = 2
- max_add_total    = 3
- max_add_court    = 0
- max_add_law      = 3
- max_remove_total = 2
- max_remove_court = 1

Local profile for the winning reproduction:
- val macro F1: 0.282430
- val LB90: 0.255176
- val LB95: 0.249112
- val avg preds: 23.40
- test avg preds: 20.50
- test Jaccard vs base winner: 0.932080
- diff vs base winner: 29/40 test queries changed, 52 cites added (all law), 7 removed

Kaggle:
- submission ref: 51580808
- public score: 0.30257
- description: "Codex winner local perturb top1 2026-04-08"
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_TEST = Path("submissions/test_submission_v11_consensus_loose_deepseekpriors.csv")
DEFAULT_NEIGHBOR_TESTS = [
    Path("submissions/test_submission_v11_consensus_ds_k32_v050_med_b2.csv"),
    Path("submissions/test_submission_v11_consensus_ds_k48_v040_b2.csv"),
    Path("submissions/test_submission_v11_consensus_ds_k64_v035_b3.csv"),
]
DEFAULT_V11_TEST = Path("submissions/test_submission_v11_staged.csv")
DEFAULT_V7_TEST = Path("submissions/test_submission_v7.csv")
DEFAULT_GOLD = Path("data/val.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/reproductions/winner_localperturb_search")
DEFAULT_VAL_NAME = "val_pred_v11_winner_localperturb_top1.csv"
DEFAULT_TEST_NAME = "test_submission_v11_winner_localperturb_top1.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["val", "test", "both"],
        default="both",
        help="Which split(s) to generate. Default writes both reproduction files.",
    )
    parser.add_argument("--base-test-csv", type=Path, default=DEFAULT_BASE_TEST)
    parser.add_argument(
        "--neighbor-test-csv",
        type=Path,
        action="append",
        default=None,
        help="Repeat exactly three times for the neighbor test CSVs.",
    )
    parser.add_argument("--base-val-csv", type=Path)
    parser.add_argument(
        "--neighbor-val-csv",
        type=Path,
        action="append",
        default=None,
        help="Optional explicit val partner CSVs for the three neighbors.",
    )
    parser.add_argument("--v11-test-csv", type=Path, default=DEFAULT_V11_TEST)
    parser.add_argument("--v7-test-csv", type=Path, default=DEFAULT_V7_TEST)
    parser.add_argument("--v11-val-csv", type=Path)
    parser.add_argument("--v7-val-csv", type=Path)
    parser.add_argument("--gold-csv", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-output-name", default=DEFAULT_VAL_NAME)
    parser.add_argument("--test-output-name", default=DEFAULT_TEST_NAME)
    parser.add_argument("--add-vote-min", type=int, default=2)
    parser.add_argument("--max-add-total", type=int, default=3)
    parser.add_argument("--max-add-court", type=int, default=0)
    parser.add_argument("--max-add-law", type=int, default=3)
    parser.add_argument("--max-remove-total", type=int, default=2)
    parser.add_argument("--max-remove-court", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Emit profile as JSON.")
    return parser.parse_args()


def to_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def derive_val_partner(test_path: Path) -> Path:
    test_path = Path(test_path)
    name = test_path.name
    if not name.startswith("test_submission_"):
        raise ValueError(f"Cannot derive val partner from {test_path}")
    return test_path.with_name("val_pred_" + name[len("test_submission_"):])


def parse_citations(value: str) -> set[str]:
    if not value or not value.strip():
        return set()
    return {piece.strip() for piece in value.split(";") if piece.strip()}


def load_predictions(path: Path) -> dict[str, set[str]]:
    out = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        key = "predicted_citations" if "predicted_citations" in reader.fieldnames else "citations"
        for row in reader:
            out[row["query_id"]] = parse_citations(row[key])
    return out


def load_gold(path: Path) -> dict[str, set[str]]:
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["query_id"]] = parse_citations(row["gold_citations"])
    return out


def is_law(citation: str) -> bool:
    return citation.startswith("Art.")


def citation_f1(predicted: set[str], gold: set[str]) -> float:
    tp = len(predicted & gold)
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def avg_stats(bundle: dict[str, set[str]]) -> tuple[float, float]:
    avg_preds = sum(len(citations) for citations in bundle.values()) / len(bundle)
    avg_court = 0.0
    for citations in bundle.values():
        if citations:
            avg_court += sum(not is_law(citation) for citation in citations) / len(citations)
    avg_court /= len(bundle)
    return avg_preds, avg_court


def average_jaccard(left: dict[str, set[str]], right: dict[str, set[str]]) -> float:
    total = 0.0
    for query_id in left:
        union = left[query_id] | right.get(query_id, set())
        total += 1.0 if not union else len(left[query_id] & right.get(query_id, set())) / len(union)
    return total / len(left)


def bootstrap_lower_bound(
    by_query_scores: dict[str, float],
    iterations: int,
    seed: int,
    alpha: float,
) -> float:
    rng = random.Random(seed)
    query_ids = list(by_query_scores)
    sampled_means = []
    for _ in range(iterations):
        sample = [by_query_scores[rng.choice(query_ids)] for _ in query_ids]
        sampled_means.append(sum(sample) / len(sample))
    sampled_means.sort()
    index = max(0, min(len(sampled_means) - 1, int(alpha * len(sampled_means))))
    return sampled_means[index]


def build(
    split: str,
    params: dict[str, int],
    preds: dict[str, dict[str, set[str]]],
    neighbor_names: list[str],
) -> dict[str, set[str]]:
    # Recovered verbatim from the original inline run, apart from args plumbing.
    out = {}
    winner = preds[f"winner_{split}"]
    refs = [preds[f"{name}_{split}"] for name in neighbor_names]
    v11 = preds[f"v11_{split}"]
    v7 = preds[f"v7_{split}"]
    for query_id in sorted(winner):
        base = set(winner[query_id])
        neighbor_sets = [ref[query_id] for ref in refs]
        add_candidates = []
        remove_candidates = []
        union = set().union(*neighbor_sets)
        for citation in union - base:
            votes = sum(citation in neighbor_set for neighbor_set in neighbor_sets)
            if votes < params["add_vote_min"]:
                continue
            kind = "law" if is_law(citation) else "court"
            score = (
                votes,
                int(citation in v11[query_id]),
                int(citation in v7[query_id]),
                int(kind == "law"),
            )
            add_candidates.append((score, citation, kind))
        add_candidates.sort(key=lambda item: (-item[0][0], -item[0][1], -item[0][2], -item[0][3], item[1]))

        absent_all = set(base)
        for neighbor_set in neighbor_sets:
            absent_all -= neighbor_set
        for citation in absent_all:
            kind = "law" if is_law(citation) else "court"
            score = (int(citation not in v11[query_id]), int(citation not in v7[query_id]), int(kind == "court"))
            remove_candidates.append((score, citation, kind))
        remove_candidates.sort(key=lambda item: (-item[0][0], -item[0][1], -item[0][2], item[1]))

        selected = set(base)
        removed_total = 0
        removed_court = 0
        if params["max_remove_total"] > 0:
            for _, citation, kind in remove_candidates:
                if removed_total >= params["max_remove_total"]:
                    break
                if kind == "court" and removed_court >= params["max_remove_court"]:
                    continue
                selected.discard(citation)
                removed_total += 1
                if kind == "court":
                    removed_court += 1

        add_total = 0
        add_court = 0
        add_law = 0
        for _, citation, kind in add_candidates:
            if add_total >= params["max_add_total"]:
                break
            if kind == "court" and add_court >= params["max_add_court"]:
                continue
            if kind == "law" and add_law >= params["max_add_law"]:
                continue
            selected.add(citation)
            add_total += 1
            if kind == "court":
                add_court += 1
            else:
                add_law += 1
        out[query_id] = selected
    return out


def write_predictions(path: Path, predictions: dict[str, set[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\r\n")
        writer.writerow(["query_id", "predicted_citations"])
        for query_id in sorted(predictions):
            writer.writerow([query_id, ";".join(sorted(predictions[query_id]))])


def build_profile(
    val_predictions: dict[str, set[str]],
    gold: dict[str, set[str]],
    test_predictions: dict[str, set[str]] | None,
    refs: dict[str, dict[str, set[str]]],
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    by_query = {
        query_id: citation_f1(val_predictions.get(query_id, set()), gold_citations)
        for query_id, gold_citations in gold.items()
    }
    report: dict[str, object] = {
        "val_macro_f1": sum(by_query.values()) / len(by_query) if by_query else 0.0,
        "val_bootstrap_lb90": bootstrap_lower_bound(by_query, bootstrap, seed, 0.10),
        "val_bootstrap_lb95": bootstrap_lower_bound(by_query, bootstrap, seed, 0.05),
        "val_query_std": statistics.pstdev(by_query.values()) if by_query else 0.0,
        "val_query_min": min(by_query.values()) if by_query else 0.0,
        "val_avg_predictions": avg_stats(val_predictions)[0],
        "val_avg_court_fraction": avg_stats(val_predictions)[1],
        "val_by_query": by_query,
    }
    if test_predictions is not None:
        report["test_avg_predictions"] = avg_stats(test_predictions)[0]
        report["test_avg_court_fraction"] = avg_stats(test_predictions)[1]
        report["test_reference_jaccard"] = {
            label: average_jaccard(test_predictions, ref_predictions)
            for label, ref_predictions in refs.items()
        }
    return report


def print_profile(report: dict[str, object], as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"val_macro_f1={report['val_macro_f1']:.6f}")
    print(f"val_bootstrap_lb90={report['val_bootstrap_lb90']:.6f}")
    print(f"val_bootstrap_lb95={report['val_bootstrap_lb95']:.6f}")
    print(f"val_query_std={report['val_query_std']:.6f}")
    print(f"val_query_min={report['val_query_min']:.6f}")
    print(f"val_avg_predictions={report['val_avg_predictions']:.2f}")
    print(f"val_avg_court_fraction={report['val_avg_court_fraction']:.4f}")
    if "test_avg_predictions" in report:
        print(f"test_avg_predictions={report['test_avg_predictions']:.2f}")
        print(f"test_avg_court_fraction={report['test_avg_court_fraction']:.4f}")
        for label, score in sorted(report["test_reference_jaccard"].items()):
            print(f"test_jaccard_{label}={score:.6f}")
    print("val_by_query:")
    for query_id, score in sorted(report["val_by_query"].items()):
        print(f"  {query_id}: {score:.6f}")


def main() -> None:
    args = parse_args()

    neighbor_test_csvs = args.neighbor_test_csv or list(DEFAULT_NEIGHBOR_TESTS)
    if len(neighbor_test_csvs) != 3:
        raise SystemExit("--neighbor-test-csv must be supplied exactly three times")

    base_test = to_repo_path(args.base_test_csv)
    base_val = to_repo_path(args.base_val_csv or derive_val_partner(args.base_test_csv))
    neighbor_tests = [to_repo_path(path) for path in neighbor_test_csvs]
    neighbor_vals_input = args.neighbor_val_csv or [derive_val_partner(path) for path in neighbor_test_csvs]
    if len(neighbor_vals_input) != 3:
        raise SystemExit("--neighbor-val-csv must be supplied zero or exactly three times")
    neighbor_vals = [to_repo_path(path) for path in neighbor_vals_input]
    v11_test = to_repo_path(args.v11_test_csv)
    v11_val = to_repo_path(args.v11_val_csv or derive_val_partner(args.v11_test_csv))
    v7_test = to_repo_path(args.v7_test_csv)
    v7_val = to_repo_path(args.v7_val_csv or derive_val_partner(args.v7_test_csv))
    gold_path = to_repo_path(args.gold_csv)
    output_dir = to_repo_path(args.output_dir)

    files = {
        "winner_val": base_val,
        "winner_test": base_test,
        "k32_val": neighbor_vals[0],
        "k32_test": neighbor_tests[0],
        "k48_val": neighbor_vals[1],
        "k48_test": neighbor_tests[1],
        "k64_val": neighbor_vals[2],
        "k64_test": neighbor_tests[2],
        "v11_test": v11_test,
        "v7_test": v7_test,
        "v11_val": v11_val if v11_val.exists() else base_val,
        "v7_val": v7_val,
    }

    preds = {label: load_predictions(path) for label, path in files.items()}
    gold = load_gold(gold_path)
    params = {
        "add_vote_min": args.add_vote_min,
        "max_add_total": args.max_add_total,
        "max_add_court": args.max_add_court,
        "max_add_law": args.max_add_law,
        "max_remove_total": args.max_remove_total,
        "max_remove_court": args.max_remove_court,
    }
    neighbor_names = ["k32", "k48", "k64"]

    val_predictions = None
    test_predictions = None
    if args.mode in {"val", "both"}:
        val_predictions = build("val", params, preds, neighbor_names)
        write_predictions(output_dir / args.val_output_name, val_predictions)
    if args.mode in {"test", "both"}:
        test_predictions = build("test", params, preds, neighbor_names)
        write_predictions(output_dir / args.test_output_name, test_predictions)

    if val_predictions is None:
        val_predictions = build("val", params, preds, neighbor_names)
    report = build_profile(
        val_predictions,
        gold,
        test_predictions,
        {
            "winner": preds["winner_test"],
            "v11": preds["v11_test"],
            "v7": preds["v7_test"],
        } if test_predictions is not None else {},
        args.bootstrap,
        args.seed,
    )
    print_profile(report, args.json)


if __name__ == "__main__":
    main()
