#!/usr/bin/env python3
"""
Private-final portfolio decision audit.

This is a decision layer on top of the Rust private split tools. It does three
things that are easy to get wrong in a public-LB chase:

1. Aggregate pair_report.tsv files across many simulated private split schemes.
2. Enumerate exact half-validation private splits for a small val set.
3. Flag public-clone/test-only candidates that should not displace private legs.

It never reads Kaggle leaderboard feedback and never submits anything.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Candidate:
    name: str
    val_path: Path | None
    test_path: Path | None
    note: str = ""


@dataclass
class CandidateFlag:
    name: str
    has_val: bool
    has_test: bool
    test_total_cites: int | None
    test_mean_cites: float | None
    jaccard_vs_public_anchor: float | None
    public_clone: bool
    promotable: bool
    reason: str


@dataclass
class SplitAgg:
    ranks: list[int]
    p_contains: list[float]
    mean_best: list[float]
    std_best: list[float]
    test_jaccard: list[float]
    runs: list[str]


@dataclass
class ExactPairStats:
    worst: float
    p05: float
    p10: float
    median: float
    mean: float
    p90: float
    best: float
    contains_winner: float
    regret: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=REPO
        / "artifacts/private_final_recheck_20260522/universe_v7_corpusclean224/combined_candidate_manifest_capped.tsv",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO / "artifacts/private_final_recheck_20260522/combined_private_sweep_v7_corpusclean224_fullpairs",
    )
    parser.add_argument("--gold", type=Path, default=REPO / "data/val.csv")
    parser.add_argument(
        "--baseline-left",
        default="intersect_bold7h_33028",
        help="One leg of the current private baseline pair.",
    )
    parser.add_argument(
        "--baseline-right",
        default="widebankG_hailmary_30702",
        help="Other leg of the current private baseline pair.",
    )
    parser.add_argument(
        "--public-anchor",
        default="public_peak_33438",
        help="Candidate name used as public-clone reference.",
    )
    parser.add_argument(
        "--extra-test-only",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional test-only candidate to flag as non-promotable.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "artifacts/private_final_recheck_20260522/decision_audit",
    )
    parser.add_argument("--clone-jaccard", type=float, default=0.98)
    parser.add_argument("--weak-hedge-jaccard", type=float, default=0.95)
    parser.add_argument("--avg-p-margin", type=float, default=0.005)
    parser.add_argument("--mean-best-margin", type=float, default=0.0002)
    parser.add_argument("--min-p-tolerance", type=float, default=0.0)
    parser.add_argument("--exact-p10-tolerance", type=float, default=0.003)
    parser.add_argument("--exact-regret-tolerance", type=float, default=0.0005)
    parser.add_argument("--max-jaccard-over-baseline", type=float, default=0.04)
    return parser.parse_args()


def repo_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def parse_citations(value: str | None) -> set[str]:
    return {piece.strip() for piece in (value or "").split(";") if piece.strip()}


def load_predictions(path: Path) -> dict[str, set[str]]:
    with path.open(newline="") as f:
        return {
            row["query_id"]: parse_citations(row.get("predicted_citations"))
            for row in csv.DictReader(f)
        }


def load_gold(path: Path) -> tuple[list[str], dict[str, set[str]]]:
    qids: list[str] = []
    gold: dict[str, set[str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"]
            qids.append(qid)
            gold[qid] = parse_citations(row.get("gold_citations"))
    return qids, gold


def f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    return (2.0 * tp) / (len(pred) + len(gold))


def atom_jaccard(left: dict[str, set[str]], right: dict[str, set[str]]) -> float:
    left_atoms = {(qid, citation) for qid, citations in left.items() for citation in citations}
    right_atoms = {(qid, citation) for qid, citations in right.items() for citation in citations}
    union = left_atoms | right_atoms
    if not union:
        return 1.0
    return len(left_atoms & right_atoms) / len(union)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return statistics.fmean(vals) if vals else math.nan


def pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def load_manifest(path: Path, extra_test_only: list[str]) -> dict[str, Candidate]:
    candidates: dict[str, Candidate] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row["name"]
            candidates[name] = Candidate(
                name=name,
                val_path=repo_path(row.get("pred_path") or None),
                test_path=repo_path(row.get("test_path") or None),
                note=row.get("note", ""),
            )
    for raw in extra_test_only:
        if "=" not in raw:
            raise ValueError(f"--extra-test-only must be NAME=PATH, got {raw!r}")
        name, path = raw.split("=", 1)
        candidates[name] = Candidate(
            name=name,
            val_path=None,
            test_path=repo_path(path),
            note="extra test-only candidate; not promotable without matching val",
        )
    return candidates


def load_split_aggregates(runs_root: Path) -> dict[tuple[str, str], SplitAgg]:
    aggregates: dict[tuple[str, str], SplitAgg] = defaultdict(
        lambda: SplitAgg([], [], [], [], [], [])
    )
    for report in sorted(runs_root.glob("*/pair_report.tsv")):
        run_name = report.parent.name
        with report.open(newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for rank, row in enumerate(reader, 1):
                key = pair_key(row["left"], row["right"])
                agg = aggregates[key]
                agg.ranks.append(rank)
                agg.p_contains.append(float(row["p_contains_private_winner"]))
                agg.mean_best.append(float(row["mean_best_private"]))
                agg.std_best.append(float(row["std_best_private"]))
                raw_j = row.get("test_jaccard")
                if raw_j not in (None, ""):
                    agg.test_jaccard.append(float(raw_j))
                agg.runs.append(run_name)
    return dict(aggregates)


def candidate_flags(
    candidates: dict[str, Candidate],
    public_anchor: str,
    clone_jaccard: float,
) -> dict[str, CandidateFlag]:
    test_preds: dict[str, dict[str, set[str]]] = {}
    for name, cand in candidates.items():
        if cand.test_path and cand.test_path.exists():
            test_preds[name] = load_predictions(cand.test_path)
    anchor_preds = test_preds.get(public_anchor)
    flags: dict[str, CandidateFlag] = {}
    for name, cand in candidates.items():
        has_val = bool(cand.val_path and cand.val_path.exists())
        has_test = bool(cand.test_path and cand.test_path.exists())
        total = None
        mean_count = None
        anchor_j = None
        clone = False
        reasons: list[str] = []
        if not has_val:
            reasons.append("missing val companion")
        if not has_test:
            reasons.append("missing test CSV")
        if has_test:
            preds = test_preds[name]
            counts = [len(v) for v in preds.values()]
            total = sum(counts)
            mean_count = total / len(counts) if counts else 0.0
            if anchor_preds is not None:
                anchor_j = atom_jaccard(preds, anchor_preds)
                clone = name != public_anchor and anchor_j >= clone_jaccard
                if clone:
                    reasons.append(f"public clone J={anchor_j:.4f}")
        promotable = has_val and has_test
        if not reasons:
            reasons.append("ok")
        flags[name] = CandidateFlag(
            name=name,
            has_val=has_val,
            has_test=has_test,
            test_total_cites=total,
            test_mean_cites=mean_count,
            jaccard_vs_public_anchor=anchor_j,
            public_clone=clone,
            promotable=promotable,
            reason="; ".join(reasons),
        )
    return flags


def exact_pair_stats(
    qids: list[str],
    gold: dict[str, set[str]],
    candidates: dict[str, Candidate],
) -> dict[tuple[str, str], ExactPairStats]:
    usable = {
        name: cand
        for name, cand in candidates.items()
        if cand.val_path is not None and cand.val_path.exists()
    }
    if len(qids) < 2 or len(usable) < 2:
        return {}
    per_query_scores: dict[str, list[float]] = {}
    for name, cand in usable.items():
        preds = load_predictions(cand.val_path)  # type: ignore[arg-type]
        per_query_scores[name] = [f1(preds.get(qid, set()), gold[qid]) for qid in qids]

    private_size = len(qids) // 2
    masks = list(itertools.combinations(range(len(qids)), private_size))
    names = sorted(usable)
    candidate_private: dict[str, list[float]] = {name: [] for name in names}
    split_winners: list[str] = []
    best_scores: list[float] = []
    for mask in masks:
        scores = {
            name: statistics.fmean(per_query_scores[name][idx] for idx in mask)
            for name in names
        }
        winner, best = max(scores.items(), key=lambda item: (item[1], item[0]))
        split_winners.append(winner)
        best_scores.append(best)
        for name, score in scores.items():
            candidate_private[name].append(score)

    pair_stats: dict[tuple[str, str], ExactPairStats] = {}
    for left, right in itertools.combinations(names, 2):
        values = [
            max(candidate_private[left][idx], candidate_private[right][idx])
            for idx in range(len(masks))
        ]
        contains = sum(
            1 for winner in split_winners if winner == left or winner == right
        ) / len(masks)
        regret = statistics.fmean(
            best_scores[idx] - values[idx] for idx in range(len(values))
        )
        pair_stats[pair_key(left, right)] = ExactPairStats(
            worst=min(values),
            p05=percentile(values, 0.05),
            p10=percentile(values, 0.10),
            median=percentile(values, 0.50),
            mean=statistics.fmean(values),
            p90=percentile(values, 0.90),
            best=max(values),
            contains_winner=contains,
            regret=regret,
        )
    return pair_stats


def write_candidate_flags(path: Path, flags: dict[str, CandidateFlag]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "name",
                "has_val",
                "has_test",
                "promotable",
                "public_clone",
                "test_total_cites",
                "test_mean_cites",
                "jaccard_vs_public_anchor",
                "reason",
            ]
        )
        for flag in sorted(flags.values(), key=lambda item: item.name):
            writer.writerow(
                [
                    flag.name,
                    int(flag.has_val),
                    int(flag.has_test),
                    int(flag.promotable),
                    int(flag.public_clone),
                    "" if flag.test_total_cites is None else flag.test_total_cites,
                    ""
                    if flag.test_mean_cites is None
                    else f"{flag.test_mean_cites:.6f}",
                    ""
                    if flag.jaccard_vs_public_anchor is None
                    else f"{flag.jaccard_vs_public_anchor:.6f}",
                    flag.reason,
                ]
            )


def decision_rows(
    aggregates: dict[tuple[str, str], SplitAgg],
    exact: dict[tuple[str, str], ExactPairStats],
    flags: dict[str, CandidateFlag],
    baseline: tuple[str, str],
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    if baseline not in aggregates:
        raise ValueError(f"baseline pair {baseline} not found in split reports")
    base = aggregates[baseline]
    base_exact = exact.get(baseline)
    base_avg_p = mean(base.p_contains)
    base_min_p = min(base.p_contains)
    base_avg_rank = mean(base.ranks)
    base_run_count = len(set(base.runs))
    base_mean_best = mean(base.mean_best)
    base_avg_j = mean(base.test_jaccard)
    base_exact_p10 = base_exact.p10 if base_exact else math.nan
    base_exact_regret = base_exact.regret if base_exact else math.nan

    rows: list[dict[str, str]] = []
    for key, agg in aggregates.items():
        left, right = key
        run_count = len(set(agg.runs))
        avg_p = mean(agg.p_contains)
        min_p = min(agg.p_contains)
        avg_rank = mean(agg.ranks)
        worst_rank = max(agg.ranks)
        avg_mean_best = mean(agg.mean_best)
        avg_std_best = mean(agg.std_best)
        avg_j = mean(agg.test_jaccard)
        e = exact.get(key)
        exact_p10 = e.p10 if e else math.nan
        exact_worst = e.worst if e else math.nan
        exact_mean = e.mean if e else math.nan
        exact_contains = e.contains_winner if e else math.nan
        exact_regret = e.regret if e else math.nan
        left_flag = flags.get(left)
        right_flag = flags.get(right)
        hard_reasons: list[str] = []
        if left_flag and not left_flag.promotable:
            hard_reasons.append(f"{left}: {left_flag.reason}")
        if right_flag and not right_flag.promotable:
            hard_reasons.append(f"{right}: {right_flag.reason}")
        if avg_j >= args.weak_hedge_jaccard:
            hard_reasons.append(f"weak hedge J={avg_j:.4f}")
        if left_flag and right_flag and left_flag.public_clone and right_flag.public_clone:
            hard_reasons.append("both legs are public clones")
        if run_count < base_run_count:
            hard_reasons.append(f"incomplete split-report support {run_count}/{base_run_count}")

        tests = {
            "mean_best": avg_mean_best >= base_mean_best + args.mean_best_margin,
            "avg_p": avg_p >= base_avg_p + args.avg_p_margin,
            "min_p": min_p >= base_min_p + args.min_p_tolerance,
            "exact_p10": bool(
                base_exact
                and e
                and exact_p10 >= base_exact_p10 - args.exact_p10_tolerance
            ),
            "exact_regret": bool(
                base_exact
                and e
                and exact_regret <= base_exact_regret + args.exact_regret_tolerance
            ),
            "diversity": avg_j <= base_avg_j + args.max_jaccard_over_baseline,
        }
        pass_count = sum(tests.values())
        if key == baseline:
            verdict = "BASELINE_KEEP"
        elif hard_reasons:
            verdict = "REJECT_HARD"
        elif (
            tests["mean_best"]
            and tests["avg_p"]
            and tests["exact_p10"]
            and tests["exact_regret"]
            and tests["diversity"]
        ):
            verdict = "REPLACE_REVIEW"
        elif (
            tests["avg_p"]
            and tests["exact_p10"]
            and tests["exact_regret"]
            and tests["diversity"]
            and avg_mean_best >= base_mean_best - args.mean_best_margin
        ):
            verdict = "PRIVATE_PROBE"
        elif (
            tests["exact_p10"]
            and tests["exact_regret"]
            and tests["diversity"]
            and avg_mean_best >= base_mean_best - 0.001
        ):
            verdict = "HEDGE_WATCH"
        else:
            verdict = "HOLD_BELOW_BASELINE"

        rows.append(
            {
                "left": left,
                "right": right,
                "verdict": verdict,
                "pass_count": str(pass_count),
                "run_count": str(run_count),
                "avg_p_contains": f"{avg_p:.6f}",
                "min_p_contains": f"{min_p:.6f}",
                "avg_rank": f"{avg_rank:.3f}",
                "worst_rank": str(worst_rank),
                "avg_mean_best_private": f"{avg_mean_best:.6f}",
                "delta_mean_best_private": f"{avg_mean_best - base_mean_best:.6f}",
                "avg_std_best_private": f"{avg_std_best:.6f}",
                "avg_test_jaccard": f"{avg_j:.6f}",
                "delta_test_jaccard": f"{avg_j - base_avg_j:.6f}",
                "exact_p10": "" if math.isnan(exact_p10) else f"{exact_p10:.6f}",
                "delta_exact_p10": ""
                if math.isnan(exact_p10) or math.isnan(base_exact_p10)
                else f"{exact_p10 - base_exact_p10:.6f}",
                "exact_worst": "" if math.isnan(exact_worst) else f"{exact_worst:.6f}",
                "exact_mean": "" if math.isnan(exact_mean) else f"{exact_mean:.6f}",
                "exact_contains_winner": ""
                if math.isnan(exact_contains)
                else f"{exact_contains:.6f}",
                "exact_regret": "" if math.isnan(exact_regret) else f"{exact_regret:.6f}",
                "delta_exact_regret": ""
                if math.isnan(exact_regret) or math.isnan(base_exact_regret)
                else f"{exact_regret - base_exact_regret:.6f}",
                "runs": ",".join(sorted(set(agg.runs))),
                "hard_reasons": "; ".join(hard_reasons),
                "tests": ",".join(k for k, ok in tests.items() if ok),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["verdict"] != "BASELINE_KEEP",
            row["verdict"] != "REPLACE_REVIEW",
            row["verdict"] != "PRIVATE_PROBE",
            row["verdict"] != "HEDGE_WATCH",
            -float(row["avg_p_contains"]),
            -float(row["delta_mean_best_private"]),
            float(row["avg_rank"]),
        ),
    )


def write_pair_decisions(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    rows: list[dict[str, str]],
    flags: dict[str, CandidateFlag],
    baseline: tuple[str, str],
) -> None:
    baseline_row = next(row for row in rows if pair_key(row["left"], row["right"]) == baseline)
    challengers = [row for row in rows if row["verdict"] == "REPLACE_REVIEW"]
    probes = [row for row in rows if row["verdict"] == "PRIVATE_PROBE"]
    hard_rejects = [row for row in rows if row["verdict"] == "REJECT_HARD"]
    clone_flags = [flag for flag in flags.values() if flag.public_clone or not flag.promotable]
    with path.open("w") as f:
        f.write("# Private Final Portfolio Decision Audit\n\n")
        f.write("## Decision\n\n")
        if challengers:
            f.write("At least one challenger passed the strict replacement-review gate.\n\n")
        elif probes:
            f.write("No strict replacement passed, but private-probe alternatives need manual review.\n\n")
        else:
            f.write("No challenger beats the current private baseline under the strict gate.\n\n")
        f.write("Current baseline pair remains:\n\n")
        f.write(f"- `{baseline[0]}`\n- `{baseline[1]}`\n\n")
        f.write("## Baseline Metrics\n\n")
        f.write("| Metric | Value |\n|---|---:|\n")
        for key in [
            "avg_p_contains",
            "min_p_contains",
            "avg_rank",
            "avg_mean_best_private",
            "delta_mean_best_private",
            "avg_std_best_private",
            "avg_test_jaccard",
            "delta_test_jaccard",
            "exact_p10",
            "delta_exact_p10",
            "exact_worst",
            "exact_mean",
            "exact_regret",
            "delta_exact_regret",
        ]:
            f.write(f"| `{key}` | `{baseline_row[key]}` |\n")
        f.write("\n## Top Pair Decisions\n\n")
        f.write(
            "| Verdict | Pair | Avg p_contains | Min p_contains | Avg rank | Jaccard | Exact p10 | Reasons |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows[:12]:
            pair = f"`{row['left']}` + `{row['right']}`"
            reasons = row["hard_reasons"] or row["tests"] or "baseline"
            f.write(
                f"| {row['verdict']} | {pair} | `{row['avg_p_contains']}` | `{row['min_p_contains']}` | `{row['avg_rank']}` | `{row['avg_test_jaccard']}` | `{row['exact_p10']}` | {reasons} |\n"
            )
        if clone_flags:
            f.write("\n## Candidate Warnings\n\n")
            f.write("| Candidate | Warning |\n|---|---|\n")
            for flag in sorted(clone_flags, key=lambda item: item.name):
                f.write(f"| `{flag.name}` | {flag.reason} |\n")
        if hard_rejects:
            f.write("\n## Hard-Rejected Pair Examples\n\n")
            for row in hard_rejects[:8]:
                f.write(
                    f"- `{row['left']}` + `{row['right']}`: {row['hard_reasons']}\n"
                )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_manifest(args.candidate_manifest, args.extra_test_only)
    flags = candidate_flags(candidates, args.public_anchor, args.clone_jaccard)
    aggregates = load_split_aggregates(args.runs_root)
    qids, gold = load_gold(args.gold)
    exact = exact_pair_stats(qids, gold, candidates)
    baseline = pair_key(args.baseline_left, args.baseline_right)
    rows = decision_rows(aggregates, exact, flags, baseline, args)

    write_candidate_flags(args.out_dir / "candidate_flags.tsv", flags)
    write_pair_decisions(args.out_dir / "pair_decisions.tsv", rows)
    write_markdown(args.out_dir / "decision.md", rows, flags, baseline)

    baseline_row = next(row for row in rows if pair_key(row["left"], row["right"]) == baseline)
    print("private_final_portfolio_decision complete")
    print(f"out_dir={args.out_dir}")
    print(
        "baseline "
        f"{baseline[0]} + {baseline[1]} "
        f"avg_p={baseline_row['avg_p_contains']} "
        f"min_p={baseline_row['min_p_contains']} "
        f"rank={baseline_row['avg_rank']}"
    )
    challengers = [row for row in rows if row["verdict"] == "PROMOTE_CHALLENGER"]
    challengers = [row for row in rows if row["verdict"] == "REPLACE_REVIEW"]
    probes = [row for row in rows if row["verdict"] == "PRIVATE_PROBE"]
    hedges = [row for row in rows if row["verdict"] == "HEDGE_WATCH"]
    print(f"replace_review={len(challengers)} private_probe={len(probes)} hedge_watch={len(hedges)}")
    for row in challengers[:5]:
        print(
            f"REPLACE_REVIEW {row['left']} + {row['right']} "
            f"avg_p={row['avg_p_contains']} "
            f"delta_mean={row['delta_mean_best_private']} "
            f"delta_p10={row['delta_exact_p10']}"
        )


if __name__ == "__main__":
    main()
