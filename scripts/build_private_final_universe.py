#!/usr/bin/env python3
"""
Build a curated private-final candidate universe.

The existing private-selection referee is only as good as the candidates it is
given. This script merges the strongest historical private-sweep manifests,
deduplicates by test payload hash, runs an exact half-val portfolio triage, and
writes a capped manifest for the expensive Rust split-simulation pass.

It does not submit anything and does not modify source submissions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]

DEFAULT_MANIFESTS = [
    "artifacts/final_selection_20260519/private_split_final_audit/final_candidate_manifest.tsv",
    "artifacts/private_leg_max_20260521/candidate_manifest.tsv",
    "artifacts/private_split_portfolio/final_val_manifest_wide.tsv",
    "artifacts/private_split_portfolio/staff_level_manifest_20260513.tsv",
    "artifacts/private_split_portfolio/staff_level_audit_manifest_20260513.tsv",
    "artifacts/elite_3h_private_stress_20260511T125331Z/manifests/elite_shortlist.tsv",
    "artifacts/private_survival_overlay_20260512T052655/candidate_manifest.tsv",
    "artifacts/cpu_final_slate_20260516/fusion_top_manifest.tsv",
    "artifacts/eight_hour_private_ai_20260511T043652Z/manifests/all_final.tsv",
]

DEFAULT_MARKDOWN_MANIFESTS = [
    "submissions/staff_fifteenth_order_final_portfolio_manifest.md",
    "submissions/staff_fifteenth_order_20260425T231618Z_final_portfolio_manifest.md",
]

CANONICAL_TEST_PATH_SUFFIXES = {
    "submissions/staff3_pairing_20260513/test_submission_private_rethink_intersect_bold7h_j955.csv": "intersect_bold7h_33028",
    "submissions/final_staff_level_20260513/test_submission_final_hedge_overlay_fusion01_plus_samesrc03.csv": "fusion_samesrc03_32274",
    "submissions/public_precision_targeted_20260518/live_refit_after_33385/test_submission_33385_nextrem_03_est33390.csv": "public_peak_33438",
    "submissions/prepared_public_20260513/test_submission_private_rethink_overlay_samesrc_02.csv": "public_peak_33438_prepared",
    "submissions/prepared_public_20260513/test_submission_pre_tomo_best_33186.csv": "pre_tomo_33186",
    "submissions/test_submission_bold_7h_widebankG_hailmary.csv": "widebankG_hailmary_30702",
    "submissions/test_submission_bold_7h_widebankG_balanced.csv": "widebankG_balanced",
}

MANDATORY_NAMES = {
    "intersect_bold7h_33028",
    "fusion_samesrc03_32274",
    "public_peak_33438",
    "pre_tomo_33186",
    "samesrc32904_fixed12_lfc1_32562",
    "public_peak_33438_prepared",
    "widebankG_hailmary_30702",
}


@dataclass
class Candidate:
    name: str
    pred_path: Path
    test_path: Path
    public_score: str = ""
    note: str = ""
    sources: list[str] = field(default_factory=list)
    test_sha: str = ""
    val_sha: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Additional TSV manifest. Defaults are used unless --no-defaults is set.",
    )
    parser.add_argument(
        "--markdown-manifest",
        action="append",
        default=[],
        help="Additional Markdown portfolio manifest with '- Test:' / '- Val:' entries.",
    )
    parser.add_argument(
        "--markdown-glob",
        action="append",
        default=["submissions/staff_fifteenth_order*_final_portfolio_manifest.md"],
        help="Repo-relative glob for Markdown portfolio manifests. Defaults to all staff fifteenth-order manifests.",
    )
    parser.add_argument("--no-defaults", action="store_true")
    parser.add_argument("--gold", type=Path, default=REPO / "data/val.csv")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "artifacts/private_final_recheck_20260522/universe",
    )
    parser.add_argument("--max-candidates", type=int, default=72)
    parser.add_argument("--public-anchor", default="public_peak_33438")
    parser.add_argument("--baseline-left", default="intersect_bold7h_33028")
    parser.add_argument("--baseline-right", default="fusion_samesrc03_32274")
    parser.add_argument(
        "--force-name",
        action="append",
        default=[],
        help="Candidate name to preserve in the capped manifest even if cap scoring would trim it.",
    )
    return parser.parse_args()


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_citations(value: str | None) -> set[str]:
    return {piece.strip() for piece in (value or "").split(";") if piece.strip()}


def load_predictions(path: Path, label_col: str = "predicted_citations") -> dict[str, set[str]]:
    with path.open(newline="") as f:
        return {
            row["query_id"]: parse_citations(row.get(label_col))
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
    return 2.0 * tp / (len(pred) + len(gold))


def atom_jaccard(left: dict[str, set[str]], right: dict[str, set[str]]) -> float:
    left_atoms = {(qid, cit) for qid, cites in left.items() for cit in cites}
    right_atoms = {(qid, cit) for qid, cites in right.items() for cit in cites}
    union = left_atoms | right_atoms
    return 1.0 if not union else len(left_atoms & right_atoms) / len(union)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    pos = (len(xs) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return statistics.fmean(vals) if vals else math.nan


def canonical_name(name: str, test_path: Path) -> str:
    test_rel = rel(test_path)
    for suffix, canonical in CANONICAL_TEST_PATH_SUFFIXES.items():
        if test_rel == suffix or test_rel.endswith(suffix):
            return canonical
    return name.strip()


def score_public_score(raw: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return -1.0
    return value


def prefer_candidate(new: Candidate, old: Candidate) -> Candidate:
    new_key = (
        new.name in MANDATORY_NAMES,
        score_public_score(new.public_score),
        -len(rel(new.test_path)),
        -len(rel(new.pred_path)),
    )
    old_key = (
        old.name in MANDATORY_NAMES,
        score_public_score(old.public_score),
        -len(rel(old.test_path)),
        -len(rel(old.pred_path)),
    )
    if new_key > old_key:
        new.sources = old.sources + new.sources
        return new
    old.sources.extend(new.sources)
    return old


def candidate_from_paths(
    name: str,
    pred_raw: str,
    test_raw: str,
    source: Path,
    public_score: str = "",
    note: str = "",
) -> tuple[Candidate | None, dict[str, str] | None]:
    pred_path = repo_path(pred_raw)
    test_path = repo_path(test_raw)
    if not pred_path.exists() or not test_path.exists():
        missing = []
        if not pred_path.exists():
            missing.append("pred_path")
        if not test_path.exists():
            missing.append("test_path")
        return None, {
            "manifest": rel(source),
            "name": name,
            "reason": "missing " + ",".join(missing),
        }
    canon = canonical_name(name, test_path)
    return (
        Candidate(
            name=canon,
            pred_path=pred_path,
            test_path=test_path,
            public_score=public_score,
            note=note,
            sources=[rel(source)],
        ),
        None,
    )


def load_tsv_manifest_rows(paths: list[str]) -> tuple[list[Candidate], list[dict[str, str]]]:
    candidates: list[Candidate] = []
    skipped: list[dict[str, str]] = []
    for raw_path in paths:
        manifest = repo_path(raw_path)
        if not manifest.exists():
            skipped.append({"manifest": raw_path, "name": "", "reason": "manifest missing"})
            continue
        with manifest.open(newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for idx, row in enumerate(reader, 2):
                name = (row.get("name") or "").strip()
                pred_raw = (row.get("pred_path") or "").strip()
                test_raw = (row.get("test_path") or "").strip()
                if not name or not pred_raw or not test_raw:
                    skipped.append(
                        {
                            "manifest": rel(manifest),
                            "name": name,
                            "reason": f"missing required field at line {idx}",
                        }
                    )
                    continue
                cand, skip = candidate_from_paths(
                    name=name,
                    pred_raw=pred_raw,
                    test_raw=test_raw,
                    source=manifest,
                    public_score=row.get("public_score", ""),
                    note=row.get("note", ""),
                )
                if cand:
                    candidates.append(cand)
                if skip:
                    skipped.append(skip)
    return candidates, skipped


def load_markdown_manifest_rows(paths: list[str]) -> tuple[list[Candidate], list[dict[str, str]]]:
    candidates: list[Candidate] = []
    skipped: list[dict[str, str]] = []
    header_re = re.compile(r"^###\s+(.+?)\s*$")
    test_re = re.compile(r"^- Test:\s+`([^`]+)`")
    val_re = re.compile(r"^- Val:\s+`([^`]+)`")
    for raw_path in paths:
        manifest = repo_path(raw_path)
        if not manifest.exists():
            skipped.append({"manifest": raw_path, "name": "", "reason": "manifest missing"})
            continue
        current_name: str | None = None
        current_test: str | None = None
        current_val: str | None = None

        def flush() -> None:
            nonlocal current_name, current_test, current_val
            if current_name and current_test and current_val:
                cand, skip = candidate_from_paths(
                    name=current_name,
                    pred_raw=current_val,
                    test_raw=current_test,
                    source=manifest,
                    note="parsed from staff portfolio Markdown",
                )
                if cand:
                    candidates.append(cand)
                if skip:
                    skipped.append(skip)
            current_name = None
            current_test = None
            current_val = None

        for line in manifest.read_text(errors="replace").splitlines():
            header = header_re.match(line)
            if header:
                flush()
                current_name = header.group(1).strip().replace("`", "")
                continue
            test = test_re.match(line)
            if test:
                current_test = test.group(1).strip()
                continue
            val = val_re.match(line)
            if val:
                current_val = val.group(1).strip()
        flush()
    return candidates, skipped


def dedupe_candidates(candidates: list[Candidate]) -> tuple[list[Candidate], list[dict[str, str]]]:
    by_test_sha: dict[str, Candidate] = {}
    duplicate_rows: list[dict[str, str]] = []
    for cand in candidates:
        cand.test_sha = sha256_file(cand.test_path)
        cand.val_sha = sha256_file(cand.pred_path)
        old = by_test_sha.get(cand.test_sha)
        if old is None:
            by_test_sha[cand.test_sha] = cand
            continue
        chosen = prefer_candidate(cand, old)
        dropped = old if chosen is cand else cand
        duplicate_rows.append(
            {
                "kept": chosen.name,
                "dropped": dropped.name,
                "test_sha": cand.test_sha[:12],
                "kept_path": rel(chosen.test_path),
                "dropped_path": rel(dropped.test_path),
            }
        )
        by_test_sha[cand.test_sha] = chosen

    by_name: dict[str, Candidate] = {}
    for cand in by_test_sha.values():
        if cand.name not in by_name:
            by_name[cand.name] = cand
            continue
        suffix = cand.test_sha[:8]
        cand.name = f"{cand.name}__{suffix}"
        by_name[cand.name] = cand
    return sorted(by_name.values(), key=lambda item: item.name), duplicate_rows


def exact_triage(
    candidates: list[Candidate],
    qids: list[str],
    gold: dict[str, set[str]],
    public_anchor: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], Counter[str]]:
    pred_cache: dict[str, dict[str, set[str]]] = {
        cand.name: load_predictions(cand.pred_path) for cand in candidates
    }
    test_cache: dict[str, dict[str, set[str]]] = {
        cand.name: load_predictions(cand.test_path) for cand in candidates
    }
    per_query: dict[str, list[float]] = {
        cand.name: [f1(pred_cache[cand.name].get(qid, set()), gold[qid]) for qid in qids]
        for cand in candidates
    }
    private_size = len(qids) // 2
    masks = list(itertools.combinations(range(len(qids)), private_size))
    names = [cand.name for cand in candidates]
    private_scores = {name: [] for name in names}
    split_winners: list[str] = []
    split_best: list[float] = []
    for mask in masks:
        scores = {
            name: statistics.fmean(per_query[name][idx] for idx in mask)
            for name in names
        }
        winner, best = max(scores.items(), key=lambda item: (item[1], item[0]))
        split_winners.append(winner)
        split_best.append(best)
        for name, score in scores.items():
            private_scores[name].append(score)

    win_counts = Counter(split_winners)
    top2_counts: Counter[str] = Counter()
    for split_idx in range(len(masks)):
        ranked = sorted(
            ((private_scores[name][split_idx], name) for name in names),
            reverse=True,
        )
        for _, name in ranked[:2]:
            top2_counts[name] += 1

    anchor_preds = test_cache.get(public_anchor)
    candidate_rows: list[dict[str, str]] = []
    for cand in candidates:
        values = private_scores[cand.name]
        test_counts = [len(cites) for cites in test_cache[cand.name].values()]
        anchor_j = atom_jaccard(test_cache[cand.name], anchor_preds) if anchor_preds else math.nan
        candidate_rows.append(
            {
                "name": cand.name,
                "local_macro_f1": f"{statistics.fmean(per_query[cand.name]):.6f}",
                "exact_worst": f"{min(values):.6f}",
                "exact_p10": f"{percentile(values, 0.10):.6f}",
                "exact_mean": f"{statistics.fmean(values):.6f}",
                "exact_p90": f"{percentile(values, 0.90):.6f}",
                "exact_win_frac": f"{win_counts[cand.name] / len(masks):.6f}",
                "exact_top2_frac": f"{top2_counts[cand.name] / len(masks):.6f}",
                "test_total_cites": str(sum(test_counts)),
                "test_mean_cites": f"{statistics.fmean(test_counts):.6f}",
                "jaccard_vs_public_anchor": "" if math.isnan(anchor_j) else f"{anchor_j:.6f}",
                "public_score": cand.public_score,
                "pred_path": rel(cand.pred_path),
                "test_path": rel(cand.test_path),
                "note": cand.note,
                "sources": ";".join(cand.sources),
                "test_sha256": cand.test_sha,
            }
        )

    pair_rows: list[dict[str, str]] = []
    pair_counts: Counter[str] = Counter()
    for left, right in itertools.combinations(names, 2):
        values = [
            max(private_scores[left][idx], private_scores[right][idx])
            for idx in range(len(masks))
        ]
        contains = sum(
            1
            for winner in split_winners
            if winner == left or winner == right
        ) / len(masks)
        regret = statistics.fmean(split_best[idx] - values[idx] for idx in range(len(masks)))
        test_j = atom_jaccard(test_cache[left], test_cache[right])
        pair_rows.append(
            {
                "left": left,
                "right": right,
                "exact_pair_worst": f"{min(values):.6f}",
                "exact_pair_p10": f"{percentile(values, 0.10):.6f}",
                "exact_pair_mean": f"{statistics.fmean(values):.6f}",
                "exact_pair_p90": f"{percentile(values, 0.90):.6f}",
                "exact_contains_winner": f"{contains:.6f}",
                "exact_regret": f"{regret:.6f}",
                "test_jaccard": f"{test_j:.6f}",
            }
        )

    pair_rows.sort(
        key=lambda row: (
            -float(row["exact_contains_winner"]),
            -float(row["exact_pair_p10"]),
            -float(row["exact_pair_mean"]),
            float(row["test_jaccard"]),
        )
    )
    for row in pair_rows[:300]:
        pair_counts[row["left"]] += 1
        pair_counts[row["right"]] += 1

    candidate_rows.sort(
        key=lambda row: (
            -float(row["exact_top2_frac"]),
            -float(row["exact_win_frac"]),
            -float(row["exact_p10"]),
            -float(row["local_macro_f1"]),
        )
    )
    return candidate_rows, pair_rows, pair_counts


def choose_capped_candidates(
    candidates: list[Candidate],
    candidate_rows: list[dict[str, str]],
    pair_counts: Counter[str],
    args: argparse.Namespace,
) -> list[Candidate]:
    by_name = {cand.name: cand for cand in candidates}
    selected: dict[str, Candidate] = {}

    def add(name: str) -> None:
        cand = by_name.get(name)
        if cand is not None:
            selected[name] = cand

    for name in (
        MANDATORY_NAMES
        | {args.baseline_left, args.baseline_right, args.public_anchor}
        | set(args.force_name)
    ):
        add(name)

    # Exact half-val private behavior is the only non-public signal we can
    # measure. Mix individual stability with pair frequency so one weird metric
    # cannot dominate the capped Rust pass.
    for row in candidate_rows[: max(20, args.max_candidates // 2)]:
        add(row["name"])
    for name, _ in pair_counts.most_common(args.max_candidates):
        add(name)

    # Keep a few high-local candidates even if they are not frequent in top
    # pairs, because they may interact well under bucketed private schemes.
    high_local = sorted(candidate_rows, key=lambda row: -float(row["local_macro_f1"]))
    for row in high_local[: max(12, args.max_candidates // 5)]:
        add(row["name"])

    # Trim by an aggregate non-public score while preserving mandatory names.
    if len(selected) > args.max_candidates:
        score_by_name = {
            row["name"]: (
                3.0 * float(row["exact_top2_frac"])
                + 2.0 * float(row["exact_win_frac"])
                + float(row["exact_p10"])
                + 0.5 * float(row["local_macro_f1"])
                + 0.02 * pair_counts[row["name"]]
            )
            for row in candidate_rows
        }
        mandatory = {
            name: cand
            for name, cand in selected.items()
            if name in MANDATORY_NAMES
            or name in {
                args.baseline_left,
                args.baseline_right,
                args.public_anchor,
                *args.force_name,
            }
        }
        rest = [
            cand
            for name, cand in selected.items()
            if name not in mandatory
        ]
        rest.sort(key=lambda cand: score_by_name.get(cand.name, -1.0), reverse=True)
        selected = dict(mandatory)
        for cand in rest:
            if len(selected) >= args.max_candidates:
                break
            selected[cand.name] = cand

    return sorted(selected.values(), key=lambda item: item.name)


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, candidates: list[Candidate]) -> None:
    rows = [
        {
            "name": cand.name,
            "pred_path": rel(cand.pred_path),
            "test_path": rel(cand.test_path),
            "public_score": cand.public_score,
            "note": cand.note,
        }
        for cand in candidates
    ]
    write_tsv(path, rows)


def write_report(
    path: Path,
    full: list[Candidate],
    capped: list[Candidate],
    skipped: list[dict[str, str]],
    duplicates: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> None:
    with path.open("w") as f:
        f.write("# Private Final Candidate Universe\n\n")
        f.write("This is a read-only candidate-universe build for private final selection. ")
        f.write("It does not use Kaggle feedback and does not modify frozen submissions.\n\n")
        f.write("## Counts\n\n")
        f.write(f"- Full deduped candidates: `{len(full)}`\n")
        f.write(f"- Capped Rust candidates: `{len(capped)}`\n")
        f.write(f"- Skipped manifest rows: `{len(skipped)}`\n")
        f.write(f"- Duplicate test payloads collapsed: `{len(duplicates)}`\n\n")
        f.write("## Top Exact Half-Val Pairs\n\n")
        f.write("| Pair | Contains Winner | P10 | Mean | Jaccard |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in pair_rows[:20]:
            f.write(
                f"| `{row['left']}` + `{row['right']}` | `{row['exact_contains_winner']}` | "
                f"`{row['exact_pair_p10']}` | `{row['exact_pair_mean']}` | `{row['test_jaccard']}` |\n"
            )
        f.write("\n## Top Exact Half-Val Candidates\n\n")
        f.write("| Candidate | Top2 | Win | P10 | Local F1 | Test Cites | J vs Public |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in candidate_rows[:25]:
            f.write(
                f"| `{row['name']}` | `{row['exact_top2_frac']}` | `{row['exact_win_frac']}` | "
                f"`{row['exact_p10']}` | `{row['local_macro_f1']}` | `{row['test_total_cites']}` | "
                f"`{row['jaccard_vs_public_anchor']}` |\n"
            )
        f.write("\n## Baseline Names For Downstream Referee\n\n")
        f.write(f"- Left: `{args.baseline_left}`\n")
        f.write(f"- Right: `{args.baseline_right}`\n")
        f.write(f"- Public anchor: `{args.public_anchor}`\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifests = ([] if args.no_defaults else DEFAULT_MANIFESTS) + args.manifest
    markdown_manifests = list(
        [] if args.no_defaults else DEFAULT_MARKDOWN_MANIFESTS
    ) + args.markdown_manifest
    for pattern in ([] if args.no_defaults else args.markdown_glob):
        markdown_manifests.extend(
            rel(path) for path in sorted(REPO.glob(pattern)) if path.is_file()
        )
    markdown_manifests = sorted(dict.fromkeys(markdown_manifests))
    raw_candidates, skipped = load_tsv_manifest_rows(manifests)
    md_candidates, md_skipped = load_markdown_manifest_rows(markdown_manifests)
    raw_candidates.extend(md_candidates)
    skipped.extend(md_skipped)
    candidates, duplicates = dedupe_candidates(raw_candidates)
    qids, gold = load_gold(args.gold)
    candidate_rows, pair_rows, pair_counts = exact_triage(
        candidates,
        qids,
        gold,
        args.public_anchor,
    )
    capped = choose_capped_candidates(candidates, candidate_rows, pair_counts, args)

    write_manifest(args.out_dir / "combined_candidate_manifest_full.tsv", candidates)
    write_manifest(args.out_dir / "combined_candidate_manifest_capped.tsv", capped)
    write_tsv(args.out_dir / "candidate_exact_triage.tsv", candidate_rows)
    write_tsv(args.out_dir / "pair_exact_triage.tsv", pair_rows)
    write_tsv(args.out_dir / "skipped_manifest_rows.tsv", skipped)
    write_tsv(args.out_dir / "duplicate_test_payloads.tsv", duplicates)
    write_report(
        args.out_dir / "universe_report.md",
        candidates,
        capped,
        skipped,
        duplicates,
        candidate_rows,
        pair_rows,
        args,
    )
    print("build_private_final_universe complete")
    print(f"out_dir={args.out_dir}")
    print(f"full_candidates={len(candidates)} capped_candidates={len(capped)}")
    print(f"top_pair={pair_rows[0]['left']} + {pair_rows[0]['right']}")
    print(f"capped_manifest={args.out_dir / 'combined_candidate_manifest_capped.tsv'}")


if __name__ == "__main__":
    main()
