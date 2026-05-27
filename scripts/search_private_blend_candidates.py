#!/usr/bin/env python3
"""
Search deterministic private-hedge blend candidates.

This is deliberately not a public-LB search. It uses only local validation
companions and applies the same recipe to validation/test payloads. The goal is
to see whether simple, low-complexity portfolio blends can beat the current
private pair under exact half-val stress before spending Rust sweep time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Source:
    name: str
    pred_path: Path
    test_path: Path
    local_f1: float
    p10: float
    win_frac: float
    top2_frac: float
    j_public: float


@dataclass
class Generated:
    name: str
    recipe: str
    val: dict[str, set[str]]
    test: dict[str, set[str]]
    sources: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO
        / "artifacts/private_final_recheck_20260522/universe_v3_allstaff160/combined_candidate_manifest_full.tsv",
    )
    parser.add_argument(
        "--triage",
        type=Path,
        default=REPO
        / "artifacts/private_final_recheck_20260522/universe_v3_allstaff160/candidate_exact_triage.tsv",
    )
    parser.add_argument("--gold", type=Path, default=REPO / "data/val.csv")
    parser.add_argument("--anchor", default="intersect_bold7h_33028")
    parser.add_argument("--current-hedge", default="widebankG_hailmary_30702")
    parser.add_argument("--baseline-hedge", default="fusion_samesrc03_32274")
    parser.add_argument("--public-anchor", default="public_peak_33438")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "artifacts/private_final_recheck_20260522/private_blend_search_v1",
    )
    parser.add_argument("--write-top", type=int, default=40)
    return parser.parse_args()


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def parse_citations(value: str | None) -> set[str]:
    return {piece.strip() for piece in (value or "").split(";") if piece.strip()}


def join_citations(citations: Iterable[str]) -> str:
    return ";".join(sorted(citations))


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


def percentile(values: list[float], pct: float) -> float:
    xs = sorted(values)
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def atom_jaccard(left: dict[str, set[str]], right: dict[str, set[str]]) -> float:
    left_atoms = {(qid, cit) for qid, cites in left.items() for cit in cites}
    right_atoms = {(qid, cit) for qid, cites in right.items() for cit in cites}
    union = left_atoms | right_atoms
    return 1.0 if not union else len(left_atoms & right_atoms) / len(union)


def sha_predictions(preds: dict[str, set[str]]) -> str:
    h = hashlib.sha256()
    for qid in sorted(preds):
        h.update(qid.encode())
        h.update(b"\0")
        h.update(join_citations(preds[qid]).encode())
        h.update(b"\n")
    return h.hexdigest()


def load_sources(manifest: Path, triage: Path) -> dict[str, Source]:
    manifest_rows: dict[str, dict[str, str]] = {}
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            manifest_rows[row["name"]] = row
    sources: dict[str, Source] = {}
    with triage.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            name = row["name"]
            manifest_row = manifest_rows.get(name)
            if not manifest_row:
                continue
            pred_path = repo_path(manifest_row["pred_path"])
            test_path = repo_path(manifest_row["test_path"])
            if not pred_path.exists() or not test_path.exists():
                continue
            try:
                j_public = float(row.get("jaccard_vs_public_anchor") or "nan")
            except ValueError:
                j_public = math.nan
            sources[name] = Source(
                name=name,
                pred_path=pred_path,
                test_path=test_path,
                local_f1=float(row["local_macro_f1"]),
                p10=float(row["exact_p10"]),
                win_frac=float(row["exact_win_frac"]),
                top2_frac=float(row["exact_top2_frac"]),
                j_public=j_public,
            )
    return sources


def exact_pair_stats(
    qids: list[str],
    gold: dict[str, set[str]],
    left: dict[str, set[str]],
    right: dict[str, set[str]],
    all_candidate_scores: list[list[float]],
) -> dict[str, float]:
    left_scores = [f1(left.get(qid, set()), gold[qid]) for qid in qids]
    right_scores = [f1(right.get(qid, set()), gold[qid]) for qid in qids]
    private_size = len(qids) // 2
    values: list[float] = []
    regrets: list[float] = []
    for mask in itertools.combinations(range(len(qids)), private_size):
        pair_value = max(
            statistics.fmean(left_scores[idx] for idx in mask),
            statistics.fmean(right_scores[idx] for idx in mask),
        )
        best_value = max(
            statistics.fmean(scores[idx] for idx in mask)
            for scores in all_candidate_scores
        )
        values.append(pair_value)
        regrets.append(best_value - pair_value)
    return {
        "worst": min(values),
        "p10": percentile(values, 0.10),
        "mean": statistics.fmean(values),
        "p90": percentile(values, 0.90),
        "regret": statistics.fmean(regrets),
    }


def candidate_scores(
    qids: list[str],
    gold: dict[str, set[str]],
    preds: dict[str, set[str]],
) -> list[float]:
    return [f1(preds.get(qid, set()), gold[qid]) for qid in qids]


def support_map(
    qid: str,
    pool: list[Source],
    pred_cache: dict[str, dict[str, set[str]]],
    weights: dict[str, float],
) -> tuple[dict[str, float], float]:
    scores: dict[str, float] = {}
    total_weight = sum(weights[src.name] for src in pool)
    for src in pool:
        weight = weights[src.name]
        for cit in pred_cache[src.name].get(qid, set()):
            scores[cit] = scores.get(cit, 0.0) + weight
    return scores, total_weight


def make_vote_candidate(
    name: str,
    recipe: str,
    qids: Iterable[str],
    pool: list[Source],
    pred_cache: dict[str, dict[str, set[str]]],
    test_cache: dict[str, dict[str, set[str]]],
    weights: dict[str, float],
    threshold: float,
) -> Generated:
    def build(cache: dict[str, dict[str, set[str]]], ids: Iterable[str]) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for qid in ids:
            scores, total = support_map(qid, pool, cache, weights)
            out[qid] = {
                cit for cit, score in scores.items() if total > 0 and score / total >= threshold
            }
        return out

    test_ids = next(iter(test_cache.values())).keys()
    return Generated(name, recipe, build(pred_cache, qids), build(test_cache, test_ids), [src.name for src in pool])


def make_base_candidate(
    name: str,
    recipe: str,
    qids: Iterable[str],
    base: Source,
    pool: list[Source],
    pred_cache: dict[str, dict[str, set[str]]],
    test_cache: dict[str, dict[str, set[str]]],
    weights: dict[str, float],
    keep_threshold: float,
    add_threshold: float,
) -> Generated:
    def build(cache: dict[str, dict[str, set[str]]], ids: Iterable[str]) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for qid in ids:
            base_cites = set(cache[base.name].get(qid, set()))
            scores, total = support_map(qid, pool, cache, weights)
            pool_cites = set(scores)
            kept = {
                cit
                for cit in base_cites
                if keep_threshold <= 0.0
                or (total > 0 and scores.get(cit, 0.0) / total >= keep_threshold)
            }
            added = {
                cit
                for cit in pool_cites - base_cites
                if total > 0 and scores.get(cit, 0.0) / total >= add_threshold
            }
            out[qid] = kept | added
        return out

    test_ids = next(iter(test_cache.values())).keys()
    return Generated(
        name,
        recipe,
        build(pred_cache, qids),
        build(test_cache, test_ids),
        [base.name] + [src.name for src in pool],
    )


def write_submission(path: Path, preds: dict[str, set[str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(preds):
            writer.writerow([qid, join_citations(preds[qid])])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    qids, gold = load_gold(args.gold)
    sources = load_sources(args.manifest, args.triage)
    required = [args.anchor, args.current_hedge, args.baseline_hedge, args.public_anchor]
    missing = [name for name in required if name not in sources]
    if missing:
        raise SystemExit(f"missing required sources: {missing}")

    ordered = sorted(
        sources.values(),
        key=lambda src: (-src.top2_frac, -src.win_frac, -src.p10, -src.local_f1, src.name),
    )
    pred_cache = {src.name: load_predictions(src.pred_path) for src in sources.values()}
    test_cache = {src.name: load_predictions(src.test_path) for src in sources.values()}
    existing_shas = {sha_predictions(test_cache[src.name]) for src in sources.values()}

    pools: dict[str, list[Source]] = {}
    pools["top6"] = ordered[:6]
    pools["top10"] = ordered[:10]
    pools["top16"] = ordered[:16]
    pools["winners"] = [src for src in ordered if src.win_frac > 0 or src.top2_frac >= 0.03][:18]
    pools["nonclone"] = [
        src
        for src in ordered
        if (math.isnan(src.j_public) or src.j_public < 0.95) and src.local_f1 >= 0.52
    ][:20]
    pools["widebank"] = [
        src for src in ordered if "widebank" in src.name.lower() or "bold7h_gsymp" in src.name
    ][:18]
    pools["fusion"] = [
        src
        for src in ordered
        if any(piece in src.name.lower() for piece in ["fusion", "overlay", "rubik"])
    ][:18]
    pools["samesrc"] = [src for src in ordered if "samesrc" in src.name.lower()][:18]
    pools = {name: pool for name, pool in pools.items() if len(pool) >= 2}

    weight_schemes: dict[str, callable[[Source], float]] = {
        "uniform": lambda src: 1.0,
        "private": lambda src: max(0.01, 2.0 * src.top2_frac + src.win_frac + src.p10),
        "p10": lambda src: max(0.01, src.p10),
        "diverse": lambda src: max(0.01, (1.0 - (src.j_public if not math.isnan(src.j_public) else 0.90))),
    }
    generated: list[Generated] = []
    for pool_name, pool in pools.items():
        for weight_name, weight_fn in weight_schemes.items():
            weights = {src.name: weight_fn(src) for src in pool}
            for threshold in [0.18, 0.24, 0.30, 0.36, 0.42, 0.50, 0.60]:
                generated.append(
                    make_vote_candidate(
                        f"vote_{pool_name}_{weight_name}_t{int(threshold*100):02d}",
                        f"vote pool={pool_name} weights={weight_name} threshold={threshold:.2f}",
                        qids,
                        pool,
                        pred_cache,
                        test_cache,
                        weights,
                        threshold,
                    )
                )
            for base_name in [args.current_hedge, args.anchor, args.baseline_hedge, args.public_anchor]:
                base = sources[base_name]
                for keep_threshold in [0.0, 0.10, 0.18, 0.24, 0.30]:
                    for add_threshold in [0.42, 0.50, 0.60, 0.70]:
                        generated.append(
                            make_base_candidate(
                                f"base_{base_name}_{pool_name}_{weight_name}_k{int(keep_threshold*100):02d}_a{int(add_threshold*100):02d}",
                                (
                                    f"base={base_name} pool={pool_name} weights={weight_name} "
                                    f"keep={keep_threshold:.2f} add={add_threshold:.2f}"
                                ),
                                qids,
                                base,
                                pool,
                                pred_cache,
                                test_cache,
                                weights,
                                keep_threshold,
                                add_threshold,
                            )
                        )

    all_candidate_scores = [
        candidate_scores(qids, gold, pred_cache[src.name]) for src in sources.values()
    ]
    anchor_val = pred_cache[args.anchor]
    current_val = pred_cache[args.current_hedge]
    baseline_val = pred_cache[args.baseline_hedge]
    public_test = test_cache[args.public_anchor]
    anchor_test = test_cache[args.anchor]
    current_pair = exact_pair_stats(qids, gold, anchor_val, current_val, all_candidate_scores)
    baseline_pair = exact_pair_stats(qids, gold, anchor_val, baseline_val, all_candidate_scores)

    rows: list[dict[str, str]] = []
    kept: list[Generated] = []
    seen: set[str] = set(existing_shas)
    for gen in generated:
        test_sha = sha_predictions(gen.test)
        if test_sha in seen:
            continue
        seen.add(test_sha)
        local_scores = candidate_scores(qids, gold, gen.val)
        pair = exact_pair_stats(qids, gold, anchor_val, gen.val, all_candidate_scores)
        test_counts = [len(cites) for cites in gen.test.values()]
        row = {
            "name": gen.name,
            "recipe": gen.recipe,
            "local_macro_f1": f"{statistics.fmean(local_scores):.6f}",
            "exact_pair_worst": f"{pair['worst']:.6f}",
            "exact_pair_p10": f"{pair['p10']:.6f}",
            "exact_pair_mean": f"{pair['mean']:.6f}",
            "exact_pair_p90": f"{pair['p90']:.6f}",
            "exact_regret": f"{pair['regret']:.6f}",
            "delta_vs_current_p10": f"{pair['p10'] - current_pair['p10']:.6f}",
            "delta_vs_current_mean": f"{pair['mean'] - current_pair['mean']:.6f}",
            "delta_vs_current_regret": f"{pair['regret'] - current_pair['regret']:.6f}",
            "delta_vs_baseline_p10": f"{pair['p10'] - baseline_pair['p10']:.6f}",
            "delta_vs_baseline_mean": f"{pair['mean'] - baseline_pair['mean']:.6f}",
            "jaccard_vs_anchor": f"{atom_jaccard(gen.test, anchor_test):.6f}",
            "jaccard_vs_public": f"{atom_jaccard(gen.test, public_test):.6f}",
            "test_total_cites": str(sum(test_counts)),
            "test_mean_cites": f"{statistics.fmean(test_counts):.6f}",
            "test_sha": test_sha,
            "sources": ";".join(gen.sources),
        }
        rows.append(row)
        kept.append(gen)

    rows_by_name = {row["name"]: row for row in rows}
    ranked = sorted(
        kept,
        key=lambda gen: (
            -float(rows_by_name[gen.name]["exact_pair_p10"]),
            -float(rows_by_name[gen.name]["exact_pair_mean"]),
            float(rows_by_name[gen.name]["exact_regret"]),
            float(rows_by_name[gen.name]["jaccard_vs_anchor"]),
        ),
    )
    top = ranked[: args.write_top]
    manifest_rows: list[dict[str, str]] = []
    submissions_dir = args.out_dir / "candidates"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    for gen in top:
        val_path = submissions_dir / f"{gen.name}_val.csv"
        test_path = submissions_dir / f"{gen.name}_test.csv"
        write_submission(val_path, gen.val)
        write_submission(test_path, gen.test)
        manifest_rows.append(
            {
                "name": gen.name,
                "pred_path": rel(val_path),
                "test_path": rel(test_path),
                "public_score": "",
                "note": rows_by_name[gen.name]["recipe"],
            }
        )

    with (args.out_dir / "blend_search_results.tsv").open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["name"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(
            sorted(
                rows,
                key=lambda row: (
                    -float(row["exact_pair_p10"]),
                    -float(row["exact_pair_mean"]),
                    float(row["exact_regret"]),
                    float(row["jaccard_vs_anchor"]),
                ),
            )
        )
    with (args.out_dir / "blend_candidate_manifest.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "pred_path", "test_path", "public_score", "note"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (args.out_dir / "blend_search_report.md").open("w") as f:
        f.write("# Private Blend Search\n\n")
        f.write("All candidates are deterministic val/test blends; no Kaggle public feedback is used.\n\n")
        f.write("## Baselines\n\n")
        f.write(
            f"- Current pair `{args.anchor}` + `{args.current_hedge}`: "
            f"p10 `{current_pair['p10']:.6f}`, mean `{current_pair['mean']:.6f}`, regret `{current_pair['regret']:.6f}`\n"
        )
        f.write(
            f"- Old pair `{args.anchor}` + `{args.baseline_hedge}`: "
            f"p10 `{baseline_pair['p10']:.6f}`, mean `{baseline_pair['mean']:.6f}`, regret `{baseline_pair['regret']:.6f}`\n\n"
        )
        f.write("## Top Generated Candidates\n\n")
        f.write("| Candidate | Pair P10 | Pair Mean | Regret | Delta P10 | J Anchor | J Public | Cites | Recipe |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for gen in top[:20]:
            row = rows_by_name[gen.name]
            f.write(
                f"| `{gen.name}` | `{row['exact_pair_p10']}` | `{row['exact_pair_mean']}` | "
                f"`{row['exact_regret']}` | `{row['delta_vs_current_p10']}` | "
                f"`{row['jaccard_vs_anchor']}` | `{row['jaccard_vs_public']}` | "
                f"`{row['test_total_cites']}` | {row['recipe']} |\n"
            )
    print("search_private_blend_candidates complete")
    print(f"out_dir={args.out_dir}")
    print(f"generated_unique={len(rows)} wrote_top={len(top)}")
    if top:
        best = rows_by_name[top[0].name]
        print(
            "best="
            f"{top[0].name} p10={best['exact_pair_p10']} mean={best['exact_pair_mean']} "
            f"delta_p10={best['delta_vs_current_p10']}"
        )


if __name__ == "__main__":
    main()
