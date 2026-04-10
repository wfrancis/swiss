from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline_v11 import (
    BASE,
    CourtDenseCache,
    CourtTextStore,
    JudgeCache,
    V11Config,
    annotate_snippets,
    apply_judgments,
    bucket_candidates,
    evaluate_predictions,
    gather_court_citations_for_judge,
    generate_candidates_for_row,
    load_assets,
    load_rows,
    select_candidates,
)

ARTIFACT_VERSION = 1


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def slice_suffix(config: V11Config) -> str:
    if config.query_offset == 0 and config.max_queries is None:
        return ""
    limit_part = "all" if config.max_queries is None else str(config.max_queries)
    return f"__offset{config.query_offset}_n{limit_part}"


def artifact_root(config: V11Config) -> Path:
    return BASE / "artifacts" / "v11" / f"{config.split}_{sanitize_name(config.prompt_version)}{slice_suffix(config)}"


def candidate_artifact_path(config: V11Config) -> Path:
    return artifact_root(config) / "candidate_bundles.pkl"


def judged_artifact_path(config: V11Config) -> Path:
    return artifact_root(config) / "judged_bundles.pkl"


def judged_json_path(config: V11Config) -> Path:
    return artifact_root(config) / "judged_bundles.json"


def manifest_path(config: V11Config) -> Path:
    return artifact_root(config) / "manifest.json"


def config_snapshot(config: V11Config) -> dict[str, Any]:
    return {
        "split": config.split,
        "use_judge": config.use_judge,
        "judge_model": config.judge_model,
        "prompt_version": config.prompt_version,
        "law_judge_topk": config.law_judge_topk,
        "court_judge_topk": config.court_judge_topk,
        "law_batch_size": config.law_batch_size,
        "court_batch_size": config.court_batch_size,
        "use_court_dense": config.use_court_dense,
        "court_dense_query_limit": config.court_dense_query_limit,
        "court_dense_topk": config.court_dense_topk,
        "max_output": config.max_output,
        "min_output": config.min_output,
        "court_fraction": config.court_fraction,
        "min_courts_if_any": config.min_courts_if_any,
        "must_keep_confidence": config.must_keep_confidence,
        "query_offset": config.query_offset,
        "max_queries": config.max_queries,
    }


def write_manifest(config: V11Config, extra: dict[str, Any]) -> None:
    root = artifact_root(config)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "config": config_snapshot(config),
        **extra,
    }
    manifest_path(config).write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def save_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def candidate_to_json(candidate: Any) -> dict[str, Any]:
    return {
        "citation": candidate.citation,
        "kind": candidate.kind,
        "raw_score": candidate.raw_score,
        "final_score": candidate.final_score,
        "baseline_rank": candidate.baseline_rank,
        "sources": candidate.sources,
        "gpt_full_freq": candidate.gpt_full_freq,
        "dense_rank": candidate.dense_rank,
        "court_dense_rank": candidate.court_dense_rank,
        "bm25_rank": candidate.bm25_rank,
        "is_explicit": candidate.is_explicit,
        "is_query_case": candidate.is_query_case,
        "snippet": candidate.snippet,
        "judge_label": candidate.judge_label,
        "judge_confidence": candidate.judge_confidence,
        "judge_reason": candidate.judge_reason,
        "auto_bucket": candidate.auto_bucket,
    }


def bundle_to_json(bundle: Any) -> dict[str, Any]:
    return {
        "query_id": bundle.query_id,
        "query": bundle.query,
        "estimated_count": bundle.estimated_count,
        "candidates": [candidate_to_json(candidate) for candidate in bundle.candidates],
    }


def default_output_path(split: str) -> Path:
    if split == "val":
        return BASE / "submissions" / "val_pred_v11_staged.csv"
    if split == "train":
        return BASE / "submissions" / "train_pred_v11_staged.csv"
    return BASE / "submissions" / "test_submission_v11_staged.csv"


def build_stage(split: str) -> Path:
    t0 = time.time()
    config = V11Config.from_env(split)
    print(
        (
            f"V11 staged build: split={split}, prompt={config.prompt_version}, "
            f"law_topk={config.law_judge_topk}, court_topk={config.court_judge_topk}, "
            f"offset={config.query_offset}, max_queries={config.max_queries}"
        ),
        flush=True,
    )
    assets = load_assets(split)
    rows = load_rows(split, max_queries=config.max_queries, query_offset=config.query_offset)
    print(f"Building bundles for {len(rows)} {split} queries...", flush=True)

    court_dense_cache = CourtDenseCache(config.court_dense_cache_path)
    bundles = []
    for row in rows:
        bundle = generate_candidates_for_row(row, assets, config, court_dense_cache)
        bucket_candidates(bundle, config)
        bundles.append(bundle)

    court_text_store = CourtTextStore(config.court_text_cache_path)
    needed_courts = gather_court_citations_for_judge(bundles)
    if needed_courts:
        print(f"Preparing {len(needed_courts)} court snippets for build artifact...", flush=True)
        court_text_store.ensure(needed_courts)
    annotate_snippets(bundles, assets, court_text_store)

    output_path = candidate_artifact_path(config)
    save_pickle(
        output_path,
        {
            "artifact_version": ARTIFACT_VERSION,
            "stage": "build",
            "config": config_snapshot(config),
            "rows": rows,
            "bundles": bundles,
        },
    )
    write_manifest(
        config,
        {
            "last_stage": "build",
            "candidate_artifact": str(output_path),
            "query_count": len(rows),
            "bundle_count": len(bundles),
            "elapsed_seconds": round(time.time() - t0, 2),
        },
    )
    print(f"Saved candidate bundles to {output_path}")
    print(f"Build stage time: {time.time() - t0:.0f}s")
    return output_path


def judge_stage(split: str) -> Path:
    t0 = time.time()
    config = V11Config.from_env(split)
    input_path = candidate_artifact_path(config)
    payload = load_pickle(input_path)
    bundles = payload["bundles"]
    judge_workers = max(1, int(os.getenv("V11_JUDGE_WORKERS", "1")))
    print(
        (
            f"V11 staged judge: split={split}, prompt={config.prompt_version}, "
            f"bundles={len(bundles)}, workers={judge_workers}"
        ),
        flush=True,
    )

    judge_cache = JudgeCache(config.cache_path)
    if judge_workers <= 1:
        for idx, bundle in enumerate(bundles, start=1):
            apply_judgments(bundle, config, judge_cache)
            print(f"  judged {idx}/{len(bundles)}: {bundle.query_id}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=judge_workers) as executor:
            future_map = {
                executor.submit(apply_judgments, bundle, config, judge_cache): bundle
                for bundle in bundles
            }
            for idx, future in enumerate(as_completed(future_map), start=1):
                bundle = future_map[future]
                future.result()
                print(f"  judged {idx}/{len(bundles)}: {bundle.query_id}", flush=True)

    output_path = judged_artifact_path(config)
    output_json_path = judged_json_path(config)
    payload["stage"] = "judged"
    save_pickle(output_path, payload)
    output_json_path.write_text(
        json.dumps(
            {
                "artifact_version": ARTIFACT_VERSION,
                "stage": "judged",
                "config": payload["config"],
                "rows": payload["rows"],
                "bundles": [bundle_to_json(bundle) for bundle in bundles],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    write_manifest(
        config,
        {
            "last_stage": "judged",
            "candidate_artifact": str(input_path),
            "judged_artifact": str(output_path),
            "judged_json": str(output_json_path),
            "query_count": len(payload["rows"]),
            "bundle_count": len(bundles),
            "elapsed_seconds": round(time.time() - t0, 2),
        },
    )
    print(f"Saved judged bundles to {output_path}")
    print(f"Saved judged JSON to {output_json_path}")
    print(f"Judge stage time: {time.time() - t0:.0f}s")
    return output_path


def select_stage(split: str, output_path: Path | None = None) -> tuple[Path, dict[str, set[str]], float | None]:
    t0 = time.time()
    config = V11Config.from_env(split)
    input_path = judged_artifact_path(config)
    payload = load_pickle(input_path)
    rows = payload["rows"]
    bundles = payload["bundles"]
    predictions: dict[str, set[str]] = {}

    evaluate = split == "val"
    gold_map = {}
    if evaluate:
        gold_map = {row["query_id"]: set(row["gold_citations"].split(";")) for row in rows}

    for bundle in bundles:
        selected = select_candidates(bundle, config)
        predictions[bundle.query_id] = {candidate.citation for candidate in selected}

        must_count = sum(1 for candidate in bundle.candidates if candidate.judge_label == "must_include")
        plausible_count = sum(1 for candidate in bundle.candidates if candidate.judge_label == "plausible")
        reject_count = sum(1 for candidate in bundle.candidates if candidate.judge_label == "reject")
        line = (
            f"  {bundle.query_id}: auto_keep={len(bundle.auto_keep)}, judge="
            f"{len(bundle.judge_laws) + len(bundle.judge_courts)}, selected={len(selected)} "
            f"(must={must_count}, plausible={plausible_count}, reject={reject_count})"
        )
        if evaluate:
            gold = gold_map[bundle.query_id]
            pred = predictions[bundle.query_id]
            tp = len(pred & gold)
            precision = tp / len(pred) if pred else 0.0
            recall = tp / len(gold) if gold else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            line += f" | gold={len(gold)}, TP={tp}, P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}"
        print(line, flush=True)

    if output_path is None:
        output_path = default_output_path(split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions):
            writer.writerow([qid, ";".join(sorted(predictions[qid]))])

    macro_f1 = None
    if evaluate:
        macro_f1, _ = evaluate_predictions(predictions, gold_map)
        print(f"\n=== V11 STAGED MACRO F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%) ===")

    avg_predictions = sum(len(pred) for pred in predictions.values()) / len(predictions) if predictions else 0.0
    write_manifest(
        config,
        {
            "last_stage": "selected",
            "candidate_artifact": str(candidate_artifact_path(config)),
            "judged_artifact": str(input_path),
            "output_csv": str(output_path),
            "query_count": len(rows),
            "average_predictions": round(avg_predictions, 3),
            "macro_f1": round(macro_f1, 6) if macro_f1 is not None else None,
            "elapsed_seconds": round(time.time() - t0, 2),
        },
    )
    print(f"Saved to {output_path}")
    print(f"Average predictions: {avg_predictions:.1f}")
    print(f"Select stage time: {time.time() - t0:.0f}s")
    return output_path, predictions, macro_f1


def full_stage(split: str, output_path: Path | None = None) -> tuple[Path, dict[str, set[str]], float | None]:
    build_stage(split)
    judge_stage(split)
    return select_stage(split, output_path=output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V11 pipeline in explicit stages.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ["build", "judge", "select", "full"]:
        sub = subparsers.add_parser(name)
        sub.add_argument("--split", choices=["train", "val", "test"], required=True)
        sub.add_argument("--offset", type=int, default=0)
        sub.add_argument("--limit", type=int)
        if name in {"select", "full"}:
            sub.add_argument("--output", type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["V11_QUERY_OFFSET"] = str(args.offset)
    if args.limit is None:
        os.environ.pop("V11_MAX_QUERIES", None)
    else:
        os.environ["V11_MAX_QUERIES"] = str(args.limit)
    if args.command == "build":
        build_stage(args.split)
    elif args.command == "judge":
        judge_stage(args.split)
    elif args.command == "select":
        select_stage(args.split, output_path=args.output)
    elif args.command == "full":
        full_stage(args.split, output_path=args.output)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
