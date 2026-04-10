#!/usr/bin/env python3
"""
Merge 3 judge passes into consensus features per candidate.

For each candidate across all queries, produces:
  - judge_agree_count: how many passes labeled must_include or plausible
  - judge_must_count: how many passes labeled must_include
  - mean_confidence: average judge_confidence across passes
  - min_confidence: minimum judge_confidence across passes
  - max_confidence: maximum judge_confidence across passes
  - consensus_label: must_include if majority says so, else plausible/reject

Writes enriched judged_bundles JSON that the Rust CV sweep can consume.

Usage:
    .venv/bin/python scripts/merge_multipass.py \
        --pass default=artifacts/v11/train_v11_strict_v1 \
        --pass enriched=artifacts/v11/train_v11_multipass_enriched \
        --pass strict=artifacts/v11/train_v11_multipass_strict \
        --gold data/train.csv \
        --output artifacts/v11_multipass_merged
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def load_all_chunks(artifact_dir: Path) -> dict:
    """Load and merge all chunk judged_bundles.json from a directory."""
    bundles_by_qid = {}
    chunk_dirs = sorted(
        [d for d in artifact_dir.iterdir() if d.is_dir() and d.name.startswith("train_")],
        key=lambda d: d.name
    )

    for chunk_dir in chunk_dirs:
        json_path = chunk_dir / "judged_bundles.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)
        for bundle in data.get("bundles", []):
            qid = bundle["query_id"]
            bundles_by_qid[qid] = bundle

    return bundles_by_qid


def merge_passes(
    pass_data: dict[str, dict[str, dict]],
    pass_names: list[str],
) -> dict[str, dict]:
    """Merge candidate labels from multiple passes."""
    # Primary pass determines the candidate list
    primary = pass_names[0]
    primary_bundles = pass_data[primary]

    merged_bundles = {}

    for qid, bundle in primary_bundles.items():
        merged_candidates = []

        for candidate in bundle.get("candidates", []):
            cite = candidate["citation"]

            # Collect labels from all passes
            labels = []
            confidences = []

            for pass_name in pass_names:
                other_bundle = pass_data.get(pass_name, {}).get(qid)
                if not other_bundle:
                    continue
                # Find matching candidate in other pass
                for other_c in other_bundle.get("candidates", []):
                    if other_c["citation"] == cite:
                        label = other_c.get("judge_label", "reject") or "reject"
                        conf = other_c.get("judge_confidence", 0.0) or 0.0
                        labels.append(label)
                        confidences.append(conf)
                        break
                else:
                    # Candidate not in this pass (different topk)
                    pass

            n_passes = len(labels)
            must_count = sum(1 for l in labels if l == "must_include")
            plausible_count = sum(1 for l in labels if l == "plausible")
            positive_count = must_count + plausible_count

            # Consensus label
            if must_count > n_passes / 2:
                consensus = "must_include"
            elif positive_count > n_passes / 2:
                consensus = "plausible"
            else:
                consensus = "reject"

            enriched = dict(candidate)
            enriched["judge_passes"] = n_passes
            enriched["judge_must_count"] = must_count
            enriched["judge_positive_count"] = positive_count
            enriched["mean_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
            enriched["min_confidence"] = min(confidences) if confidences else 0.0
            enriched["max_confidence"] = max(confidences) if confidences else 0.0
            enriched["consensus_label"] = consensus
            # Override judge_label with consensus for downstream selectors
            enriched["original_judge_label"] = candidate.get("judge_label")
            enriched["judge_label"] = consensus
            enriched["judge_confidence"] = enriched["mean_confidence"]

            merged_candidates.append(enriched)

        merged_bundle = dict(bundle)
        merged_bundle["candidates"] = merged_candidates
        merged_bundles[qid] = merged_bundle

    return merged_bundles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass", dest="passes", action="append", required=True,
                        help="name=artifacts_dir, repeat for each pass")
    parser.add_argument("--gold", type=Path, default=BASE / "data/train.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Parse pass arguments
    pass_names = []
    pass_dirs = {}
    for p in args.passes:
        name, path = p.split("=", 1)
        pass_names.append(name)
        pass_dirs[name] = Path(path)

    print(f"Loading {len(pass_names)} passes: {pass_names}")

    # Load all passes
    pass_data = {}
    for name in pass_names:
        artifact_dir = pass_dirs[name]
        if artifact_dir.is_dir():
            bundles = load_all_chunks(artifact_dir)
        else:
            # Single file
            with open(artifact_dir) as f:
                data = json.load(f)
            bundles = {b["query_id"]: b for b in data.get("bundles", [])}
        pass_data[name] = bundles
        print(f"  {name}: {len(bundles)} bundles")

    # Merge
    merged = merge_passes(pass_data, pass_names)
    print(f"\nMerged: {len(merged)} bundles with consensus features")

    # Load gold for rows
    rows = []
    with open(args.gold) as f:
        for row in csv.DictReader(f):
            rows.append({"query_id": row["query_id"], "gold_citations": row.get("gold_citations", "")})

    # Build output artifact (compatible with Rust cv_sweep)
    output_artifact = {
        "config": {
            "min_output": 10,
            "max_output": 40,
            "court_fraction": 0.25,
            "min_courts_if_any": 4,
            "must_keep_confidence": 0.86,
        },
        "rows": rows,
        "bundles": list(merged.values()),
    }

    args.output.mkdir(parents=True, exist_ok=True)

    # Write as single merged file
    out_path = args.output / "judged_bundles.json"
    with open(out_path, "w") as f:
        json.dump(output_artifact, f, ensure_ascii=False)
    print(f"Wrote merged artifact: {out_path} ({os.path.getsize(out_path) / 1024 / 1024:.1f} MB)")

    # Stats
    total_candidates = sum(len(b["candidates"]) for b in merged.values())
    must_consensus = sum(
        1 for b in merged.values()
        for c in b["candidates"]
        if c.get("consensus_label") == "must_include"
    )
    plausible_consensus = sum(
        1 for b in merged.values()
        for c in b["candidates"]
        if c.get("consensus_label") == "plausible"
    )
    reject_consensus = sum(
        1 for b in merged.values()
        for c in b["candidates"]
        if c.get("consensus_label") == "reject"
    )

    print(f"\nConsensus stats ({total_candidates} candidates):")
    print(f"  must_include: {must_consensus} ({must_consensus/total_candidates*100:.1f}%)")
    print(f"  plausible:    {plausible_consensus} ({plausible_consensus/total_candidates*100:.1f}%)")
    print(f"  reject:       {reject_consensus} ({reject_consensus/total_candidates*100:.1f}%)")

    # Agreement stats
    full_agree = sum(
        1 for b in merged.values()
        for c in b["candidates"]
        if c.get("judge_positive_count", 0) == c.get("judge_passes", 1)
        or c.get("judge_positive_count", 0) == 0
    )
    print(f"  Full agreement: {full_agree} ({full_agree/total_candidates*100:.1f}%)")


if __name__ == "__main__":
    main()
