#!/usr/bin/env python3
"""Cross-encoder rerank over court-dense bundles (BGE-reranker-v2-m3).

For each query, score EVERY (query, candidate_text) pair in the candidate pool
with a multilingual cross-encoder. Save raw scores to
artifacts/ce_rerank/{split}_scores.json so downstream tuning (K sweep,
ensembling) is cheap.

Candidate text comes from artifacts/citation_text_index.pkl (100% coverage over
court-dense pools; fallback to bundle snippet, then citation string).

Hypothesis: structurally different from LLM-as-selector (deterministic scalar
scorer, no verbosity bias, trained on millions of relevance pairs, operates
over the FULL ~2500 pool not the top-120). Pool-oracle F1 at full pool is
0.77 vs 0.628 at K=120, so uncapping the pool matters.

Usage:
  python scripts/ce_rerank_v1.py --split val
  python scripts/ce_rerank_v1.py --split test
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ARTIFACT_PATHS = {
    "val":  REPO / "artifacts/v11/val_v11_courtdense_ds_val1/judged_bundles.json",
    "test": REPO / "artifacts/v11/test_v11_courtdense_ds_test1/judged_bundles.json",
}
CITE_TEXT_INDEX = REPO / "artifacts/citation_text_index.pkl"
OUT_DIR = REPO / "artifacts/ce_rerank"


def build_query_text(query: str, max_chars: int = 1800) -> str:
    """Trim the query if it is extremely long. BGE-reranker-v2-m3 has
    max_length=512 tokens (~2000 chars of legal German/English); anything
    beyond that is truncated anyway — keep the head where the legal question
    is stated."""
    q = (query or "").strip().replace("\n", " ")
    return q if len(q) <= max_chars else q[:max_chars]


def build_candidate_text(
    cite: str, snippet: str, cite_text: dict[str, str], max_chars: int = 1400
) -> str:
    """Prefer corpus snippet from citation_text_index. Fall back to bundle
    snippet, then the citation string itself."""
    text = cite_text.get(cite) or (snippet or "") or cite
    text = text.replace("\n", " ").strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    # Prepend the citation label so the model can still anchor on it even if
    # the corpus text is very long.
    return f"{cite}\n{text}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--query-max-chars", type=int, default=1800)
    ap.add_argument("--cand-max-chars", type=int, default=1400)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap queries (smoke)")
    args = ap.parse_args()

    path = ARTIFACT_PATHS[args.split]
    print(f"Loading bundles: {path}", flush=True)
    bundles = json.loads(path.read_text())["bundles"]
    if args.limit:
        bundles = bundles[: args.limit]
    total_pairs = sum(len(b["candidates"]) for b in bundles)
    print(f"  {len(bundles)} queries, {total_pairs:,} total pairs", flush=True)

    # Collect only the cites we actually need (dramatically reduces memory:
    # ~100-150k cites for a split vs 2.16M in the full index)
    needed_cites: set[str] = set()
    for b in bundles:
        for c in b["candidates"]:
            needed_cites.add(c["citation"])
    print(f"  {len(needed_cites):,} unique cites needed", flush=True)

    t_load = time.time()
    print(f"Loading cite text index (filtered): {CITE_TEXT_INDEX}", flush=True)
    with CITE_TEXT_INDEX.open("rb") as f:
        full_index: dict[str, str] = pickle.load(f)
    cite_text: dict[str, str] = {k: full_index[k] for k in needed_cites if k in full_index}
    del full_index
    import gc; gc.collect()
    print(f"  {len(cite_text):,} cites loaded  ({time.time()-t_load:.1f}s)", flush=True)

    t_load = time.time()
    print(f"Loading model: {args.model}  (device={args.device}, max_length={args.max_length})", flush=True)
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(args.model, device=args.device, max_length=args.max_length)
    print(f"  loaded in {time.time()-t_load:.1f}s", flush=True)

    # Warm up the model once so per-query timings are meaningful
    _ = model.predict([("warmup query", "warmup text")], batch_size=args.batch_size, show_progress_bar=False)

    out_path = args.out or (OUT_DIR / f"{args.split}_scores.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(".partial.json")

    # Resume: load any existing partial scores so we skip already-done qids
    scores_by_qid: dict[str, dict[str, float]] = {}
    if partial_path.exists():
        try:
            scores_by_qid = json.loads(partial_path.read_text())
            print(f"Resuming from {partial_path}: {len(scores_by_qid)} queries already scored", flush=True)
        except Exception as e:
            print(f"Could not load partial ({e}); starting fresh", flush=True)
            scores_by_qid = {}

    pairs_done = sum(len(v) for v in scores_by_qid.values())
    t_start = time.time()

    for b in bundles:
        qid = b["query_id"]
        if qid in scores_by_qid:
            print(f"  [{qid}] already scored ({len(scores_by_qid[qid])} cites), skipping", flush=True)
            continue
        query = build_query_text(b["query"], args.query_max_chars)
        cands = b["candidates"]
        pairs = []
        cites = []
        for c in cands:
            cite = c["citation"]
            text = build_candidate_text(
                cite,
                c.get("snippet") or "",
                cite_text,
                args.cand_max_chars,
            )
            pairs.append((query, text))
            cites.append(cite)

        t_q = time.time()
        scores = model.predict(
            pairs,
            batch_size=args.batch_size,
            show_progress_bar=False,
        )
        scores_by_qid[qid] = {cite: float(s) for cite, s in zip(cites, scores)}
        pairs_done += len(pairs)
        dt_q = time.time() - t_q
        elapsed = time.time() - t_start
        rate = pairs_done / elapsed if elapsed > 0 else 0
        eta_min = (total_pairs - pairs_done) / rate / 60 if rate > 0 else 0

        # CHECKPOINT: write partial after every query (atomic rename)
        tmp_path = partial_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(scores_by_qid))
        tmp_path.replace(partial_path)

        print(
            f"  [{qid}] {len(pairs)} pairs in {dt_q:.1f}s  ({len(pairs)/dt_q:.0f}/s)  "
            f"| {pairs_done:,}/{total_pairs:,}  elapsed={elapsed/60:.1f}m  ETA={eta_min:.1f}m  "
            f"[checkpointed]",
            flush=True,
        )

    out_path.write_text(json.dumps(scores_by_qid))
    # Clean up partial
    if partial_path.exists():
        partial_path.unlink()
    total_min = (time.time() - t_start) / 60
    print(f"\nDone in {total_min:.1f} min", flush=True)
    print(f"Wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
