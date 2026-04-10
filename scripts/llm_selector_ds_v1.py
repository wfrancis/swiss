#!/usr/bin/env python3
"""
LLM joint selector v1: deepseek-reasoner over court-dense judged bundles.

Hypothesis: a strong reasoner doing JOINT selection (pick 20 from 80) beats the
existing per-candidate judge -> scalar classifier path. Same model on both sides
isolates the structural change (joint vs per-candidate) from model quality.

For each query:
  1. Load all judged candidates from val_v11_courtdense_ds_val1 / test1
  2. Pre-filter to top-K (all must_include + top-N of remaining by signal)
  3. Format compact prompt with cite/kind/judge_label/judge_reason/raw_score
  4. Call deepseek-reasoner once, ask for 20-30 cites as JSON
  5. Filter hallucinated cites, save predictions

Auth: V11_API_KEY env var (DeepSeek key). Base URL defaults to api.deepseek.com/v1.
Cost: ~$0.01 per query at top-K=80 (deepseek-reasoner ~$0.55/M in, ~$2.19/M out).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent

ARTIFACT_PATHS = {
    "val":  REPO / "artifacts/v11/val_v11_courtdense_ds_val1/judged_bundles.json",
    "test": REPO / "artifacts/v11/test_v11_courtdense_ds_test1/judged_bundles.json",
}

GOLD_PATH = REPO / "data/val.csv"
CITE_TEXT_INDEX = REPO / "artifacts/citation_text_index.pkl"

_CITE_TEXT: dict[str, str] | None = None


def load_cite_text_index() -> dict[str, str]:
    global _CITE_TEXT
    if _CITE_TEXT is None:
        with CITE_TEXT_INDEX.open("rb") as f:
            _CITE_TEXT = pickle.load(f)
    return _CITE_TEXT


def load_bundles(split: str) -> list[dict]:
    path = ARTIFACT_PATHS[split]
    data = json.loads(path.read_text())
    return data["bundles"]


def load_gold() -> dict[str, set[str]]:
    with GOLD_PATH.open() as f:
        return {
            row["query_id"]: {c for c in row["gold_citations"].split(";") if c}
            for row in csv.DictReader(f)
        }


def candidate_signal(c: dict) -> float:
    """Composite ranking signal for selecting which non-must_include candidates
    to surface to the LLM. Take the max of the strongest available signals."""
    raw = c.get("raw_score") or 0.0
    final = c.get("final_score") or 0.0
    cd = c.get("court_dense_rank")
    cd_score = (1.0 / (1 + cd)) if cd is not None else 0.0
    dense = c.get("dense_rank")
    dense_score = (1.0 / (1 + dense)) if dense is not None else 0.0
    return max(raw, final, cd_score, dense_score)


def filter_top_k(candidates: list[dict], top_k: int) -> list[dict]:
    """Always keep all must_include; add top-N of remaining by signal."""
    must = [c for c in candidates if c.get("judge_label") == "must_include"]
    rest = [c for c in candidates if c.get("judge_label") != "must_include"]
    rest.sort(key=lambda c: -candidate_signal(c))
    n_more = max(0, top_k - len(must))
    pool = must + rest[:n_more]
    pool.sort(key=lambda c: -candidate_signal(c))
    return pool


def truncate(s: str, n: int) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def format_candidate_line(idx: int, c: dict, prompt_version: str,
                          snippet_chars: int = 0,
                          cite_text_index: dict[str, str] | None = None) -> str:
    cite = c["citation"]
    kind = c.get("kind", "?")
    label = c.get("judge_label") or "?"
    reason = truncate(c.get("judge_reason") or "", 140)
    raw = c.get("raw_score") or 0.0
    if prompt_version == "v1":
        return f'[{idx}] cite="{cite}" kind={kind} judge={label} raw={raw:.2f} reason="{reason}"'
    flags = []
    if c.get("is_explicit"):
        flags.append("EXPLICIT")
    if c.get("is_query_case"):
        flags.append("QUERY_CASE")
    flag_str = f" flags={','.join(flags)}" if flags else ""
    base = f'[{idx}] cite="{cite}" kind={kind} judge={label} raw={raw:.2f}{flag_str} reason="{reason}"'
    if snippet_chars > 0 and cite_text_index is not None:
        # Prefer corpus text (full snippet); fall back to bundle's snippet field
        text = cite_text_index.get(cite) or (c.get("snippet") or "")
        if text:
            text = truncate(text, snippet_chars)
            base += f'\n      text="{text}"'
    return base


SYSTEM_PROMPT_V1 = (
    "You are an expert Swiss legal researcher selecting citations for the "
    "Bundesgericht (Swiss Federal Supreme Court). Given a legal question and "
    "candidate citations (each with a per-candidate judge's analysis), choose "
    "the subset that a BGer decision for this question would actually cite. "
    "Prefer precision: a small set of correct citations beats a large set with "
    "errors. Override per-candidate judge labels when joint reasoning across "
    "the full candidate set suggests a different selection. Always return "
    "valid JSON."
)

SYSTEM_PROMPT_V2 = (
    "You are an expert Swiss legal researcher selecting citations for the "
    "Bundesgericht (Swiss Federal Supreme Court). You are given a legal "
    "question and a set of candidate citations, each pre-judged by a "
    "per-candidate scorer with one of three labels:\n"
    "  - must_include: HIGH-CONFIDENCE pick by the per-candidate judge. "
    "DEFAULT = keep all must_include unless joint reasoning gives a strong "
    "reason to drop one.\n"
    "  - plausible: promising but uncertain. Add to your selection when joint "
    "reasoning supports it.\n"
    "  - reject: judge thought not relevant. You may override only if you have "
    "a strong reason from joint reasoning.\n"
    "Citations marked EXPLICIT are directly named in the query — almost always "
    "keep them.\n"
    "BGer decisions typically cite 25-35 citations per case. Aim for that "
    "range. Both over-predicting and under-predicting hurt the macro F1 metric "
    "this is judged on. Always return valid JSON."
)


def build_user_prompt(query: str, candidates: list[dict], prompt_version: str,
                      snippet_chars: int = 0,
                      cite_text_index: dict[str, str] | None = None) -> str:
    cand_lines = "\n".join(
        format_candidate_line(i + 1, c, prompt_version, snippet_chars, cite_text_index)
        for i, c in enumerate(candidates)
    )
    if prompt_version == "v1":
        return f"""Query (English):
{query}

Candidates (sorted by retrieval signal, strongest first):
{cand_lines}

Task: Select the citations a BGer decision for this query would cite.
- Use citations EXACTLY as written above (do not modify the format).
- Target 20-30 citations.
- You MAY override per-candidate judge labels when joint reasoning suggests a different choice.
- You MUST NOT invent citations not in the list above.
- Prefer precision over recall.

Output JSON: {{"selected": ["citation_1", "citation_2", ...]}}"""
    return f"""Query (English):
{query}

Candidates (sorted by retrieval signal, strongest first):
{cand_lines}

Task: Select the citations a BGer decision for THIS query would cite.

Selection guidance:
- DEFAULT: include ALL must_include cites. Drop one only if joint reasoning across the full candidate set gives a strong reason.
- ADD plausible cites that fit the legal theory of the query.
- Override reject only when the candidate is a clear miss by the per-candidate judge.
- Cites marked EXPLICIT are directly mentioned in the query — almost always keep.
- TARGET: 25-35 citations (BGer decisions cite ~25 on average; under- and over-predicting both hurt macro F1).
- Use citations EXACTLY as written above (do not modify the format).
- You MUST NOT invent citations not in the list above.

Output JSON: {{"selected": ["citation_1", "citation_2", ...]}}"""


def call_deepseek(system: str, user: str, model: str = "deepseek-reasoner",
                  max_tokens: int = 8000) -> dict:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["V11_API_KEY"],
        base_url=os.environ.get("V11_BASE_URL", "https://api.deepseek.com/v1"),
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    return json.loads(content)


def select_for_query(bundle: dict, top_k: int, prompt_version: str,
                     snippet_chars: int = 0,
                     cite_text_index: dict[str, str] | None = None,
                     max_tokens: int = 8000) -> tuple[str, set[str], dict]:
    qid = bundle["query_id"]
    query = bundle["query"]
    pool = filter_top_k(bundle["candidates"], top_k)
    valid_cites = {c["citation"] for c in pool}
    n_must = sum(1 for c in pool if c.get("judge_label") == "must_include")

    system = SYSTEM_PROMPT_V2 if prompt_version == "v2" else SYSTEM_PROMPT_V1
    user = build_user_prompt(query, pool, prompt_version, snippet_chars, cite_text_index)
    try:
        result = call_deepseek(system, user, max_tokens=max_tokens)
    except Exception as e:
        return qid, set(), {"error": f"{type(e).__name__}: {e}", "n_pool": len(pool), "n_must": n_must}

    raw_selected = result.get("selected") or []
    if not isinstance(raw_selected, list):
        return qid, set(), {"error": f"selected not a list: {type(raw_selected).__name__}",
                            "n_pool": len(pool), "n_must": n_must}

    selected = {c for c in raw_selected if isinstance(c, str) and c in valid_cites}
    n_dropped = len(raw_selected) - len(selected)
    info = {
        "n_pool": len(pool),
        "n_must": n_must,
        "n_raw_selected": len(raw_selected),
        "n_selected": len(selected),
        "n_dropped": n_dropped,
    }
    return qid, selected, info


def per_query_f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(gold)
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--top-k", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap queries (for smoke tests)")
    ap.add_argument("--out-csv", type=str, default=None)
    ap.add_argument("--prompt-version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--snippet-chars", type=int, default=600,
                    help="max chars of corpus snippet text per candidate (0 = no snippets)")
    ap.add_argument("--max-tokens", type=int, default=12000,
                    help="DeepSeek max_tokens (response cap; CoT eats output budget)")
    args = ap.parse_args()

    if "V11_API_KEY" not in os.environ:
        print("ERROR: V11_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    bundles = load_bundles(args.split)
    if args.limit:
        bundles = bundles[: args.limit]

    print(f"Split: {args.split}", flush=True)
    print(f"Queries: {len(bundles)}", flush=True)
    print(f"Top-K candidates per query: {args.top_k}", flush=True)
    print(f"Workers: {args.workers}", flush=True)
    print(f"Prompt version: {args.prompt_version}", flush=True)
    print(f"Snippet chars: {args.snippet_chars}", flush=True)
    print(f"Max tokens: {args.max_tokens}", flush=True)
    print(f"Model: deepseek-reasoner", flush=True)
    print()

    cite_text_index = None
    if args.snippet_chars > 0:
        t_load = time.time()
        print(f"Loading citation->text index from {CITE_TEXT_INDEX} ...", flush=True)
        cite_text_index = load_cite_text_index()
        print(f"  loaded {len(cite_text_index):,} entries  ({time.time()-t_load:.1f}s)", flush=True)

    gold = load_gold() if args.split == "val" else None

    t0 = time.time()
    predictions: dict[str, set[str]] = {}
    infos: dict[str, dict] = {}
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                select_for_query,
                b,
                args.top_k,
                args.prompt_version,
                args.snippet_chars,
                cite_text_index,
                args.max_tokens,
            ): b["query_id"]
            for b in bundles
        }
        for fut in as_completed(futures):
            qid, selected, info = fut.result()
            predictions[qid] = selected
            infos[qid] = info
            n_done += 1
            elapsed = time.time() - t0
            err = f" ERR={info['error']}" if "error" in info else ""
            print(
                f"  [{n_done}/{len(bundles)}] {qid}: "
                f"pool={info['n_pool']} must={info['n_must']} "
                f"raw={info.get('n_raw_selected', 0)} "
                f"kept={len(selected)} drop={info.get('n_dropped', 0)} "
                f"({elapsed:.0f}s){err}",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/len(bundles):.1f}s/query avg)", flush=True)

    # Score on val
    if gold:
        f1s = []
        print("\nPer-query F1:")
        for qid in sorted(predictions):
            pred = predictions[qid]
            g = gold.get(qid, set())
            f1 = per_query_f1(pred, g)
            f1s.append(f1)
            tp = len(pred & g)
            p = tp / len(pred) if pred else 0.0
            r = tp / len(g) if g else 0.0
            print(
                f"  {qid}: pred={len(pred):3d} gold={len(g):3d} tp={tp:3d} "
                f"p={p:.3f} r={r:.3f} F1={f1:.3f}"
            )
        macro = mean(f1s)
        print(f"\n=== val macro F1: {macro:.4f} ({macro*100:.2f}%) ===")
        print(f"Baseline (0.30257) reference: val F1 ~28.24%")

    # Save CSV
    out_path = (
        Path(args.out_csv)
        if args.out_csv
        else REPO / f"submissions/llm_selector_ds_v1_{args.split}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions):
            w.writerow([qid, ";".join(sorted(predictions[qid]))])
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
