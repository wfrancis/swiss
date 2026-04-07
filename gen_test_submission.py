"""
Generate test submission using V3 pipeline.
Same logic as run_val_eval_v3 but for test.csv queries.
"""
import csv
import gc
import json
import pickle
import re
import time
from pathlib import Path

BASE = Path(__file__).parent


def tokenize(text):
    return [t for t in re.findall(r'[a-zäöüß]+', text.lower()) if len(t) > 1]


def main():
    t0 = time.time()

    # Load assets
    print("Loading assets...", flush=True)
    expansions = json.loads((BASE / "precompute" / "test_query_expansions.json").read_text())
    case_citations = json.loads((BASE / "precompute" / "test_case_citations.json").read_text())

    with open(BASE / "index" / "bm25_laws.pkl", "rb") as f:
        law_data = pickle.load(f)
    law_bm25 = law_data["bm25"]
    law_cites = law_data["citations"]
    law_set = set(law_cites)

    with open(BASE / "index" / "court_citations.pkl", "rb") as f:
        all_court_cites = pickle.load(f)
    court_set = set(all_court_cites)

    case_prefix_map = {}
    for cit in all_court_cites:
        base = re.sub(r'\s+E\.\s+.*$', '', cit).strip()
        if base not in case_prefix_map:
            case_prefix_map[base] = []
        case_prefix_map[base].append(cit)

    del all_court_cites
    gc.collect()

    # Load test queries
    with open(BASE / "data" / "test.csv", "r") as f:
        test_rows = list(csv.DictReader(f))

    print(f"\nProcessing {len(test_rows)} test queries...", flush=True)
    predictions = {}

    for row in test_rows:
        qid = row["query_id"]
        query = row["query"]
        exp = expansions.get(qid, {})
        cases = case_citations.get(qid, {})

        scored = {}

        # GPT specific articles
        for art in exp.get("specific_articles", []):
            scored[art] = 0.90

        # Explicit refs from query text
        explicit = set(re.findall(
            r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+(?:\s+lit\.\s+[a-z])?)?\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü]+',
            query
        ))
        for art in explicit:
            scored[art] = max(scored.get(art, 0), 0.95)

        # BM25 law search
        bm25_hits = {}
        for bq in exp.get("bm25_queries_laws", []):
            tokens = tokenize(bq)
            if not tokens:
                continue
            scores_arr = law_bm25.get_scores(tokens)
            top_idx = scores_arr.argsort()[-40:][::-1]
            for idx in top_idx:
                s = scores_arr[idx]
                if s > 0:
                    cit = law_cites[idx]
                    bm25_hits[cit] = max(bm25_hits.get(cit, 0), s)

        if exp.get("german_terms"):
            tokens = tokenize(" ".join(exp["german_terms"]))
            if tokens:
                scores_arr = law_bm25.get_scores(tokens)
                top_idx = scores_arr.argsort()[-40:][::-1]
                for idx in top_idx:
                    s = scores_arr[idx]
                    if s > 0:
                        cit = law_cites[idx]
                        bm25_hits[cit] = max(bm25_hits.get(cit, 0), s)

        if bm25_hits:
            max_bm25 = max(bm25_hits.values())
            for cit, s in bm25_hits.items():
                norm = (s / max_bm25) * 0.70
                scored[cit] = max(scored.get(cit, 0), norm)

        key_statutes = exp.get("key_statutes", [])
        for cit in bm25_hits:
            for stat in key_statutes:
                if stat in cit:
                    scored[cit] = min(scored.get(cit, 0) * 1.3, 0.92)

        # GPT case citations
        gpt_court = cases.get("expanded", [])
        for cit in gpt_court:
            if cit in court_set:
                scored[cit] = max(scored.get(cit, 0), 0.85)
            else:
                scored[cit] = max(scored.get(cit, 0), 0.45)

        # Co-citation expansion
        seen_cases = set()
        for cit in gpt_court:
            base = re.sub(r'\s+E\.\s+.*$', '', cit).strip()
            if base in seen_cases:
                continue
            seen_cases.add(base)
            siblings = case_prefix_map.get(base, [])
            parent_score = scored.get(cit, 0.5)
            for sib in siblings[:20]:
                if sib not in scored:
                    scored[sib] = parent_score * 0.35

        # Prune
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        estimated = exp.get("estimated_citation_count", 25)

        verified_ranked = []
        for cit, s in ranked:
            if cit in law_set or cit in court_set or cit in explicit:
                verified_ranked.append((cit, s))

        if verified_ranked:
            max_score = verified_ranked[0][1]
            target = max(estimated, 15)
            cutoff = target
            for i in range(target, len(verified_ranked)):
                if verified_ranked[i][1] < max_score * 0.12:
                    cutoff = i
                    break
                cutoff = i + 1
            cutoff = min(cutoff, 55)
        else:
            cutoff = 0

        selected = [cit for cit, _ in verified_ranked[:cutoff]]
        predictions[qid] = selected

        print(f"  {qid}: {len(selected)} citations", flush=True)

    # Write submission
    Path(BASE / "submissions").mkdir(exist_ok=True)
    out_path = BASE / "submissions" / "test_submission.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions.keys()):
            cites_str = ";".join(predictions[qid])
            writer.writerow([qid, cites_str])

    print(f"\nSubmission saved to {out_path}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
