"""
Efficient val evaluation: processes all queries through each shard once.
This avoids reloading 500MB pickle files per query.
"""
import csv
import gc
import json
import pickle
import re
import time
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent


def tokenize(text):
    return [t for t in re.findall(r'[a-zäöüß]+', text.lower()) if len(t) > 1]


def citation_f1(pred, gold):
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main():
    t0 = time.time()

    # Load query expansions
    print("Loading expansions...", flush=True)
    expansions = json.loads((BASE / "precompute" / "val_query_expansions.json").read_text())

    # Load val queries
    with open(BASE / "data" / "val.csv", "r") as f:
        val_rows = list(csv.DictReader(f))

    # Build query sets: for each query, collect all German search terms
    query_data = {}
    for row in val_rows:
        qid = row["query_id"]
        query = row["query"]
        gold = {c.strip() for c in row["gold_citations"].split(";")}
        exp = expansions.get(qid, {})

        # Collect all BM25 queries for this query
        law_queries = exp.get("bm25_queries_laws", [])
        court_queries = exp.get("bm25_queries_court", [])
        german_terms = exp.get("german_terms", [])
        specific_articles = exp.get("specific_articles", [])

        # Extract explicit refs from query text
        explicit_statutes = set(re.findall(
            r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+(?:\s+lit\.\s+[a-z])?)?\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü]+',
            query
        ))

        query_data[qid] = {
            "query": query,
            "gold": gold,
            "law_queries": law_queries,
            "court_queries": court_queries,
            "german_terms": german_terms,
            "specific_articles": specific_articles,
            "explicit_statutes": explicit_statutes,
            "estimated_citations": exp.get("estimated_citation_count", 25),
            "law_results": {},   # citation -> max_score
            "court_results": {},
        }

    # === PHASE 1: Law BM25 search ===
    print("\n=== Law BM25 search ===", flush=True)
    with open(BASE / "index" / "bm25_laws.pkl", "rb") as f:
        law_data = pickle.load(f)
    law_bm25 = law_data["bm25"]
    law_citations = law_data["citations"]
    law_set = set(law_citations)

    for qid, qd in query_data.items():
        results = {}
        # Search with each BM25 query
        for bq in qd["law_queries"]:
            tokens = tokenize(bq)
            if not tokens:
                continue
            scores = law_bm25.get_scores(tokens)
            top_idx = scores.argsort()[-30:][::-1]
            for idx in top_idx:
                if scores[idx] > 0:
                    cit = law_citations[idx]
                    results[cit] = max(results.get(cit, 0), float(scores[idx]))

        # Search with German terms
        if qd["german_terms"]:
            tokens = tokenize(" ".join(qd["german_terms"]))
            if tokens:
                scores = law_bm25.get_scores(tokens)
                top_idx = scores.argsort()[-30:][::-1]
                for idx in top_idx:
                    if scores[idx] > 0:
                        cit = law_citations[idx]
                        results[cit] = max(results.get(cit, 0), float(scores[idx]))

        qd["law_results"] = results
        print(f"  {qid}: {len(results)} law candidates", flush=True)

    del law_bm25, law_data
    gc.collect()
    print(f"Law search done ({time.time()-t0:.0f}s)", flush=True)

    # === PHASE 2: Court BM25 search (shard by shard) ===
    print("\n=== Court BM25 search (4 shards) ===", flush=True)
    shard_paths = sorted((BASE / "index").glob("bm25_court_shard*.pkl"))

    for si, shard_path in enumerate(shard_paths):
        t1 = time.time()
        print(f"\n  Loading shard {si}...", flush=True)
        with open(shard_path, "rb") as f:
            shard_data = pickle.load(f)
        shard_bm25 = shard_data["bm25"]
        shard_citations = shard_data["citations"]
        print(f"  Loaded ({len(shard_citations)} entries, {time.time()-t1:.0f}s)", flush=True)

        # Search all queries against this shard
        for qid, qd in query_data.items():
            results = qd["court_results"]

            for bq in qd["court_queries"]:
                tokens = tokenize(bq)
                if not tokens:
                    continue
                scores = shard_bm25.get_scores(tokens)
                top_idx = scores.argsort()[-20:][::-1]
                for idx in top_idx:
                    if scores[idx] > 0:
                        cit = shard_citations[idx]
                        results[cit] = max(results.get(cit, 0), float(scores[idx]))

            # Also search with German terms
            if qd["german_terms"]:
                tokens = tokenize(" ".join(qd["german_terms"]))
                if tokens:
                    scores = shard_bm25.get_scores(tokens)
                    top_idx = scores.argsort()[-20:][::-1]
                    for idx in top_idx:
                        if scores[idx] > 0:
                            cit = shard_citations[idx]
                            results[cit] = max(results.get(cit, 0), float(scores[idx]))

        del shard_bm25, shard_data
        gc.collect()
        print(f"  Shard {si} done ({time.time()-t1:.0f}s)", flush=True)

    print(f"\nCourt search done ({time.time()-t0:.0f}s)", flush=True)

    # === PHASE 3: Consolidate & Prune ===
    print("\n=== Consolidate & Evaluate ===", flush=True)

    # Load court citations for existence check
    with open(BASE / "index" / "court_citations.pkl", "rb") as f:
        all_court_citations = set(pickle.load(f))

    # Also load law citation set
    with open(BASE / "index" / "bm25_laws.pkl", "rb") as f:
        law_data = pickle.load(f)
    all_law_citations = set(law_data["citations"])
    del law_data
    gc.collect()

    f1_scores = []
    predictions = {}

    for row in val_rows:
        qid = row["query_id"]
        qd = query_data[qid]
        gold = qd["gold"]
        gold_laws = {c for c in gold if c.startswith("Art.")}
        gold_court = gold - gold_laws

        # Merge all sources
        scored = {}

        # Law results
        for cit, score in qd["law_results"].items():
            scored[cit] = max(scored.get(cit, 0), min(score / 15.0, 1.0))

        # Court results
        for cit, score in qd["court_results"].items():
            scored[cit] = max(scored.get(cit, 0), min(score / 15.0, 1.0))

        # Explicit/GPT-identified articles (high confidence)
        for art in qd["specific_articles"]:
            if art in all_law_citations:
                scored[art] = max(scored.get(art, 0), 0.9)
            else:
                scored[art] = max(scored.get(art, 0), 0.8)

        for art in qd["explicit_statutes"]:
            if art in all_law_citations:
                scored[art] = max(scored.get(art, 0), 0.95)

        # Co-citation expansion for top court results
        top_court = sorted(
            [(c, s) for c, s in scored.items()
             if c.startswith("BGE") or re.search(r'\d[A-Z]_\d+/\d{4}', c)],
            key=lambda x: x[1], reverse=True
        )[:15]
        for cit, sc in top_court:
            base_case = re.sub(r'\s+E\.\s+.*$', '', cit).strip()
            for full_cit in all_court_citations:
                if full_cit.startswith(base_case) and full_cit not in scored:
                    scored[full_cit] = sc * 0.4  # Siblings get reduced score

        # Boost items found in both law+court search
        for cit in qd["law_results"]:
            if cit in qd["court_results"]:
                scored[cit] = scored.get(cit, 0) * 1.5

        # Rank and prune
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        estimated = qd["estimated_citations"]

        # Dynamic cutoff
        if ranked:
            max_score = ranked[0][1]
            threshold = max_score * 0.10
            cutoff = max(estimated, 15)
            for i in range(cutoff, len(ranked)):
                if ranked[i][1] < threshold:
                    cutoff = i
                    break
                cutoff = i + 1
            cutoff = min(cutoff, 60)
        else:
            cutoff = 0

        selected = set(cit for cit, _ in ranked[:cutoff])
        predictions[qid] = selected

        # Evaluate
        f1 = citation_f1(selected, gold)
        tp = len(selected & gold)
        tp_law = len(selected & gold_laws)
        tp_court = len(selected & gold_court)

        print(f"\n{qid}: {len(gold)} gold ({len(gold_laws)} law, {len(gold_court)} court)")
        print(f"  Predicted: {len(selected)} | F1: {f1:.4f}")
        print(f"  Law: {tp_law}/{len(gold_laws)} | Court: {tp_court}/{len(gold_court)} | Total tp: {tp}")
        if tp > 0:
            hits = sorted(selected & gold)[:8]
            print(f"  Hits: {hits}")

        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"\n{'='*60}")
    print(f"MACRO F1: {macro_f1:.4f}")
    print(f"Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")

    # Write submission
    Path(BASE / "submissions").mkdir(exist_ok=True)
    with open(BASE / "submissions" / "val_pred.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions.keys()):
            cites_str = ";".join(sorted(predictions[qid]))
            writer.writerow([qid, cites_str])
    print(f"\nSubmission saved to submissions/val_pred.csv")


if __name__ == "__main__":
    main()
