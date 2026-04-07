"""
V4 pipeline: Dense + BM25 + GPT precompute combined.
Adds dense retrieval to V3's BM25 + GPT pipeline.
"""
import csv
import gc
import json
import pickle
import re
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent

def tokenize(text):
    return [t for t in re.findall(r'[a-zäöüß]+', text.lower()) if len(t) > 1]

def main():
    t0 = time.time()

    # Load assets
    print("Loading assets...", flush=True)
    expansions = json.loads((BASE / "precompute" / "val_query_expansions.json").read_text())
    case_citations = json.loads((BASE / "precompute" / "val_case_citations.json").read_text())

    # BM25
    with open(BASE / "index" / "bm25_laws.pkl", "rb") as f:
        law_data = pickle.load(f)
    law_bm25 = law_data["bm25"]
    law_cites_bm25 = law_data["citations"]
    law_set = set(law_cites_bm25)

    # Court citations
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

    # Dense retrieval
    print("Loading FAISS law index...", flush=True)
    faiss_index = faiss.read_index(str(BASE / "index" / "faiss_laws.index"))
    with open(BASE / "index" / "faiss_laws_citations.pkl", "rb") as f:
        faiss_law_cites = pickle.load(f)
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    print(f"Loaded {faiss_index.ntotal} FAISS vectors", flush=True)

    # Load val
    with open(BASE / "data" / "val.csv", "r") as f:
        val_rows = list(csv.DictReader(f))

    # Load gold
    gold_map = {}
    for row in val_rows:
        gold_map[row["query_id"]] = set(row["gold_citations"].split(";"))

    print(f"\nProcessing {len(val_rows)} val queries...", flush=True)
    predictions = {}

    for row in val_rows:
        qid = row["query_id"]
        query = row["query"]
        exp = expansions.get(qid, {})
        cases = case_citations.get(qid, {})

        scored = {}

        # === Source 1: GPT specific articles (high confidence) ===
        for art in exp.get("specific_articles", []):
            scored[art] = 0.90

        # === Source 2: Explicit refs from query text ===
        explicit = set(re.findall(
            r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+(?:\s+lit\.\s+[a-z])?)?\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü]+',
            query
        ))
        for art in explicit:
            scored[art] = max(scored.get(art, 0), 0.95)

        # === Source 3: Dense retrieval (top-200 laws) ===
        q_emb = model.encode([f"query: {query}"], normalize_embeddings=True)
        d_scores, d_indices = faiss_index.search(q_emb.astype(np.float32), 200)
        for rank, (score, idx) in enumerate(zip(d_scores[0], d_indices[0])):
            cit = faiss_law_cites[idx]
            # Normalize dense score: top score ~0.80, scale to 0.60
            norm_score = float(score) * 0.75
            # Boost top-ranked results
            if rank < 20:
                norm_score *= 1.2
            scored[cit] = max(scored.get(cit, 0), norm_score)

        # === Source 4: BM25 law search ===
        bm25_hits = {}
        for bq in exp.get("bm25_queries_laws", []):
            tokens = tokenize(bq)
            if not tokens:
                continue
            scores_arr = law_bm25.get_scores(tokens)
            top_idx = scores_arr.argsort()[-60:][::-1]  # Increased from 40 to 60
            for idx in top_idx:
                s = scores_arr[idx]
                if s > 0:
                    cit = law_cites_bm25[idx]
                    bm25_hits[cit] = max(bm25_hits.get(cit, 0), s)

        if exp.get("german_terms"):
            tokens = tokenize(" ".join(exp["german_terms"]))
            if tokens:
                scores_arr = law_bm25.get_scores(tokens)
                top_idx = scores_arr.argsort()[-60:][::-1]
                for idx in top_idx:
                    s = scores_arr[idx]
                    if s > 0:
                        cit = law_cites_bm25[idx]
                        bm25_hits[cit] = max(bm25_hits.get(cit, 0), s)

        if bm25_hits:
            max_bm25 = max(bm25_hits.values())
            for cit, s in bm25_hits.items():
                norm = (s / max_bm25) * 0.70
                scored[cit] = max(scored.get(cit, 0), norm)

        # Boost if matches key statute
        key_statutes = exp.get("key_statutes", [])
        for cit in list(scored.keys()):
            for stat in key_statutes:
                if stat in cit:
                    scored[cit] = min(scored.get(cit, 0) * 1.3, 0.92)

        # === Source 5: Agreement bonus ===
        # Citations found by BOTH dense and BM25 get a boost
        for cit in scored:
            in_dense = cit in set(faiss_law_cites[idx] for idx in d_indices[0][:200])
            in_bm25 = cit in bm25_hits
            if in_dense and in_bm25:
                scored[cit] = min(scored[cit] * 1.25, 0.95)

        # === Source 6: GPT case citations ===
        gpt_court = cases.get("expanded", [])
        for cit in gpt_court:
            if cit in court_set:
                scored[cit] = max(scored.get(cit, 0), 0.85)
            else:
                scored[cit] = max(scored.get(cit, 0), 0.45)

        # === Source 7: Co-citation expansion ===
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

        # === Prune and select ===
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        estimated = exp.get("estimated_citation_count", 25)

        verified_ranked = []
        for cit, s in ranked:
            if cit in law_set or cit in court_set or cit in explicit:
                verified_ranked.append((cit, s))

        if verified_ranked:
            # Score-gap cutoff: find largest drop in scores
            scores_list = [s for _, s in verified_ranked]

            # Method: take candidates above a meaningful score threshold
            # High-confidence: score > 0.60
            # Medium-confidence: score > 0.35
            high_conf = sum(1 for s in scores_list if s >= 0.60)
            med_conf = sum(1 for s in scores_list if s >= 0.35)

            # Use estimated count as guide, but clamp
            target = min(max(estimated, 10), 45)

            # Find score elbow: largest gap in top candidates
            best_gap_idx = target
            if len(scores_list) > 5:
                gaps = []
                for i in range(1, min(len(scores_list), 50)):
                    gaps.append((scores_list[i-1] - scores_list[i], i))
                gaps.sort(reverse=True)
                # Use the biggest gap that's within reasonable range
                for gap_size, gap_idx in gaps[:3]:
                    if 5 <= gap_idx <= 45 and gap_size > 0.02:
                        best_gap_idx = gap_idx
                        break

            # Final cutoff: blend estimate with score gap
            cutoff = min(best_gap_idx, target + 10)
            cutoff = max(cutoff, high_conf)  # At least include high-conf
            cutoff = min(cutoff, 45)
        else:
            cutoff = 0

        selected = set(cit for cit, _ in verified_ranked[:cutoff])
        predictions[qid] = selected

        # Per-query stats
        gold = gold_map[qid]
        tp = len(selected & gold)
        prec = tp / len(selected) if selected else 0
        rec = tp / len(gold) if gold else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {qid}: pred={len(selected)}, gold={len(gold)}, TP={tp}, P={prec:.2f}, R={rec:.2f}, F1={f1:.2f}")

    # Macro F1
    f1_scores = []
    for qid in predictions:
        gold = gold_map[qid]
        pred = predictions[qid]
        tp = len(pred & gold)
        prec = tp / len(pred) if pred else 0
        rec = tp / len(gold) if gold else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"\n=== V4 MACRO F1: {macro_f1:.4f} ({macro_f1*100:.2f}%) ===")
    print(f"Total time: {time.time()-t0:.0f}s")

    # Save predictions
    out_path = BASE / "submissions" / "val_pred_v4.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions.keys()):
            cites_str = ";".join(predictions[qid])
            writer.writerow([qid, cites_str])
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
