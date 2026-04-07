"""
V5 pipeline: All sources + cross-encoder reranker.
1. Gather candidates from dense + BM25 + GPT
2. Look up actual text for each candidate
3. Rerank with bge-reranker-v2-m3
4. Score knee detection for dynamic cutoff
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
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE = Path(__file__).parent

def tokenize(text):
    return [t for t in re.findall(r'[a-zäöüß]+', text.lower()) if len(t) > 1]

def find_cutoff(scores, min_k=5, max_k=45):
    """Find cutoff using score knee detection."""
    if len(scores) <= min_k:
        return len(scores)

    # Find largest gap
    best_gap = 0
    best_idx = min_k
    for i in range(min_k, min(len(scores), max_k)):
        gap = scores[i-1] - scores[i]
        if gap > best_gap:
            best_gap = gap
            best_idx = i

    # Also check absolute threshold — reranker scores below 0.001 are noise
    for i in range(min_k, min(len(scores), max_k)):
        if scores[i] < 0.0005:
            return min(i, best_idx)

    return best_idx

def main():
    t0 = time.time()
    csv.field_size_limit(10000000)

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
        base_cit = re.sub(r'\s+E\.\s+.*$', '', cit).strip()
        if base_cit not in case_prefix_map:
            case_prefix_map[base_cit] = []
        case_prefix_map[base_cit].append(cit)
    del all_court_cites
    gc.collect()

    # Dense retrieval
    print("Loading FAISS law index...", flush=True)
    faiss_index = faiss.read_index(str(BASE / "index" / "faiss_laws.index"))
    with open(BASE / "index" / "faiss_laws_citations.pkl", "rb") as f:
        faiss_law_cites = pickle.load(f)
    embed_model = SentenceTransformer("intfloat/multilingual-e5-large")

    # Citation text lookup
    print("Loading citation texts for reranker...", flush=True)
    text_map = {}
    with open(BASE / "data" / "laws_de.csv", "r") as f:
        for row in csv.DictReader(f):
            text_map[row["citation"]] = f"{row['citation']} {row.get('title', '')} {row['text']}"[:512]
    with open(BASE / "data" / "court_considerations.csv", "r") as f:
        for row in csv.DictReader(f):
            text_map[row["citation"]] = f"{row['citation']} {row['text']}"[:512]
    print(f"Loaded {len(text_map)} citation texts", flush=True)

    # Reranker
    print("Loading reranker...", flush=True)
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

    # Val data
    with open(BASE / "data" / "val.csv", "r") as f:
        val_rows = list(csv.DictReader(f))
    gold_map = {row["query_id"]: set(row["gold_citations"].split(";")) for row in val_rows}

    print(f"\nProcessing {len(val_rows)} val queries...\n", flush=True)
    predictions = {}

    for row in val_rows:
        qid = row["query_id"]
        query = row["query"]
        exp = expansions.get(qid, {})
        cases = case_citations.get(qid, {})
        t1 = time.time()

        candidates = set()  # Collect all candidate citation strings

        # === Source 1: Explicit refs from query ===
        explicit = set(re.findall(
            r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+(?:\s+lit\.\s+[a-z])?)?\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü]+',
            query
        ))
        candidates.update(explicit)

        # === Source 2: GPT specific articles ===
        for art in exp.get("specific_articles", []):
            candidates.add(art)

        # === Source 3: Dense retrieval (top-200 laws) ===
        q_emb = embed_model.encode([f"query: {query}"], normalize_embeddings=True)
        d_scores, d_indices = faiss_index.search(q_emb.astype(np.float32), 200)
        for idx in d_indices[0]:
            candidates.add(faiss_law_cites[idx])

        # === Source 4: BM25 law search ===
        for bq in exp.get("bm25_queries_laws", []):
            tokens = tokenize(bq)
            if not tokens:
                continue
            scores_arr = law_bm25.get_scores(tokens)
            top_idx = scores_arr.argsort()[-80:][::-1]
            for idx in top_idx:
                if scores_arr[idx] > 0:
                    candidates.add(law_cites_bm25[idx])

        if exp.get("german_terms"):
            tokens = tokenize(" ".join(exp["german_terms"]))
            if tokens:
                scores_arr = law_bm25.get_scores(tokens)
                top_idx = scores_arr.argsort()[-80:][::-1]
                for idx in top_idx:
                    if scores_arr[idx] > 0:
                        candidates.add(law_cites_bm25[idx])

        # === Source 5: GPT case citations ===
        for cit in cases.get("expanded", []):
            if cit in court_set:
                candidates.add(cit)

        # === Source 6: Co-citation expansion ===
        gpt_court = cases.get("expanded", [])
        seen_cases = set()
        for cit in gpt_court:
            base_cit = re.sub(r'\s+E\.\s+.*$', '', cit).strip()
            if base_cit in seen_cases:
                continue
            seen_cases.add(base_cit)
            siblings = case_prefix_map.get(base_cit, [])
            for sib in siblings[:15]:
                candidates.add(sib)

        # Filter to corpus-verified
        verified = [c for c in candidates if c in law_set or c in court_set or c in explicit]
        print(f"  {qid}: {len(verified)} candidates to rerank...", end="", flush=True)

        # === Rerank with cross-encoder ===
        # Look up text for each candidate
        pairs_to_score = []
        valid_cites = []
        for cit in verified:
            text = text_map.get(cit, cit)  # Fallback to citation string itself
            pairs_to_score.append((query, text))
            valid_cites.append(cit)

        if pairs_to_score:
            reranker_scores = reranker.predict(pairs_to_score)
            # Sort by score
            scored = sorted(zip(valid_cites, reranker_scores), key=lambda x: x[1], reverse=True)
            score_values = [s for _, s in scored]

            # Dynamic cutoff
            cutoff = find_cutoff(score_values)
            selected = set(cit for cit, _ in scored[:cutoff])
        else:
            selected = set()

        predictions[qid] = selected

        # Per-query stats
        gold = gold_map[qid]
        tp = len(selected & gold)
        prec = tp / len(selected) if selected else 0
        rec = tp / len(gold) if gold else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        # Show top reranker scores
        if pairs_to_score:
            top3 = scored[:3]
            top3_str = ", ".join(f"{c}={s:.4f}" for c, s in top3)
        else:
            top3_str = "none"
        print(f" cutoff={cutoff}, pred={len(selected)}, TP={tp}, P={prec:.2f}, R={rec:.2f}, F1={f1:.2f} ({time.time()-t1:.0f}s)")
        print(f"    top3: {top3_str}")

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
    print(f"\n=== V5 MACRO F1: {macro_f1:.4f} ({macro_f1*100:.2f}%) ===")
    print(f"Total time: {time.time()-t0:.0f}s")

    # Save
    out_path = BASE / "submissions" / "val_pred_v5.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions.keys()):
            cites_str = ";".join(predictions[qid])
            writer.writerow([qid, cites_str])
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
