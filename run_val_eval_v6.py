"""
V6 pipeline: All sources combined with corpus verification.
Dense + BM25 + GPT old + GPT full citations.
Key: only predict citations that exist in the corpus.
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

    print("Loading assets...", flush=True)
    expansions = json.loads((BASE / "precompute" / "val_query_expansions.json").read_text())
    case_citations = json.loads((BASE / "precompute" / "val_case_citations.json").read_text())
    full_citations = json.loads((BASE / "precompute" / "val_full_citations.json").read_text())

    # BM25
    with open(BASE / "index" / "bm25_laws.pkl", "rb") as f:
        law_data = pickle.load(f)
    law_bm25 = law_data["bm25"]
    law_cites_bm25 = law_data["citations"]
    law_set = set(law_cites_bm25)

    # Court
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

    # Dense
    print("Loading FAISS...", flush=True)
    faiss_index = faiss.read_index(str(BASE / "index" / "faiss_laws.index"))
    with open(BASE / "index" / "faiss_laws_citations.pkl", "rb") as f:
        faiss_law_cites = pickle.load(f)
    embed_model = SentenceTransformer("intfloat/multilingual-e5-large")

    # Val
    with open(BASE / "data" / "val.csv", "r") as f:
        val_rows = list(csv.DictReader(f))
    gold_map = {row["query_id"]: set(row["gold_citations"].split(";")) for row in val_rows}

    print(f"\nProcessing {len(val_rows)} queries...\n", flush=True)
    predictions = {}

    for row in val_rows:
        qid = row["query_id"]
        query = row["query"]
        exp = expansions.get(qid, {})
        cases = case_citations.get(qid, {})
        full = full_citations.get(qid, {})

        scored = {}

        # === HIGH confidence: GPT full citation predictions ===
        # These are direct GPT predictions — high signal
        for art in full.get("law_citations", []):
            if art in law_set:
                scored[art] = 0.88
            else:
                scored[art] = 0.40  # May need fuzzy matching later

        for cit in full.get("court_citations", []):
            if cit in court_set:
                scored[cit] = 0.85
            else:
                # Try co-citation matching
                base_cit = re.sub(r'\s+E\.\s+.*$', '', cit).strip()
                siblings = case_prefix_map.get(base_cit, [])
                for sib in siblings[:10]:
                    scored[sib] = max(scored.get(sib, 0), 0.70)

        # === GPT old specific articles ===
        for art in exp.get("specific_articles", []):
            scored[art] = max(scored.get(art, 0), 0.90)

        # === Explicit refs from query ===
        explicit = set(re.findall(
            r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+(?:\s+lit\.\s+[a-z])?)?\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü]+',
            query
        ))
        for art in explicit:
            scored[art] = max(scored.get(art, 0), 0.95)

        # === Dense retrieval top-200 ===
        q_emb = embed_model.encode([f"query: {query}"], normalize_embeddings=True)
        d_scores, d_indices = faiss_index.search(q_emb.astype(np.float32), 200)
        dense_cites = set()
        for rank, (score, idx) in enumerate(zip(d_scores[0], d_indices[0])):
            cit = faiss_law_cites[idx]
            dense_cites.add(cit)
            norm = float(score) * 0.65
            if rank < 10:
                norm *= 1.3
            scored[cit] = max(scored.get(cit, 0), norm)

        # === BM25 ===
        bm25_hits = {}
        for bq in exp.get("bm25_queries_laws", []):
            tokens = tokenize(bq)
            if not tokens:
                continue
            scores_arr = law_bm25.get_scores(tokens)
            for idx in scores_arr.argsort()[-80:][::-1]:
                s = scores_arr[idx]
                if s > 0:
                    cit = law_cites_bm25[idx]
                    bm25_hits[cit] = max(bm25_hits.get(cit, 0), s)

        if exp.get("german_terms"):
            tokens = tokenize(" ".join(exp["german_terms"]))
            if tokens:
                scores_arr = law_bm25.get_scores(tokens)
                for idx in scores_arr.argsort()[-80:][::-1]:
                    s = scores_arr[idx]
                    if s > 0:
                        cit = law_cites_bm25[idx]
                        bm25_hits[cit] = max(bm25_hits.get(cit, 0), s)

        if bm25_hits:
            max_bm25 = max(bm25_hits.values())
            for cit, s in bm25_hits.items():
                norm = (s / max_bm25) * 0.65
                scored[cit] = max(scored.get(cit, 0), norm)

        # === GPT old court citations ===
        for cit in cases.get("expanded", []):
            if cit in court_set:
                scored[cit] = max(scored.get(cit, 0), 0.85)

        # === Co-citation expansion ===
        gpt_court = list(set(cases.get("expanded", []) + full.get("court_citations", [])))
        seen = set()
        for cit in gpt_court:
            base_cit = re.sub(r'\s+E\.\s+.*$', '', cit).strip()
            if base_cit in seen:
                continue
            seen.add(base_cit)
            siblings = case_prefix_map.get(base_cit, [])
            parent_score = scored.get(cit, 0.5)
            for sib in siblings[:15]:
                if sib not in scored:
                    scored[sib] = parent_score * 0.30

        # === Agreement boost ===
        for cit in scored:
            sources = 0
            if cit in dense_cites: sources += 1
            if cit in bm25_hits: sources += 1
            if cit in set(full.get("law_citations", [])): sources += 1
            if cit in set(exp.get("specific_articles", [])): sources += 1
            if sources >= 2:
                scored[cit] = min(scored[cit] * 1.2, 0.95)
            if sources >= 3:
                scored[cit] = min(scored[cit] * 1.3, 0.98)

        # === Always include Art. 100 Abs. 1 BGG (appears in 9/10 queries) ===
        if "Art. 100 Abs. 1 BGG" in law_set:
            scored["Art. 100 Abs. 1 BGG"] = max(scored.get("Art. 100 Abs. 1 BGG", 0), 0.80)

        # === Filter to corpus-verified and select ===
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        verified = [(c, s) for c, s in ranked if c in law_set or c in court_set or c in explicit]

        # Dynamic cutoff: keep only high-confidence predictions
        # Count how many from high-signal sources (GPT full + GPT old + explicit)
        high_signal = set()
        high_signal.update(c for c in full.get("law_citations", []) if c in law_set)
        high_signal.update(c for c in full.get("court_citations", []) if c in court_set)
        high_signal.update(c for c in cases.get("expanded", []) if c in court_set)
        high_signal.update(exp.get("specific_articles", []))
        high_signal.update(explicit)

        # Start with high-signal count, then add from retrieval
        base_count = len([c for c, s in verified if c in high_signal])
        retrieval_bonus = max(5, int(base_count * 0.3))  # Add 30% from retrieval
        target = base_count + retrieval_bonus
        target = max(target, 10)
        target = min(target, 40)

        # Also cut off when scores drop sharply
        cutoff = min(target, len(verified))
        if len(verified) > cutoff:
            for i in range(cutoff, len(verified)):
                if verified[i][1] >= 0.50:
                    cutoff = i + 1
                else:
                    break
        cutoff = min(cutoff, 40)

        selected = set(c for c, _ in verified[:cutoff])
        predictions[qid] = selected

        # Stats
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
    print(f"\n=== V6 MACRO F1: {macro_f1:.4f} ({macro_f1*100:.2f}%) ===")
    print(f"Total time: {time.time()-t0:.0f}s")

    out_path = BASE / "submissions" / "val_pred_v6.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions.keys()):
            writer.writerow([qid, ";".join(predictions[qid])])
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
