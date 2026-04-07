"""Test combined recall with ALL sources including GPT full citations."""
import csv
import json
import pickle
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE = Path(__file__).parent

def tokenize(text):
    return [t for t in re.findall(r'[a-zäöüß]+', text.lower()) if len(t) > 1]

print("Loading assets...", flush=True)

# Dense
index = faiss.read_index(str(BASE / "index" / "faiss_laws.index"))
with open(BASE / "index" / "faiss_laws_citations.pkl", "rb") as f:
    law_citations = pickle.load(f)
model = SentenceTransformer("intfloat/multilingual-e5-large")

# BM25
with open(BASE / "index" / "bm25_laws.pkl", "rb") as f:
    law_data = pickle.load(f)
law_bm25 = law_data["bm25"]
law_cites_bm25 = law_data["citations"]

# GPT precompute (old)
expansions = json.loads((BASE / "precompute" / "val_query_expansions.json").read_text())
case_citations = json.loads((BASE / "precompute" / "val_case_citations.json").read_text())

# GPT full citations (new)
full_citations = json.loads((BASE / "precompute" / "val_full_citations.json").read_text())

# Court citations
with open(BASE / "index" / "court_citations.pkl", "rb") as f:
    all_court_cites = pickle.load(f)
court_set = set(all_court_cites)
law_set = set(law_citations)

with open(BASE / "data" / "val.csv", "r") as f:
    val_rows = list(csv.DictReader(f))

print("Loaded.\n", flush=True)

total_gold = 0
total_found = {
    "dense": 0, "bm25": 0, "gpt_old": 0, "gpt_full": 0,
    "gpt_court_old": 0, "gpt_court_full": 0, "combined": 0
}

for row in val_rows:
    qid = row["query_id"]
    query = row["query"]
    gold = set(row["gold_citations"].split(";"))
    exp = expansions.get(qid, {})
    cases = case_citations.get(qid, {})
    full = full_citations.get(qid, {})

    # Dense top-200
    q_emb = model.encode([f"query: {query}"], normalize_embeddings=True)
    _, d_indices = index.search(q_emb.astype(np.float32), 200)
    dense_hits = set(law_citations[idx] for idx in d_indices[0])

    # BM25
    bm25_hits = set()
    for bq in exp.get("bm25_queries_laws", []):
        tokens = tokenize(bq)
        if not tokens:
            continue
        scores_arr = law_bm25.get_scores(tokens)
        for idx in scores_arr.argsort()[-80:][::-1]:
            if scores_arr[idx] > 0:
                bm25_hits.add(law_cites_bm25[idx])
    if exp.get("german_terms"):
        tokens = tokenize(" ".join(exp["german_terms"]))
        if tokens:
            scores_arr = law_bm25.get_scores(tokens)
            for idx in scores_arr.argsort()[-80:][::-1]:
                if scores_arr[idx] > 0:
                    bm25_hits.add(law_cites_bm25[idx])

    # GPT old specific articles
    gpt_old_laws = set(exp.get("specific_articles", []))

    # GPT full citations (new)
    gpt_full_laws = set(full.get("law_citations", []))
    gpt_full_court = set(full.get("court_citations", []))

    # GPT old court
    gpt_old_court = set(c for c in cases.get("expanded", []) if c in court_set)

    # Combined (all sources)
    all_laws = dense_hits | bm25_hits | gpt_old_laws | gpt_full_laws
    all_court = gpt_old_court | gpt_full_court

    # Recall
    gold_laws = set(c for c in gold if c in law_set)
    gold_court = set(c for c in gold if c not in law_set)

    dense_found = len(gold_laws & dense_hits)
    bm25_found = len(gold_laws & bm25_hits)
    gpt_old_found = len(gold_laws & gpt_old_laws)
    gpt_full_found = len(gold & (gpt_full_laws | gpt_full_court))
    gpt_court_old_found = len(gold_court & gpt_old_court)
    gpt_court_full_found = len(gold_court & gpt_full_court)
    combined_found = len(gold & (all_laws | all_court))

    total_gold += len(gold)
    total_found["dense"] += dense_found
    total_found["bm25"] += bm25_found
    total_found["gpt_old"] += gpt_old_found
    total_found["gpt_full"] += gpt_full_found
    total_found["gpt_court_old"] += gpt_court_old_found
    total_found["gpt_court_full"] += gpt_court_full_found
    total_found["combined"] += combined_found

    print(f"{qid}: {len(gold)} gold")
    print(f"  Dense:          {dense_found}")
    print(f"  BM25:           {bm25_found}")
    print(f"  GPT old arts:   {gpt_old_found}")
    print(f"  GPT full (new): {gpt_full_found}")
    print(f"  Court GPT old:  {gpt_court_old_found}")
    print(f"  Court GPT full: {gpt_court_full_found}")
    print(f"  COMBINED:       {combined_found}/{len(gold)} = {combined_found/len(gold):.0%}")

print(f"\n=== OVERALL ===")
print(f"Total gold: {total_gold}")
for k, v in total_found.items():
    print(f"  {k}: {v}/{total_gold} = {v/total_gold:.1%}")
