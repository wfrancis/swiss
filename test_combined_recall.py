"""Test combined recall: dense + BM25 + GPT precompute on val queries."""
import csv
import json
import pickle
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE = Path(__file__).parent
INDEX = BASE / "index"
DATA = BASE / "data"

def tokenize(text):
    return [t for t in re.findall(r'[a-zäöüß]+', text.lower()) if len(t) > 1]

print("Loading assets...", flush=True)

# Dense
index = faiss.read_index(str(INDEX / "faiss_laws.index"))
with open(INDEX / "faiss_laws_citations.pkl", "rb") as f:
    law_citations = pickle.load(f)
model = SentenceTransformer("intfloat/multilingual-e5-large")

# BM25
with open(INDEX / "bm25_laws.pkl", "rb") as f:
    law_data = pickle.load(f)
law_bm25 = law_data["bm25"]
law_cites_bm25 = law_data["citations"]

# GPT precompute
expansions = json.loads((BASE / "precompute" / "val_query_expansions.json").read_text())
case_citations = json.loads((BASE / "precompute" / "val_case_citations.json").read_text())

# Court citations for verification
with open(INDEX / "court_citations.pkl", "rb") as f:
    all_court_cites = pickle.load(f)
court_set = set(all_court_cites)
law_set = set(law_citations)

# Val data
with open(DATA / "val.csv", "r") as f:
    val_rows = list(csv.DictReader(f))

print(f"Loaded all assets\n", flush=True)

total_gold = 0
total_found_dense = 0
total_found_bm25 = 0
total_found_gpt = 0
total_found_combined = 0
total_gold_laws = 0
total_gold_court = 0

for row in val_rows:
    qid = row["query_id"]
    query = row["query"]
    gold = row["gold_citations"].split(";")
    exp = expansions.get(qid, {})
    cases = case_citations.get(qid, {})

    gold_laws = [c for c in gold if c in law_set]
    gold_court = [c for c in gold if c not in law_set]

    # === Source 1: Dense retrieval (top-200 laws) ===
    q_emb = model.encode([f"query: {query}"], normalize_embeddings=True)
    scores, indices = index.search(q_emb.astype(np.float32), 200)
    dense_hits = set(law_citations[idx] for idx in indices[0])

    # === Source 2: BM25 (same as V3 pipeline) ===
    bm25_hits = set()
    for bq in exp.get("bm25_queries_laws", []):
        tokens = tokenize(bq)
        if not tokens:
            continue
        scores_arr = law_bm25.get_scores(tokens)
        top_idx = scores_arr.argsort()[-40:][::-1]
        for idx in top_idx:
            if scores_arr[idx] > 0:
                bm25_hits.add(law_cites_bm25[idx])

    if exp.get("german_terms"):
        tokens = tokenize(" ".join(exp["german_terms"]))
        if tokens:
            scores_arr = law_bm25.get_scores(tokens)
            top_idx = scores_arr.argsort()[-40:][::-1]
            for idx in top_idx:
                if scores_arr[idx] > 0:
                    bm25_hits.add(law_cites_bm25[idx])

    # === Source 3: GPT specific articles ===
    gpt_law_hits = set()
    for art in exp.get("specific_articles", []):
        if art in law_set:
            gpt_law_hits.add(art)

    # === Source 4: GPT court citations ===
    gpt_court_hits = set()
    for cit in cases.get("expanded", []):
        if cit in court_set:
            gpt_court_hits.add(cit)

    # Combined
    combined_laws = dense_hits | bm25_hits | gpt_law_hits
    combined_court = gpt_court_hits  # No dense court yet

    # Recall per source
    dense_found = sum(1 for c in gold_laws if c in dense_hits)
    bm25_found = sum(1 for c in gold_laws if c in bm25_hits)
    gpt_law_found = sum(1 for c in gold_laws if c in gpt_law_hits)
    combined_law_found = sum(1 for c in gold_laws if c in combined_laws)
    court_found = sum(1 for c in gold_court if c in combined_court)

    total_gold += len(gold)
    total_gold_laws += len(gold_laws)
    total_gold_court += len(gold_court)
    total_found_dense += dense_found
    total_found_bm25 += bm25_found
    total_found_gpt += gpt_law_found
    total_found_combined += combined_law_found + court_found

    print(f"{qid}: {len(gold)} gold ({len(gold_laws)} law, {len(gold_court)} court)")
    print(f"  Dense law recall:    {dense_found}/{len(gold_laws)}")
    print(f"  BM25 law recall:     {bm25_found}/{len(gold_laws)}")
    print(f"  GPT law recall:      {gpt_law_found}/{len(gold_laws)}")
    print(f"  Combined law recall: {combined_law_found}/{len(gold_laws)}")
    print(f"  Court recall (GPT):  {court_found}/{len(gold_court)}")

    # Show what each source uniquely finds
    only_dense = set(c for c in gold_laws if c in dense_hits and c not in bm25_hits and c not in gpt_law_hits)
    only_bm25 = set(c for c in gold_laws if c in bm25_hits and c not in dense_hits and c not in gpt_law_hits)
    only_gpt = set(c for c in gold_laws if c in gpt_law_hits and c not in dense_hits and c not in bm25_hits)
    if only_dense:
        print(f"  [UNIQUE to dense]: {', '.join(only_dense)}")
    if only_bm25:
        print(f"  [UNIQUE to BM25]:  {', '.join(only_bm25)}")
    if only_gpt:
        print(f"  [UNIQUE to GPT]:   {', '.join(only_gpt)}")

print(f"\n=== OVERALL ===")
print(f"Total gold: {total_gold} ({total_gold_laws} law, {total_gold_court} court)")
print(f"Dense law recall:    {total_found_dense}/{total_gold_laws} = {total_found_dense/total_gold_laws:.1%}")
print(f"BM25 law recall:     {total_found_bm25}/{total_gold_laws} = {total_found_bm25/total_gold_laws:.1%}")
print(f"GPT law recall:      {total_found_gpt}/{total_gold_laws} = {total_found_gpt/total_gold_laws:.1%}")
print(f"Combined law recall: {total_found_combined - sum(1 for r in val_rows for c in r['gold_citations'].split(';') if c not in law_set and c in court_set and c in case_citations.get(r['query_id'], {}).get('expanded', []))}/{total_gold_laws}")
print(f"Combined TOTAL recall: {total_found_combined}/{total_gold} = {total_found_combined/total_gold:.1%}")
