"""Test dense retrieval recall on val queries against gold citations."""
import csv
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE = Path(__file__).parent
INDEX = BASE / "index"
DATA = BASE / "data"
MODEL_NAME = "intfloat/multilingual-e5-large"

# Load FAISS index + citations
print("Loading FAISS law index...", flush=True)
index = faiss.read_index(str(INDEX / "faiss_laws.index"))
with open(INDEX / "faiss_laws_citations.pkl", "rb") as f:
    law_citations = pickle.load(f)
print(f"Loaded {index.ntotal} vectors", flush=True)

# Load model
print("Loading model...", flush=True)
model = SentenceTransformer(MODEL_NAME)

# Load val queries + gold
print("\nLoading val queries...", flush=True)
with open(DATA / "val.csv", "r") as f:
    val_rows = list(csv.DictReader(f))

law_citation_set = set(law_citations)

for row in val_rows:
    qid = row["query_id"]
    query = row["query"]
    gold = row["gold_citations"].split(";")

    # Split gold into laws vs court
    gold_laws = [c for c in gold if c in law_citation_set]
    gold_court = [c for c in gold if c not in law_citation_set]

    # Embed query
    q_emb = model.encode([f"query: {query}"], normalize_embeddings=True)

    # Search top-200
    scores, indices = index.search(q_emb.astype(np.float32), 200)
    top200 = set(law_citations[idx] for idx in indices[0])

    # Check recall
    found = [c for c in gold_laws if c in top200]
    recall = len(found) / len(gold_laws) if gold_laws else 0

    print(f"\n{qid}: {len(gold_laws)} gold laws, {len(found)} found in top-200 ({recall:.0%})")
    if gold_laws:
        for c in gold_laws:
            status = "FOUND" if c in top200 else "MISS"
            # Get rank if found
            rank = "N/A"
            if c in top200:
                for r, idx in enumerate(indices[0]):
                    if law_citations[idx] == c:
                        rank = r + 1
                        break
            print(f"  [{status}] {c}" + (f" (rank {rank})" if rank != "N/A" else ""))

# Overall stats
print("\n=== OVERALL ===")
total_gold_laws = 0
total_found = 0
for row in val_rows:
    gold = row["gold_citations"].split(";")
    gold_laws = [c for c in gold if c in law_citation_set]

    q_emb = model.encode([f"query: {row['query']}"], normalize_embeddings=True)
    scores, indices = index.search(q_emb.astype(np.float32), 200)
    top200 = set(law_citations[idx] for idx in indices[0])

    total_gold_laws += len(gold_laws)
    total_found += sum(1 for c in gold_laws if c in top200)

print(f"Total gold laws: {total_gold_laws}")
print(f"Found in top-200: {total_found}")
print(f"Overall recall: {total_found/total_gold_laws:.1%}")
