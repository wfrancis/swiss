"""
Embed corpus with multilingual-e5-large for dense retrieval.
Creates FAISS indexes for both laws_de.csv and court_considerations.csv.

multilingual-e5-large maps English and German to the same vector space,
bridging the cross-lingual gap that BM25 cannot handle.
"""
import csv
import gc
import os
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
INDEX = BASE / "index"

MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH_SIZE = 64  # Conservative for CPU/RAM
COURT_SHARD_SIZE = 200_000  # Smaller shards to limit peak memory


def embed_laws():
    """Embed laws_de.csv and build FAISS index."""
    print("=== Embedding laws_de.csv ===", flush=True)
    t0 = time.time()

    # Load model
    print("Loading model...", flush=True)
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded: {MODEL_NAME} (dim={dim})", flush=True)

    # Load corpus
    print("Loading laws...", flush=True)
    citations = []
    texts = []
    with open(DATA / "laws_de.csv", "r") as f:
        for row in csv.DictReader(f):
            citations.append(row["citation"])
            # For e5 models, prefix with "passage: " for documents
            text = f"passage: {row['citation']} {row.get('title', '')} {row['text']}"
            texts.append(text[:512])  # Truncate to model's sweet spot

    print(f"Loaded {len(citations)} laws", flush=True)

    # Embed in batches
    print(f"Embedding {len(texts)} texts (batch_size={BATCH_SIZE})...", flush=True)
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # For cosine similarity via dot product
    )
    print(f"Embedded in {time.time()-t0:.0f}s, shape: {embeddings.shape}", flush=True)

    # Build FAISS index
    print("Building FAISS index...", flush=True)
    index = faiss.IndexFlatIP(dim)  # Inner product (= cosine for normalized vectors)
    index.add(embeddings.astype(np.float32))
    print(f"FAISS index: {index.ntotal} vectors", flush=True)

    # Save
    faiss.write_index(index, str(INDEX / "faiss_laws.index"))
    with open(INDEX / "faiss_laws_citations.pkl", "wb") as f:
        pickle.dump(citations, f)

    size = os.path.getsize(INDEX / "faiss_laws.index") / 1e6
    print(f"Saved faiss_laws.index ({size:.0f} MB)", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s\n", flush=True)

    # Quick test: search with an English query
    print("Quick test: English query → German law results", flush=True)
    query = "query: pre-trial detention extension proportionality criminal procedure"
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(q_emb.astype(np.float32), 10)
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        print(f"  {i+1}. [{score:.4f}] {citations[idx]}", flush=True)

    del model, embeddings
    gc.collect()
    return index, citations


def embed_court():
    """Embed court_considerations.csv in shards and build FAISS index."""
    print("=== Embedding court_considerations.csv ===", flush=True)
    t0 = time.time()
    csv.field_size_limit(10000000)

    # Count total rows
    print("Counting rows...", flush=True)
    total = 0
    with open(DATA / "court_considerations.csv", "r") as f:
        for _ in csv.DictReader(f):
            total += 1
    print(f"Total: {total} rows", flush=True)

    # Load model
    print("Loading model...", flush=True)
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()

    # Build FAISS index incrementally — add each shard and discard to avoid OOM
    index = faiss.IndexFlatIP(dim)
    all_citations = []

    n_shards = (total + COURT_SHARD_SIZE - 1) // COURT_SHARD_SIZE
    for shard_id in range(n_shards):
        start = shard_id * COURT_SHARD_SIZE
        end = min(start + COURT_SHARD_SIZE, total)
        print(f"\n--- Shard {shard_id+1}/{n_shards} ({start}-{end}) ---", flush=True)

        citations = []
        texts = []
        t1 = time.time()
        with open(DATA / "court_considerations.csv", "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i < start:
                    continue
                if i >= end:
                    break
                citations.append(row["citation"])
                text = f"passage: {row['citation']} {row['text']}"
                texts.append(text[:512])

        print(f"  Loaded {len(citations)} rows ({time.time()-t1:.0f}s)", flush=True)

        # Embed
        t2 = time.time()
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        print(f"  Embedded in {time.time()-t2:.0f}s", flush=True)

        # Add to FAISS immediately and discard embeddings
        index.add(embeddings.astype(np.float32))
        all_citations.extend(citations)
        print(f"  FAISS index now: {index.ntotal} vectors", flush=True)

        del texts, citations, embeddings
        gc.collect()

    print(f"\nFinal FAISS index: {index.ntotal} vectors", flush=True)

    # Save
    print("Saving FAISS index to disk...", flush=True)
    faiss.write_index(index, str(INDEX / "faiss_court.index"))
    with open(INDEX / "faiss_court_citations.pkl", "wb") as f:
        pickle.dump(all_citations, f)

    size = os.path.getsize(INDEX / "faiss_court.index") / 1e9
    print(f"Saved faiss_court.index ({size:.1f} GB)", flush=True)

    # Quick test
    print("\nQuick test: English query → German court results", flush=True)
    query = "query: pre-trial detention extension proportionality criminal procedure"
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(q_emb.astype(np.float32), 10)
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        print(f"  {i+1}. [{score:.4f}] {all_citations[idx]}", flush=True)

    print(f"\nTotal court embedding time: {time.time()-t0:.0f}s", flush=True)

    del model, index
    gc.collect()


if __name__ == "__main__":
    # Default: embed laws first (fast, ~5 min)
    if "--court-only" not in sys.argv:
        embed_laws()

    if "--laws-only" not in sys.argv:
        embed_court()
