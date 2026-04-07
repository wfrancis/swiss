"""
Embed the court corpus using OpenAI text-embedding-3-small.
- Dedupes by citation (2.5M → 2M unique)
- Truncates text to ~500 chars (~125 tokens) for cost
- Batches 2000 records per API call
- CHECKPOINTING: saves progress every 50k records so we can resume

Cost estimate: ~$4-5 total
"""
import csv
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
INDEX = BASE / "index"

MODEL = "text-embedding-3-small"
DIMENSIONS = 512
MAX_CHARS = 500
BATCH_SIZE = 2000
CHECKPOINT_INTERVAL = 50000  # Save every 50k records

CHECKPOINT_PATH = INDEX / "faiss_court_openai_checkpoint.npy"
CITATIONS_PATH = INDEX / "faiss_court_openai_checkpoint_citations.pkl"
STATE_PATH = INDEX / "faiss_court_openai_state.pkl"


def load_and_dedupe():
    """Load court considerations, dedupe by citation."""
    print(f"Loading court_considerations.csv...", flush=True)
    citation_to_text = {}
    with open(DATA / "court_considerations.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            cit = row["citation"]
            text = row["text"][:MAX_CHARS]
            if cit in citation_to_text:
                if len(text) > len(citation_to_text[cit]):
                    citation_to_text[cit] = text
            else:
                citation_to_text[cit] = text
            if (i + 1) % 500000 == 0:
                print(f"  Processed {i+1:,} rows, {len(citation_to_text):,} unique", flush=True)

    print(f"Total unique citations: {len(citation_to_text):,}", flush=True)
    return citation_to_text


def embed_batch(texts, retries=3):
    """Embed a batch of texts. Retries on failure."""
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model=MODEL,
                input=texts,
                dimensions=DIMENSIONS,
            )
            return [np.array(d.embedding, dtype=np.float32) for d in response.data]
        except Exception as e:
            print(f"    Attempt {attempt+1}/{retries} failed: {e}", flush=True)
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def save_checkpoint(embeddings, citations, start_idx, completed):
    """Save current progress to disk."""
    # Only save the completed portion
    np.save(CHECKPOINT_PATH, embeddings[:completed])
    with open(CITATIONS_PATH, "wb") as f:
        pickle.dump(citations[:completed], f)
    with open(STATE_PATH, "wb") as f:
        pickle.dump({"completed": completed, "total": len(citations)}, f)
    print(f"    [checkpoint saved at {completed:,}]", flush=True)


def load_checkpoint():
    """Load previous checkpoint if it exists."""
    if STATE_PATH.exists() and CHECKPOINT_PATH.exists():
        with open(STATE_PATH, "rb") as f:
            state = pickle.load(f)
        embeddings = np.load(CHECKPOINT_PATH)
        with open(CITATIONS_PATH, "rb") as f:
            saved_citations = pickle.load(f)
        print(f"Resuming from checkpoint at {state['completed']:,}/{state['total']:,}", flush=True)
        return state["completed"], embeddings, saved_citations
    return 0, None, None


def main():
    test_mode = "--test" in sys.argv
    force_restart = "--restart" in sys.argv

    if test_mode:
        print("=== TEST MODE (1000 records) ===")

    if force_restart:
        print("Forcing restart — removing checkpoints", flush=True)
        for p in [CHECKPOINT_PATH, CITATIONS_PATH, STATE_PATH]:
            if p.exists():
                p.unlink()

    # Load dataset
    citation_to_text = load_and_dedupe()
    if test_mode:
        items = list(citation_to_text.items())[:1000]
    else:
        items = list(citation_to_text.items())

    n = len(items)
    texts = [f"{cit} {text}" for cit, text in items]
    citations = [cit for cit, _ in items]

    # Check for checkpoint
    start_idx, saved_embeddings, saved_citations = load_checkpoint()

    # Validate checkpoint matches current dataset
    if saved_citations is not None:
        # Check first few citations match
        match = all(saved_citations[i] == citations[i] for i in range(min(100, len(saved_citations))))
        if not match:
            print("WARNING: checkpoint citations don't match current dataset, starting over", flush=True)
            start_idx = 0
            saved_embeddings = None

    # Allocate full array
    all_embeddings = np.zeros((n, DIMENSIONS), dtype=np.float32)
    if saved_embeddings is not None and start_idx > 0:
        all_embeddings[:start_idx] = saved_embeddings
        print(f"Loaded {start_idx:,} embeddings from checkpoint", flush=True)

    print(f"\nEmbedding {n-start_idx:,} remaining citations ({n:,} total)...", flush=True)
    print(f"Model: {MODEL}, dim={DIMENSIONS}, max_chars={MAX_CHARS}, batch={BATCH_SIZE}", flush=True)

    total_chars = sum(len(t) for t in texts[start_idx:])
    est_tokens = total_chars / 4
    est_cost = est_tokens / 1e6 * 0.02
    print(f"Estimated remaining: {est_tokens/1e6:.1f}M tokens, ${est_cost:.2f}", flush=True)

    t0 = time.time()
    last_checkpoint = start_idx

    for i in range(start_idx, n, BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        embeddings = embed_batch(batch_texts)
        all_embeddings[i:i + len(embeddings)] = np.array(embeddings)

        completed = i + len(embeddings)

        # Progress logging
        if (i // BATCH_SIZE + 1) % 5 == 0 or i == start_idx:
            elapsed = time.time() - t0
            done_since_start = completed - start_idx
            rate = done_since_start / elapsed if elapsed > 0 else 0
            eta = (n - completed) / rate if rate > 0 else 0
            print(f"  {completed:,}/{n:,} ({completed/n*100:.1f}%) | {rate:.0f} rec/s | ETA {eta/60:.1f}min", flush=True)

        # Checkpoint
        if completed - last_checkpoint >= CHECKPOINT_INTERVAL:
            save_checkpoint(all_embeddings, citations, start_idx, completed)
            last_checkpoint = completed

    print(f"\nEmbedding complete in {(time.time()-t0)/60:.1f} min", flush=True)

    # Final save
    print("Normalizing vectors...", flush=True)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    all_embeddings = all_embeddings / norms

    print("Building FAISS index...", flush=True)
    index = faiss.IndexFlatIP(DIMENSIONS)
    index.add(all_embeddings)

    suffix = "_test" if test_mode else ""
    index_path = INDEX / f"faiss_court_openai{suffix}.index"
    cite_path = INDEX / f"faiss_court_openai_citations{suffix}.pkl"

    faiss.write_index(index, str(index_path))
    with open(cite_path, "wb") as f:
        pickle.dump(citations, f)

    index_size = index_path.stat().st_size / 1e9
    print(f"\nSaved:")
    print(f"  {index_path} ({index_size:.2f} GB)")
    print(f"  {cite_path}")
    print(f"  Total citations indexed: {len(citations):,}")

    # Clean up checkpoint files
    if not test_mode:
        print("Cleaning up checkpoints...", flush=True)
        for p in [CHECKPOINT_PATH, CITATIONS_PATH, STATE_PATH]:
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    main()
