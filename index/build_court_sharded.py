"""
Build sharded BM25 index for court_considerations.csv.
Splits into N shards to reduce peak memory usage.
"""
import csv
import gc
import pickle
import re
import sys
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_DIR = Path(__file__).parent.parent / "index"
N_SHARDS = 4


def tokenize(text):
    return [t for t in re.findall(r'[a-zäöüß]+', text.lower()) if len(t) > 1]


def main():
    csv.field_size_limit(10000000)
    t0 = time.time()

    # First pass: count rows and split into shards
    print("Counting rows...", flush=True)
    total = 0
    with open(DATA_DIR / "court_considerations.csv", "r") as f:
        for _ in csv.DictReader(f):
            total += 1
    shard_size = total // N_SHARDS + 1
    print(f"Total rows: {total}, shard size: {shard_size}", flush=True)

    # Build each shard
    for shard_id in range(N_SHARDS):
        start = shard_id * shard_size
        end = min(start + shard_size, total)
        print(f"\n--- Shard {shard_id} ({start}-{end}) ---", flush=True)

        citations = []
        tokenized = []

        t1 = time.time()
        with open(DATA_DIR / "court_considerations.csv", "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i < start:
                    continue
                if i >= end:
                    break
                citations.append(row["citation"])
                text = f"{row['citation']} {row['text']}"
                tokenized.append(tokenize(text))

        print(f"  Loaded {len(citations)} rows in {time.time()-t1:.0f}s", flush=True)

        t2 = time.time()
        bm25 = BM25Okapi(tokenized)
        print(f"  BM25 built in {time.time()-t2:.0f}s", flush=True)

        # Save shard - only BM25 and citations (skip raw text to save memory)
        shard_path = INDEX_DIR / f"bm25_court_shard{shard_id}.pkl"
        with open(shard_path, "wb") as f:
            pickle.dump({"bm25": bm25, "citations": citations}, f)
        size = shard_path.stat().st_size / 1e6
        print(f"  Saved {shard_path.name} ({size:.0f} MB)", flush=True)

        # Free memory
        del bm25, tokenized, citations
        gc.collect()

    # Also save a citation-only index for fast lookup
    print("\nBuilding citation lookup...", flush=True)
    all_citations = []
    with open(DATA_DIR / "court_considerations.csv", "r") as f:
        for row in csv.DictReader(f):
            all_citations.append(row["citation"])

    with open(INDEX_DIR / "court_citations.pkl", "wb") as f:
        pickle.dump(all_citations, f)
    print(f"Saved {len(all_citations)} citations to court_citations.pkl", flush=True)

    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
