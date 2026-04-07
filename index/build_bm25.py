"""
Build BM25 indexes over laws_de.csv and court_considerations.csv.
Serializes to pickle for fast loading in the Kaggle notebook.
"""
import csv
import pickle
import re
import sys
import time
from pathlib import Path
from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_DIR = Path(__file__).parent.parent / "index"


def tokenize_german(text: str) -> list:
    """Simple German tokenizer: lowercase, split on non-alphanumeric, remove short tokens."""
    text = text.lower()
    # Keep German special chars
    tokens = re.findall(r'[a-zäöüß]+', text)
    # Remove very short tokens (but keep legal abbreviations)
    return [t for t in tokens if len(t) > 1]


def build_law_index():
    """Build BM25 over laws_de.csv."""
    print("Building BM25 index for laws_de.csv...")
    t0 = time.time()

    citations = []
    texts = []
    titles = []

    with open(DATA_DIR / "laws_de.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            citations.append(row["citation"])
            # Combine citation + title + text for richer matching
            combined = f"{row['citation']} {row.get('title', '')} {row['text']}"
            texts.append(combined)
            titles.append(row.get("title", ""))

    print(f"  Loaded {len(citations)} law entries in {time.time()-t0:.1f}s")

    # Tokenize
    t1 = time.time()
    tokenized = [tokenize_german(t) for t in texts]
    print(f"  Tokenized in {time.time()-t1:.1f}s")

    # Build BM25
    t2 = time.time()
    bm25 = BM25Okapi(tokenized)
    print(f"  BM25 built in {time.time()-t2:.1f}s")

    # Save
    index_data = {
        "bm25": bm25,
        "citations": citations,
        "titles": titles,
        "texts": texts,
    }
    out_path = INDEX_DIR / "bm25_laws.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(index_data, f)
    print(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return bm25, citations


def build_court_index():
    """Build BM25 over court_considerations.csv."""
    print("\nBuilding BM25 index for court_considerations.csv...")
    t0 = time.time()

    csv.field_size_limit(10000000)
    citations = []
    texts = []

    with open(DATA_DIR / "court_considerations.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            citations.append(row["citation"])
            combined = f"{row['citation']} {row['text']}"
            texts.append(combined)

    print(f"  Loaded {len(citations)} court entries in {time.time()-t0:.1f}s")

    # Tokenize
    t1 = time.time()
    tokenized = [tokenize_german(t) for t in texts]
    print(f"  Tokenized in {time.time()-t1:.1f}s")

    # Build BM25
    t2 = time.time()
    bm25 = BM25Okapi(tokenized)
    print(f"  BM25 built in {time.time()-t2:.1f}s")

    # Save
    index_data = {
        "bm25": bm25,
        "citations": citations,
        "texts": texts,
    }
    out_path = INDEX_DIR / "bm25_court.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(index_data, f)
    print(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return bm25, citations


def test_search(bm25, citations, query: str, top_k: int = 10):
    """Test a BM25 search query."""
    tokens = tokenize_german(query)
    scores = bm25.get_scores(tokens)
    top_indices = scores.argsort()[-top_k:][::-1]
    print(f"\nQuery: '{query}'")
    print(f"Tokens: {tokens}")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. [{scores[idx]:.2f}] {citations[idx]}")


if __name__ == "__main__":
    # Build law index first (smaller, faster)
    bm25_laws, law_cites = build_law_index()

    # Test with known German legal terms
    test_search(bm25_laws, law_cites, "Sorgfaltspflicht Vertrag Obligationenrecht")
    test_search(bm25_laws, law_cites, "Untersuchungshaft Fluchtgefahr Strafprozessordnung")
    test_search(bm25_laws, law_cites, "Umweltverträglichkeitsprüfung Lärmschutz")

    # Build court index (large, slow)
    if "--laws-only" not in sys.argv:
        bm25_court, court_cites = build_court_index()
        test_search(bm25_court, court_cites, "Untersuchungshaft Kollusionsgefahr Verhältnismässigkeit")
        test_search(bm25_court, court_cites, "Invalidenversicherung berufliche Eingliederung Arbeitsfähigkeit")
