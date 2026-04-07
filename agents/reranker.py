"""
Cross-encoder reranker for scoring (query, citation_text) pairs.
Uses BAAI/bge-reranker-v2-m3 for multilingual relevance scoring.
"""
import csv
import pickle
from pathlib import Path
from sentence_transformers import CrossEncoder

BASE = Path(__file__).parent.parent

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        print(f"Loading reranker: {model_name}...", flush=True)
        self.model = CrossEncoder(model_name, max_length=512)
        print("Reranker loaded.", flush=True)

    def score_pairs(self, query, citation_texts):
        """Score a list of (query, text) pairs. Returns list of scores."""
        if not citation_texts:
            return []
        pairs = [(query, text) for text in citation_texts]
        scores = self.model.predict(pairs)
        return scores.tolist()

    def rerank(self, query, candidates, top_k=None):
        """
        Rerank candidates by cross-encoder score.

        candidates: list of (citation_string, text) tuples
        Returns: list of (citation_string, score) sorted by score desc
        """
        if not candidates:
            return []
        texts = [text for _, text in candidates]
        scores = self.score_pairs(query, texts)
        ranked = sorted(
            zip([cit for cit, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True
        )
        if top_k:
            ranked = ranked[:top_k]
        return ranked


def build_citation_text_map():
    """Build citation -> text lookup from both corpora."""
    DATA = BASE / "data"
    text_map = {}

    # Laws
    print("Loading law texts...", flush=True)
    with open(DATA / "laws_de.csv", "r") as f:
        for row in csv.DictReader(f):
            text_map[row["citation"]] = f"{row['citation']} {row.get('title', '')} {row['text']}"[:512]

    # Court (large — load lazily or subset)
    print("Loading court texts...", flush=True)
    csv.field_size_limit(10000000)
    with open(DATA / "court_considerations.csv", "r") as f:
        for row in csv.DictReader(f):
            text_map[row["citation"]] = f"{row['citation']} {row['text']}"[:512]

    print(f"Loaded {len(text_map)} citation texts", flush=True)
    return text_map


if __name__ == "__main__":
    # Quick test
    reranker = Reranker()
    query = "Can a court extend pre-trial detention for collusion risk?"
    candidates = [
        ("Art. 221 Abs. 1 StPO", "Art. 221 Abs. 1 StPO Untersuchungshaft Kollusionsgefahr"),
        ("Art. 100 Abs. 1 BGG", "Art. 100 Abs. 1 BGG Beschwerdefrist 30 Tage"),
        ("Art. 1 Abs. 1 OR", "Art. 1 Abs. 1 OR Vertrag Antrag Annahme"),
    ]
    results = reranker.rerank(query, candidates)
    for cit, score in results:
        print(f"  [{score:.4f}] {cit}")
