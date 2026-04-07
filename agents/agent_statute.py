"""
Agent A: Statute Retrieval
Searches laws_de.csv using BM25 + optional dense retrieval.
"""
import pickle
import re
from pathlib import Path
from typing import List, Tuple


class StatuteAgent:
    def __init__(self, index_path: str = None):
        if index_path is None:
            index_path = Path(__file__).parent.parent / "index" / "bm25_laws.pkl"
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.citations = data["citations"]
        self.texts = data["texts"]
        self.titles = data.get("titles", [""] * len(self.citations))

    def tokenize(self, text: str) -> list:
        text = text.lower()
        tokens = re.findall(r'[a-zäöüß]+', text)
        return [t for t in tokens if len(t) > 1]

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float, str]]:
        """
        Search statutes by query.
        Returns list of (citation, score, text_snippet).
        """
        tokens = self.tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((
                    self.citations[idx],
                    float(scores[idx]),
                    self.texts[idx][:500],
                ))
        return results

    def search_multi(self, queries: List[str], top_k: int = 50) -> List[Tuple[str, float, str]]:
        """
        Run multiple queries and merge results by max score per citation.
        """
        seen = {}
        for q in queries:
            for citation, score, text in self.search(q, top_k):
                if citation not in seen or score > seen[citation][0]:
                    seen[citation] = (score, text)
        results = [(cit, score, text) for cit, (score, text) in seen.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_by_statute_prefix(self, prefix: str, top_k: int = 100) -> List[Tuple[str, float, str]]:
        """
        Find all articles matching a statute abbreviation (e.g., 'StPO', 'OR').
        Uses direct citation string matching, not BM25.
        """
        results = []
        for i, cit in enumerate(self.citations):
            if prefix in cit:
                results.append((cit, 1.0, self.texts[i][:500]))
        return results[:top_k]
