"""
Agent B: Case Law Retrieval
Searches court_considerations.csv using sharded BM25 indexes.
"""
import gc
import pickle
import re
from pathlib import Path
from typing import List, Tuple


class CaselawAgent:
    def __init__(self, index_dir: str = None):
        if index_dir is None:
            index_dir = Path(__file__).parent.parent / "index"
        else:
            index_dir = Path(index_dir)

        # Load shards lazily — just store paths
        self.index_dir = index_dir
        self.shard_paths = sorted(index_dir.glob("bm25_court_shard*.pkl"))
        self.n_shards = len(self.shard_paths)

        # Load citation lookup for co-citation search
        cit_path = index_dir / "court_citations.pkl"
        if cit_path.exists():
            with open(cit_path, "rb") as f:
                self.all_citations = pickle.load(f)
        else:
            self.all_citations = []

        print(f"CaselawAgent: {self.n_shards} shards, {len(self.all_citations)} total citations")

    def tokenize(self, text: str) -> list:
        text = text.lower()
        tokens = re.findall(r'[a-zäöüß]+', text)
        return [t for t in tokens if len(t) > 1]

    def _search_shard(self, shard_path: Path, tokens: list, top_k: int) -> List[Tuple[str, float]]:
        """Search a single shard and return top-k (citation, score) pairs."""
        with open(shard_path, "rb") as f:
            data = pickle.load(f)
        bm25 = data["bm25"]
        citations = data["citations"]

        scores = bm25.get_scores(tokens)
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((citations[idx], float(scores[idx])))

        del bm25, data
        gc.collect()
        return results

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float, str]]:
        """Search across all shards and merge results."""
        tokens = self.tokenize(query)
        if not tokens:
            return []

        # Search each shard
        all_results = {}
        for shard_path in self.shard_paths:
            shard_results = self._search_shard(shard_path, tokens, top_k)
            for cit, score in shard_results:
                if cit not in all_results or score > all_results[cit]:
                    all_results[cit] = score

        # Sort by score and return top_k
        ranked = sorted(all_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(cit, score, "") for cit, score in ranked]

    def search_multi(self, queries: List[str], top_k: int = 50) -> List[Tuple[str, float, str]]:
        """Run multiple queries and merge by max score."""
        seen = {}
        for q in queries:
            for citation, score, _ in self.search(q, top_k):
                if citation not in seen or score > seen[citation]:
                    seen[citation] = score
        results = [(cit, score, "") for cit, score in seen.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_by_case_number(self, case_prefix: str) -> List[Tuple[str, float, str]]:
        """
        Find all considerations for a specific case (e.g., 'BGE 137 IV 122').
        Uses the citation lookup — no BM25 needed.
        """
        base = re.sub(r'\s+E\.\s+.*$', '', case_prefix).strip()
        results = []
        for cit in self.all_citations:
            if cit.startswith(base):
                results.append((cit, 1.0, ""))
        return results
