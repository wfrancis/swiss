"""
Orchestrator: Coordinates multi-agent retrieval pipeline.

Implements Claude Code swarm patterns:
1. Plan-Then-Execute: Classify domain, plan agents, execute
2. Multi-Gate Filtering: BM25 -> dedup -> rerank -> threshold
3. Four-Phase Consolidation: Orient -> Gather -> Consolidate -> Prune
4. Cross-Query Learning: Track what works across queries
"""
import json
import re
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter

from agents.agent_statute import StatuteAgent
from agents.agent_caselaw import CaselawAgent


class Orchestrator:
    def __init__(
        self,
        law_index_path: str = None,
        court_index_dir: str = None,
        glossary_path: str = None,
        query_expansions_path: str = None,
    ):
        # Load agents
        print("Loading statute agent...")
        self.statute_agent = StatuteAgent(law_index_path)
        print("Loading caselaw agent...")
        self.caselaw_agent = CaselawAgent(court_index_dir)

        # Load precomputed assets
        self.glossary = {}
        if glossary_path and Path(glossary_path).exists():
            self.glossary = json.loads(Path(glossary_path).read_text())
            print(f"Loaded glossary: {len(self.glossary.get('flat_lookup', {}))} terms")

        self.query_expansions = {}
        if query_expansions_path and Path(query_expansions_path).exists():
            self.query_expansions = json.loads(Path(query_expansions_path).read_text())
            print(f"Loaded query expansions: {len(self.query_expansions)} queries")

        # Cross-query learning state
        self.query_stats = []

    def translate_terms(self, english_query: str) -> List[str]:
        """
        Extract German legal terms from English query using glossary.
        Returns list of German terms.
        """
        german_terms = []
        query_lower = english_query.lower()
        flat = self.glossary.get("flat_lookup", {})

        for en_term, info in flat.items():
            if en_term in query_lower:
                german_terms.append(info["de"])

        return german_terms

    def extract_statute_refs(self, query: str) -> List[str]:
        """Extract explicit statute references from query text."""
        # Match patterns like "Art. 221 StPO", "Art. 10a USG", etc.
        pattern = r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+(?:\s+lit\.\s+[a-z])?)?\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü]+'
        matches = re.findall(pattern, query)
        return matches

    def extract_case_refs(self, query: str) -> List[str]:
        """Extract explicit case references from query text."""
        # BGE pattern
        bge = re.findall(r'BGE\s+\d+\s+[IVX]+\s+\d+', query)
        # Federal court case number pattern
        cases = re.findall(r'\d[A-Z]_\d+/\d{4}', query)
        return bge + cases

    def orient(self, query_id: str, query: str) -> Dict:
        """
        Phase 1: Orient — Classify query and plan retrieval strategy.
        """
        plan = {
            "query_id": query_id,
            "explicit_statutes": self.extract_statute_refs(query),
            "explicit_cases": self.extract_case_refs(query),
            "german_terms": [],
            "bm25_queries_laws": [],
            "bm25_queries_court": [],
            "estimated_citations": 25,  # Default based on val avg
        }

        # Use precomputed expansions if available
        if query_id in self.query_expansions:
            exp = self.query_expansions[query_id]
            plan["german_terms"] = exp.get("german_terms", [])
            plan["bm25_queries_laws"] = exp.get("bm25_queries_laws", [])
            plan["bm25_queries_court"] = exp.get("bm25_queries_court", [])
            plan["estimated_citations"] = exp.get("estimated_citation_count", 25)
            plan["key_statutes"] = exp.get("key_statutes", [])
            plan["specific_articles"] = exp.get("specific_articles", [])
        else:
            # Fallback: use glossary for translation
            plan["german_terms"] = self.translate_terms(query)
            # Generate basic BM25 queries from German terms
            if plan["german_terms"]:
                plan["bm25_queries_laws"] = [" ".join(plan["german_terms"][:5])]
                plan["bm25_queries_court"] = [" ".join(plan["german_terms"][:5])]

        # Adjust estimates from cross-query learning
        if self.query_stats:
            avg_citations = sum(s["n_citations"] for s in self.query_stats) / len(self.query_stats)
            plan["estimated_citations"] = max(plan["estimated_citations"], int(avg_citations * 0.8))

        return plan

    def gather(self, plan: Dict) -> Dict[str, List[Tuple[str, float, str]]]:
        """
        Phase 2: Gather — Run all retrieval agents.
        """
        results = {"statute": [], "caselaw": [], "explicit": []}

        # Agent A: Statute retrieval
        if plan["bm25_queries_laws"]:
            results["statute"] = self.statute_agent.search_multi(
                plan["bm25_queries_laws"], top_k=100
            )

        # Also search with raw German terms
        if plan["german_terms"]:
            term_query = " ".join(plan["german_terms"])
            term_results = self.statute_agent.search(term_query, top_k=50)
            # Merge
            seen = {r[0] for r in results["statute"]}
            for r in term_results:
                if r[0] not in seen:
                    results["statute"].append(r)
                    seen.add(r[0])

        # Search by specific statute prefixes
        for statute in plan.get("key_statutes", []):
            prefix_results = self.statute_agent.search_by_statute_prefix(statute, top_k=50)
            # These get lower base scores since they're prefix matches
            seen_s = {r[0] for r in results["statute"]}
            for cit, _, text in prefix_results:
                if cit not in seen_s:
                    results["statute"].append((cit, 0.5, text))
                    seen_s.add(cit)

        # Agent B: Caselaw retrieval
        if plan["bm25_queries_court"]:
            results["caselaw"] = self.caselaw_agent.search_multi(
                plan["bm25_queries_court"], top_k=100
            )

        if plan["german_terms"]:
            term_query = " ".join(plan["german_terms"])
            term_results = self.caselaw_agent.search(term_query, top_k=50)
            seen_c = {r[0] for r in results["caselaw"]}
            for r in term_results:
                if r[0] not in seen_c:
                    results["caselaw"].append(r)
                    seen_c.add(r[0])

        # Explicit references from query text
        for art in plan["explicit_statutes"]:
            results["explicit"].append((art, 100.0, "explicit_mention"))
        for case in plan["explicit_cases"]:
            results["explicit"].append((case, 100.0, "explicit_mention"))

        # Specific articles from GPT expansion
        for art in plan.get("specific_articles", []):
            results["explicit"].append((art, 90.0, "gpt_identified"))

        # Agent D: Co-citation expansion
        # For top court results, find sibling Erwägungen from same case
        top_cases = set()
        for cit, score, _ in results["caselaw"][:20]:
            # Extract base case number
            base = re.sub(r'\s+E\.\s+.*$', '', cit).strip()
            if base not in top_cases:
                top_cases.add(base)
                siblings = self.caselaw_agent.search_by_case_number(base)
                seen_c2 = {r[0] for r in results["caselaw"]}
                for sib_cit, _, sib_text in siblings[:10]:
                    if sib_cit not in seen_c2:
                        results["caselaw"].append((sib_cit, score * 0.5, sib_text))
                        seen_c2.add(sib_cit)

        return results

    def consolidate(self, results: Dict, plan: Dict) -> List[Tuple[str, float]]:
        """
        Phase 3: Consolidate — Merge, deduplicate, score.
        Multi-source agreement boosts scores.
        """
        citation_scores = {}
        citation_sources = {}

        # Process each source
        for source, items in results.items():
            for citation, score, text in items:
                citation = citation.strip()
                if citation not in citation_scores:
                    citation_scores[citation] = 0.0
                    citation_sources[citation] = set()

                # Normalize score to 0-1 range for each source
                if source == "explicit":
                    norm_score = 1.0  # Explicit mentions get max
                else:
                    # BM25 scores vary widely; use relative ranking
                    norm_score = min(score / 20.0, 1.0)

                citation_scores[citation] = max(citation_scores[citation], norm_score)
                citation_sources[citation].add(source)

        # Boost for multi-source agreement (Pattern 5: Priority Queue)
        for cit in citation_scores:
            n_sources = len(citation_sources[cit])
            if n_sources >= 2:
                citation_scores[cit] *= 1.5
            if n_sources >= 3:
                citation_scores[cit] *= 1.3

        # Sort by score
        ranked = sorted(citation_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def prune(self, ranked: List[Tuple[str, float]], plan: Dict) -> List[str]:
        """
        Phase 4: Prune — Dynamic cutoff based on score distribution.
        """
        if not ranked:
            return []

        estimated = plan.get("estimated_citations", 25)

        # Strategy: Take top N where N is estimated, but also check for score dropoff
        scores = [s for _, s in ranked]

        # Find natural cutoff: where score drops below 30% of max
        max_score = scores[0] if scores else 1.0
        threshold = max_score * 0.15

        # Take at least estimated count, but stop at threshold
        cutoff = min(len(ranked), max(estimated, 10))
        for i in range(min(estimated, len(scores)), len(scores)):
            if scores[i] < threshold:
                cutoff = i
                break
            cutoff = i + 1

        # Cap at reasonable maximum
        cutoff = min(cutoff, 60)

        selected = [cit for cit, score in ranked[:cutoff]]
        return selected

    def process_query(self, query_id: str, query: str) -> List[str]:
        """
        Full pipeline: Orient -> Gather -> Consolidate -> Prune
        """
        # Phase 1: Orient
        plan = self.orient(query_id, query)
        print(f"  Orient: {len(plan['german_terms'])} DE terms, "
              f"{len(plan['bm25_queries_laws'])} law queries, "
              f"{len(plan['bm25_queries_court'])} court queries, "
              f"est={plan['estimated_citations']} citations")

        # Phase 2: Gather
        results = self.gather(plan)
        total = sum(len(v) for v in results.values())
        print(f"  Gather: {len(results['statute'])} statute, "
              f"{len(results['caselaw'])} caselaw, "
              f"{len(results['explicit'])} explicit = {total} total candidates")

        # Phase 3: Consolidate
        ranked = self.consolidate(results, plan)
        print(f"  Consolidate: {len(ranked)} unique citations after merge")

        # Phase 4: Prune
        selected = self.prune(ranked, plan)
        print(f"  Prune: {len(selected)} final citations")

        # Cross-query learning update
        self.query_stats.append({
            "query_id": query_id,
            "n_citations": len(selected),
            "n_candidates": total,
        })

        return selected


def run_pipeline(
    queries_path: str,
    output_path: str,
    law_index: str = None,
    court_index_dir: str = None,
    glossary: str = None,
    expansions: str = None,
):
    """Run the full pipeline on a set of queries."""
    base = Path(__file__).parent.parent

    orch = Orchestrator(
        law_index_path=law_index or str(base / "index" / "bm25_laws.pkl"),
        court_index_dir=court_index_dir or str(base / "index"),
        glossary_path=glossary or str(base / "precompute" / "legal_glossary.json"),
        query_expansions_path=expansions,
    )

    # Load queries
    with open(queries_path, "r") as f:
        queries = list(csv.DictReader(f))

    # Process each query
    predictions = {}
    for i, row in enumerate(queries):
        qid = row["query_id"]
        query = row["query"]
        print(f"\n[{i+1}/{len(queries)}] Processing {qid}...")
        citations = orch.process_query(qid, query)
        predictions[qid] = citations

    # Write submission
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions.keys()):
            cites_str = ";".join(predictions[qid])
            writer.writerow([qid, cites_str])

    print(f"\nSubmission written to {output_path}")
    return predictions


if __name__ == "__main__":
    import sys
    base = Path(__file__).parent.parent

    # Default: run on val set
    queries = sys.argv[1] if len(sys.argv) > 1 else str(base / "data" / "val.csv")
    output = sys.argv[2] if len(sys.argv) > 2 else str(base / "submissions" / "val_pred.csv")

    # Ensure output dir exists
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    # Determine which expansion file to use
    if "val" in queries:
        exp = str(base / "precompute" / "val_query_expansions.json")
    elif "test" in queries:
        exp = str(base / "precompute" / "test_query_expansions.json")
    else:
        exp = None

    run_pipeline(queries, output, expansions=exp)
