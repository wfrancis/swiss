"""
Generate domain-specific query expansion templates using GPT-5.4.
For each legal domain, pre-generate German BM25 search queries
that would find relevant statutes and case law.
"""
import json
import os
import csv
import sys
from pathlib import Path
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

client = OpenAI()

# Sample val queries to understand the pattern
VAL_SAMPLES = []
val_path = Path(__file__).parent.parent / "data" / "val.csv"
if val_path.exists():
    with open(val_path, "r") as f:
        for row in csv.DictReader(f):
            VAL_SAMPLES.append({
                "query": row["query"][:500],
                "gold_citations": row["gold_citations"][:300],
            })


def generate_expansions_for_query(query: str, query_id: str) -> dict:
    """Generate German search terms and BM25 queries for an English legal question."""
    resp = client.chat.completions.create(
        model="gpt-5.4",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Swiss legal research expert. Given an English legal question about "
                    "Swiss law, generate German search queries to find relevant statutes and court "
                    "decisions in a corpus of Swiss federal law.\n\n"
                    "Output JSON with:\n"
                    '{\n'
                    '  "domain": "primary legal domain",\n'
                    '  "sub_domains": ["list", "of", "related", "domains"],\n'
                    '  "key_statutes": ["OR", "ZGB", "StGB", ...],\n'
                    '  "german_terms": ["list of German legal terms relevant to the question"],\n'
                    '  "bm25_queries_laws": ["5-10 German keyword queries for statute search"],\n'
                    '  "bm25_queries_court": ["5-10 German keyword queries for court decision search"],\n'
                    '  "specific_articles": ["Art. 221 StPO", ...] (if identifiable from the question),\n'
                    '  "estimated_citation_count": number,\n'
                    '  "reasoning": "brief explanation of search strategy"\n'
                    "}\n\n"
                    "For BM25 queries, use specific German legal terminology. Include both broad "
                    "and narrow queries. For court searches, include case-specific factual terms."
                ),
            },
            {
                "role": "user",
                "content": f"Query ID: {query_id}\n\nQuestion:\n{query}",
            },
        ],
    )
    return json.loads(resp.choices[0].message.content)


def generate_generic_domain_expansions(domain: str) -> dict:
    """Generate generic search templates for a legal domain."""
    resp = client.chat.completions.create(
        model="gpt-5.4-mini",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Swiss legal search expert. For a given legal domain, generate "
                    "generic German BM25 search query templates that would help find relevant "
                    "statutes and court decisions.\n\n"
                    "Output JSON with:\n"
                    '{\n'
                    '  "key_statutes": ["abbreviations of relevant Swiss laws"],\n'
                    '  "common_terms": ["frequent German legal terms in this domain"],\n'
                    '  "query_templates": ["template queries with {PLACEHOLDER} for specific terms"],\n'
                    '  "landmark_cases": ["well-known BGE numbers if any"]\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": f"Generate search templates for Swiss {domain}.",
            },
        ],
    )
    return json.loads(resp.choices[0].message.content)


def main():
    output_dir = Path(__file__).parent.parent / "precompute"

    # 1. Generate expansions for val queries (for tuning)
    print("=== Generating expansions for val queries ===")
    val_expansions = {}
    for i, sample in enumerate(VAL_SAMPLES):
        qid = f"val_{i+1:03d}"
        print(f"[{i+1}/{len(VAL_SAMPLES)}] {qid}: {sample['query'][:80]}...")
        try:
            result = generate_expansions_for_query(sample["query"], qid)
            val_expansions[qid] = result
            print(f"  -> {len(result.get('german_terms', []))} terms, "
                  f"{len(result.get('bm25_queries_laws', []))} law queries")
        except Exception as e:
            print(f"  ERROR: {e}")

    (output_dir / "val_query_expansions.json").write_text(
        json.dumps(val_expansions, ensure_ascii=False, indent=2)
    )

    # 2. Generate expansions for test queries
    print("\n=== Generating expansions for test queries ===")
    test_path = Path(__file__).parent.parent / "data" / "test.csv"
    test_expansions = {}
    if test_path.exists():
        with open(test_path, "r") as f:
            test_rows = list(csv.DictReader(f))
        for i, row in enumerate(test_rows):
            qid = row["query_id"]
            print(f"[{i+1}/{len(test_rows)}] {qid}: {row['query'][:80]}...")
            try:
                result = generate_expansions_for_query(row["query"], qid)
                test_expansions[qid] = result
                print(f"  -> {len(result.get('german_terms', []))} terms")
            except Exception as e:
                print(f"  ERROR: {e}")

    (output_dir / "test_query_expansions.json").write_text(
        json.dumps(test_expansions, ensure_ascii=False, indent=2)
    )

    # 3. Generate generic domain templates
    print("\n=== Generating domain templates ===")
    domains = [
        "criminal law", "criminal procedure", "civil obligations/contracts",
        "family law", "property law", "inheritance law", "civil procedure",
        "administrative law", "constitutional law", "social insurance",
        "immigration law", "tax law", "environmental law", "planning law",
        "competition law", "intellectual property", "debt enforcement",
        "labor law", "tenancy law", "corporate law", "traffic law",
        "data protection", "federal court procedure", "international private law",
        "unfair competition", "banking/finance law",
    ]
    domain_templates = {}
    for i, domain in enumerate(domains):
        print(f"[{i+1}/{len(domains)}] {domain}")
        try:
            result = generate_generic_domain_expansions(domain)
            domain_templates[domain] = result
        except Exception as e:
            print(f"  ERROR: {e}")

    (output_dir / "domain_templates.json").write_text(
        json.dumps(domain_templates, ensure_ascii=False, indent=2)
    )

    print("\nDone! Generated:")
    print(f"  - {len(val_expansions)} val query expansions")
    print(f"  - {len(test_expansions)} test query expansions")
    print(f"  - {len(domain_templates)} domain templates")


if __name__ == "__main__":
    main()
