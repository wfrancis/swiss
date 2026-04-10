"""
Generate domain-specific query expansion templates using GPT-5.4.
For each legal domain, pre-generate German BM25 search queries
that would find relevant statutes and case law.
"""
import json
import os
import csv
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
QUERY_EXPANSIONS_MODEL = os.getenv("QUERY_EXPANSIONS_MODEL", "gpt-5.4")
DOMAIN_TEMPLATES_MODEL = os.getenv("DOMAIN_TEMPLATES_MODEL", "gpt-5.4-mini")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

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
        model=QUERY_EXPANSIONS_MODEL,
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
        model=DOMAIN_TEMPLATES_MODEL,
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


def load_split_rows(
    split: str,
    offset: int,
    limit: int | None,
    query_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    split_path = Path(__file__).parent.parent / "data" / f"{split}.csv"
    with split_path.open("r") as f:
        rows = list(csv.DictReader(f))
    if query_ids is not None:
        rows = [row for row in rows if row["query_id"] in query_ids]
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def process_split(
    split: str,
    offset: int,
    limit: int | None,
    query_ids: set[str] | None = None,
    max_workers: int = 1,
) -> int:
    output_dir = Path(__file__).parent.parent / "precompute"
    out_path = output_dir / f"{split}_query_expansions.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    rows = load_split_rows(split, offset=offset, limit=limit, query_ids=query_ids)

    print(f"=== Generating expansions for {split} queries ===")
    processed = 0
    pending_rows = []
    for i, row in enumerate(rows, start=1):
        qid = row["query_id"]
        if qid in results:
            print(f"[{i}/{len(rows)}] {qid}: skip existing", flush=True)
            continue
        pending_rows.append((i, row))

    if max_workers <= 1:
        for i, row in pending_rows:
            qid = row["query_id"]
            print(f"[{i}/{len(rows)}] {qid}: {row['query'][:80]}...", flush=True)
            try:
                result = generate_expansions_for_query(row["query"], qid)
                results[qid] = result
                out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
                print(
                    f"  -> {len(result.get('german_terms', []))} terms, "
                    f"{len(result.get('bm25_queries_laws', []))} law queries",
                    flush=True,
                )
                processed += 1
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
        return processed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(generate_expansions_for_query, row["query"], row["query_id"]): (i, row)
            for i, row in pending_rows
        }
        total = len(future_map)
        for done_idx, future in enumerate(as_completed(future_map), start=1):
            i, row = future_map[future]
            qid = row["query_id"]
            print(f"[{done_idx}/{total}] {qid}: {row['query'][:80]}...", flush=True)
            try:
                result = future.result()
                results[qid] = result
                out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
                print(
                    f"  -> {len(result.get('german_terms', []))} terms, "
                    f"{len(result.get('bm25_queries_laws', []))} law queries",
                    flush=True,
                )
                processed += 1
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)

    return processed


def generate_domain_templates() -> int:
    output_dir = Path(__file__).parent.parent / "precompute"
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
    return len(domain_templates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("splits", nargs="*", choices=["train", "val", "test"])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--query-ids", type=Path)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--skip-domain-templates", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    splits = args.splits or ["val", "test"]
    query_ids = None
    if args.query_ids:
        query_ids = {
            line.strip()
            for line in args.query_ids.read_text().splitlines()
            if line.strip()
        }

    total_processed = 0
    for split in splits:
        total_processed += process_split(
            split,
            offset=args.offset,
            limit=args.limit,
            query_ids=query_ids,
            max_workers=args.max_workers,
        )

    template_count = None
    if not args.skip_domain_templates and not args.splits:
        template_count = generate_domain_templates()

    print("\nDone! Generated:")
    print(f"  - {total_processed} new query expansions across {', '.join(splits)}")
    if template_count is not None:
        print(f"  - {template_count} domain templates")


if __name__ == "__main__":
    main()
