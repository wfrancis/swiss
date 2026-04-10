"""
Generate specific court case citations using GPT-5.4.
For each query, ask GPT to identify the actual BGE/case numbers likely cited.
This is the key differentiator — BM25 finds the topic, GPT identifies specific cases.
"""
import csv
import json
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
CASE_CITATIONS_MODEL = os.getenv("CASE_CITATIONS_MODEL", "gpt-5.4")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

BASE = Path(__file__).parent.parent


def generate_case_citations(query: str, query_id: str, domain_info: dict = None) -> dict:
    """Ask GPT-5.4 to identify specific Swiss court decisions relevant to a query."""
    domain_context = ""
    if domain_info:
        domain_context = f"""
Domain: {domain_info.get('domain', 'unknown')}
Key statutes: {', '.join(domain_info.get('key_statutes', []))}
Specific articles identified: {', '.join(domain_info.get('specific_articles', [])[:10])}
"""

    resp = client.chat.completions.create(
        model=CASE_CITATIONS_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Swiss federal court (Bundesgericht) legal research expert. "
                    "Given a legal question about Swiss law, identify the specific Swiss court "
                    "decisions (BGE and individual case numbers) that are most likely to be cited "
                    "as authoritative precedent.\n\n"
                    "Output JSON with:\n"
                    "{\n"
                    '  "bge_citations": [\n'
                    '    {"case": "BGE 137 IV 122", "erwägungen": ["E. 4.1", "E. 4.2", "E. 6.2", "E. 6.4"], "relevance": "leading case on X"},\n'
                    '    ...\n'
                    "  ],\n"
                    '  "individual_cases": [\n'
                    '    {"case": "1B_210/2023", "erwägungen": ["E. 4.1"], "relevance": "recent application of X"},\n'
                    '    ...\n'
                    "  ],\n"
                    '  "reasoning": "brief explanation"\n'
                    "}\n\n"
                    "IMPORTANT:\n"
                    "- Include Erwägung (E.) numbers when you can identify specific relevant sections\n"
                    "- Format BGE as 'BGE NNN ROMAN PAGE' (e.g., BGE 137 IV 122)\n"
                    "- Format individual cases as 'XY_NNN/YYYY' (e.g., 1B_210/2023)\n"
                    "- Include both classic leading BGE decisions and recent individual cases\n"
                    "- Be comprehensive — list 10-30 potentially relevant decisions\n"
                    "- For each case, include all Erwägungen that might be relevant"
                ),
            },
            {
                "role": "user",
                "content": f"Query ID: {query_id}\n{domain_context}\n\nQuestion:\n{query}",
            },
        ],
    )
    return json.loads(resp.choices[0].message.content)


def expand_case_citations(result: dict) -> list:
    """Expand GPT output into a flat list of full citation strings."""
    citations = []

    for bge in result.get("bge_citations", []):
        case = bge["case"]
        for e in bge.get("erwägungen", []):
            full = f"{case} {e}"
            citations.append(full)
        # Also add the case without Erwägung for matching
        if not bge.get("erwägungen"):
            citations.append(case)

    for ind in result.get("individual_cases", []):
        case = ind["case"]
        for e in ind.get("erwägungen", []):
            full = f"{case} {e}"
            citations.append(full)
        if not ind.get("erwägungen"):
            citations.append(case)

    return citations


def load_split_rows(
    split: str,
    offset: int,
    limit: int | None,
    query_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    with open(BASE / "data" / f"{split}.csv", "r") as f:
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
    exp_path = BASE / "precompute" / f"{split}_query_expansions.json"
    out_path = BASE / "precompute" / f"{split}_case_citations.json"
    domain_expansions = json.loads(exp_path.read_text()) if exp_path.exists() else {}
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    rows = load_split_rows(split, offset=offset, limit=limit, query_ids=query_ids)

    print(f"=== Generating case citations for {split} queries ===")
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
            print(f"[{i}/{len(rows)}] {qid}...", flush=True)
            try:
                domain_info = domain_expansions.get(qid, {})
                result = generate_case_citations(row["query"], qid, domain_info)
                expanded = expand_case_citations(result)
                results[qid] = {"raw": result, "expanded": expanded}
                out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
                print(f"  -> {len(expanded)} case citations", flush=True)
                processed += 1
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                results[qid] = {"raw": {}, "expanded": []}
                out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        return processed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                generate_case_citations,
                row["query"],
                row["query_id"],
                domain_expansions.get(row["query_id"], {}),
            ): (i, row)
            for i, row in pending_rows
        }
        total = len(future_map)
        for done_idx, future in enumerate(as_completed(future_map), start=1):
            i, row = future_map[future]
            qid = row["query_id"]
            print(f"[{done_idx}/{total}] {qid}...", flush=True)
            try:
                result = future.result()
                expanded = expand_case_citations(result)
                results[qid] = {"raw": result, "expanded": expanded}
                out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
                print(f"  -> {len(expanded)} case citations", flush=True)
                processed += 1
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                results[qid] = {"raw": {}, "expanded": []}
                out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("splits", nargs="*", choices=["train", "val", "test"])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--query-ids", type=Path)
    parser.add_argument("--max-workers", type=int, default=1)
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
    processed = 0
    for split in splits:
        processed += process_split(
            split,
            offset=args.offset,
            limit=args.limit,
            query_ids=query_ids,
            max_workers=args.max_workers,
        )
    print(f"\nDone! New rows processed across {', '.join(splits)}: {processed}")


if __name__ == "__main__":
    main()
