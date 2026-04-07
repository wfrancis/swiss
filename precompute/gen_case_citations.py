"""
Generate specific court case citations using GPT-5.4.
For each query, ask GPT to identify the actual BGE/case numbers likely cited.
This is the key differentiator — BM25 finds the topic, GPT identifies specific cases.
"""
import csv
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI
client = OpenAI()

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
        model="gpt-5.4",
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


def main():
    # Load existing query expansions for domain context
    val_exp_path = BASE / "precompute" / "val_query_expansions.json"
    test_exp_path = BASE / "precompute" / "test_query_expansions.json"

    val_exp = json.loads(val_exp_path.read_text()) if val_exp_path.exists() else {}
    test_exp = json.loads(test_exp_path.read_text()) if test_exp_path.exists() else {}

    # Process val queries
    print("=== Generating case citations for val queries ===")
    val_cases = {}
    with open(BASE / "data" / "val.csv", "r") as f:
        val_rows = list(csv.DictReader(f))

    for i, row in enumerate(val_rows):
        qid = row["query_id"]
        print(f"[{i+1}/{len(val_rows)}] {qid}...", flush=True)
        try:
            domain_info = val_exp.get(qid, {})
            result = generate_case_citations(row["query"], qid, domain_info)
            expanded = expand_case_citations(result)
            val_cases[qid] = {"raw": result, "expanded": expanded}
            print(f"  -> {len(expanded)} case citations", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            val_cases[qid] = {"raw": {}, "expanded": []}

    (BASE / "precompute" / "val_case_citations.json").write_text(
        json.dumps(val_cases, ensure_ascii=False, indent=2)
    )

    # Process test queries
    print("\n=== Generating case citations for test queries ===")
    test_cases = {}
    with open(BASE / "data" / "test.csv", "r") as f:
        test_rows = list(csv.DictReader(f))

    for i, row in enumerate(test_rows):
        qid = row["query_id"]
        print(f"[{i+1}/{len(test_rows)}] {qid}...", flush=True)
        try:
            domain_info = test_exp.get(qid, {})
            result = generate_case_citations(row["query"], qid, domain_info)
            expanded = expand_case_citations(result)
            test_cases[qid] = {"raw": result, "expanded": expanded}
            print(f"  -> {len(expanded)} case citations", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            test_cases[qid] = {"raw": {}, "expanded": []}

    (BASE / "precompute" / "test_case_citations.json").write_text(
        json.dumps(test_cases, ensure_ascii=False, indent=2)
    )

    print(f"\nDone! Val: {len(val_cases)}, Test: {len(test_cases)}")


if __name__ == "__main__":
    main()
