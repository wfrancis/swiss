"""
Ask GPT-5.4 to predict the COMPLETE list of expected citations for each query.
Instead of 5 specific articles, ask for ALL articles + court decisions
that a Swiss Federal Supreme Court decision would cite.
"""
import csv
import json
import os
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
OUT = BASE / "precompute"

SYSTEM_PROMPT = """You are an expert Swiss legal researcher with deep knowledge of Swiss federal law,
cantonal law, and Federal Supreme Court (BGer/BGE) jurisprudence.

Your task: Given a legal question, predict the COMPLETE list of Swiss legal citations that a
Federal Supreme Court decision answering this question would cite.

Swiss court decisions typically cite:
1. SUBSTANTIVE articles: the core legal provisions directly relevant to the issue
2. PROCEDURAL articles: jurisdiction, standing, appeal deadlines, costs, legal aid
3. CONSTITUTIONAL articles: fundamental rights if applicable
4. LEADING CASES (BGE): landmark Federal Supreme Court decisions on the topic
5. RECENT CASES: recent unreported decisions (format: 1B_xxx/yyyy, 5A_xxx/yyyy, etc.)

For a typical Federal Supreme Court decision, expect 15-40 total citations.

IMPORTANT citation format rules:
- Laws: "Art. 221 Abs. 1 StPO", "Art. 100 Abs. 1 BGG", "Art. 8 Abs. 1 BV"
- BGE: "BGE 137 IV 122 E. 4.2" (with Erwägung number)
- Unreported: "1B_210/2023 E. 4.1" (with Erwägung number)
- Always include paragraph (Abs.) and subsection (lit.) when applicable
- Use official Swiss statute abbreviations (StPO, StGB, ZGB, OR, BGG, BV, ATSG, IVG, etc.)

Common procedural citations by domain:
- Criminal appeals to BGer: Art. 100 Abs. 1 BGG, Art. 78 Abs. 1 BGG, Art. 80 Abs. 1 BGG
- Civil appeals: Art. 100 Abs. 1 BGG, Art. 72 Abs. 1 BGG, Art. 74 Abs. 1 BGG
- Social insurance: Art. 82 BGG, Art. 113 BGG
- Pre-trial detention appeals: Art. 393 Abs. 1 StPO, Art. 396 Abs. 1 StPO, Art. 382 Abs. 1 StPO
- Cost allocation: Art. 422 Abs. 1 StPO, Art. 428 Abs. 1 StPO (criminal), Art. 106 ZPO (civil)
- Legal aid: Art. 135 Abs. 3 StPO, Art. 135 Abs. 4 StPO
- Bundesstrafgericht jurisdiction: Art. 37 Abs. 1 StBOG, Art. 39 Abs. 1 StBOG

Respond with a JSON object containing:
- "law_citations": list of article citations (strings in exact format)
- "court_citations": list of BGE and unreported case citations (strings in exact format)
- "reasoning": brief explanation of why these citations are expected
"""

def predict_citations(query, qid):
    """Ask GPT-5.4 to predict full citation list."""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Predict ALL citations for this legal question:\n\n{query}"}
        ],
        temperature=0.3,
        max_tokens=4000,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def main():
    # Load val queries
    with open(DATA / "val.csv", "r") as f:
        val_rows = list(csv.DictReader(f))

    # Load test queries
    with open(DATA / "test.csv", "r") as f:
        test_rows = list(csv.DictReader(f))

    # Process val
    print("=== Val queries ===")
    val_results = {}
    for row in val_rows:
        qid = row["query_id"]
        query = row["query"]
        print(f"\n{qid}: {query[:80]}...", flush=True)
        try:
            result = predict_citations(query, qid)
            val_results[qid] = result
            n_law = len(result.get("law_citations", []))
            n_court = len(result.get("court_citations", []))
            print(f"  → {n_law} laws + {n_court} court = {n_law + n_court} total")
        except Exception as e:
            print(f"  ERROR: {e}")
            val_results[qid] = {"law_citations": [], "court_citations": [], "error": str(e)}
        time.sleep(0.5)

    # Save val
    out_path = OUT / "val_full_citations.json"
    with open(out_path, "w") as f:
        json.dump(val_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved val results to {out_path}")

    # Check recall on val
    print("\n=== Val Recall Check ===")
    for row in val_rows:
        qid = row["query_id"]
        gold = set(row["gold_citations"].split(";"))
        pred_laws = set(val_results[qid].get("law_citations", []))
        pred_court = set(val_results[qid].get("court_citations", []))
        pred_all = pred_laws | pred_court

        found = gold & pred_all
        print(f"  {qid}: gold={len(gold)}, pred={len(pred_all)}, found={len(found)}, recall={len(found)/len(gold):.1%}")

    # Process test
    print("\n=== Test queries ===")
    test_results = {}
    for row in test_rows:
        qid = row["query_id"]
        query = row["query"]
        print(f"\n{qid}: {query[:80]}...", flush=True)
        try:
            result = predict_citations(query, qid)
            test_results[qid] = result
            n_law = len(result.get("law_citations", []))
            n_court = len(result.get("court_citations", []))
            print(f"  → {n_law} laws + {n_court} court = {n_law + n_court} total")
        except Exception as e:
            print(f"  ERROR: {e}")
            test_results[qid] = {"law_citations": [], "court_citations": [], "error": str(e)}
        time.sleep(0.5)

    # Save test
    out_path = OUT / "test_full_citations.json"
    with open(out_path, "w") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved test results to {out_path}")


if __name__ == "__main__":
    main()
