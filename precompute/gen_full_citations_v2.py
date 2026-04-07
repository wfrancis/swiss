"""
V2: Use GPT-5.4 (not GPT-4.1) with enhanced prompt + citation count estimation.
Also: better error handling, retry on JSON errors.
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
3. CONSTITUTIONAL articles: fundamental rights if applicable (Art. 29 BV, Art. 9 BV, etc.)
4. LEADING CASES (BGE): landmark Federal Supreme Court decisions on the topic
5. RECENT CASES: recent unreported decisions (format: 1B_xxx/yyyy, 5A_xxx/yyyy, etc.)
6. AUXILIARY provisions: costs (Art. 66/68 BGG), legal aid (Art. 64 BGG), value thresholds

For a typical Federal Supreme Court decision, expect 15-45 total citations.

CITATION FORMAT RULES (be EXACT):
- Laws: "Art. 221 Abs. 1 StPO", "Art. 100 Abs. 1 BGG", "Art. 8 Abs. 1 BV"
- With subsections: "Art. 221 Abs. 1 lit. b StPO" (use "lit." for letter subsections)
- BGE: "BGE 137 IV 122 E. 4.2" (BGE [volume] [roman numeral division] [page] E. [consideration number])
- Unreported: "1B_210/2023 E. 4.1" (case number format with E. for Erwägung)
- Use Abs. for paragraph numbers, lit. for letter subdivisions
- ALWAYS use official Swiss statute abbreviations: StPO, StGB, ZGB, OR, BGG, BV, ATSG, IVG, SchKG, ZPO, IPRG, StBOG, etc.

COMMON PROCEDURAL CITATIONS BY DOMAIN (include these!):
- All BGer appeals: Art. 42 Abs. 1 BGG (requirements), Art. 100 Abs. 1 BGG (deadline), Art. 106 Abs. 1 BGG (legal grounds)
- Criminal: Art. 78 Abs. 1 BGG, Art. 80 Abs. 1 BGG (jurisdiction), Art. 81 Abs. 1 BGG (standing)
- Civil: Art. 72 Abs. 1 BGG, Art. 74 Abs. 1 BGG (value threshold), Art. 75 Abs. 1 BGG
- Social insurance: Art. 82 lit. a BGG, Art. 113 BGG
- Pre-trial detention: Art. 393 Abs. 1 StPO, Art. 396 Abs. 1 StPO, Art. 382 Abs. 1 StPO
- Bundesstrafgericht: Art. 37 Abs. 1 StBOG, Art. 39 Abs. 1 StBOG
- Costs: Art. 66 Abs. 1 BGG (loser pays), Art. 68 Abs. 1 BGG (party compensation)
- Criminal costs: Art. 422 Abs. 1 StPO, Art. 428 Abs. 1 StPO
- Legal aid: Art. 64 Abs. 1 BGG, Art. 135 Abs. 3-4 StPO (criminal)

IMPORTANT: Be comprehensive. A real BGer decision cites MANY provisions. Include ALL that would
reasonably appear, even seemingly minor procedural provisions.

Respond with a JSON object:
{
  "law_citations": ["Art. X Abs. Y Statute", ...],
  "court_citations": ["BGE XXX Roman Page E. X.Y", "1B_xxx/yyyy E. X.Y", ...],
  "estimated_total_count": <integer estimate of total citations expected>,
  "reasoning": "brief one-line explanation"
}
"""


def predict_citations(query, qid, retries=2):
    """Ask GPT-5.4 to predict full citation list."""
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Predict ALL citations for this Swiss Federal Supreme Court legal question:\n\n{query}"}
                ],
                temperature=0.2,
                max_completion_tokens=8000,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            usage = response.usage
            return result, usage
        except json.JSONDecodeError as e:
            print(f"  Retry {attempt+1}/{retries}: JSON error - {e}")
            if attempt == retries:
                return {"law_citations": [], "court_citations": [], "error": str(e)}, None
        except Exception as e:
            print(f"  Retry {attempt+1}/{retries}: {e}")
            if attempt == retries:
                return {"law_citations": [], "court_citations": [], "error": str(e)}, None
        time.sleep(1)


def process_dataset(csv_path, out_name, gold_available=False):
    """Process all queries in a dataset."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    results = {}
    total_cost = 0

    for row in rows:
        qid = row["query_id"]
        query = row["query"]
        print(f"\n{qid}: {query[:80]}...", flush=True)

        result, usage = predict_citations(query, qid)
        results[qid] = result

        n_law = len(result.get("law_citations", []))
        n_court = len(result.get("court_citations", []))
        est = result.get("estimated_total_count", "?")

        if usage:
            cost = usage.prompt_tokens / 1e6 * 2.50 + usage.completion_tokens / 1e6 * 15.0
            total_cost += cost
            print(f"  → {n_law} laws + {n_court} court = {n_law + n_court} total (est={est}) | cost=${cost:.4f}")
        else:
            print(f"  → ERROR: {result.get('error', 'unknown')}")

        time.sleep(0.3)

    # Save
    out_path = OUT / out_name
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
    print(f"Total cost: ${total_cost:.4f}")

    # Check recall if gold available
    if gold_available:
        print("\n=== Recall Check ===")
        for row in rows:
            qid = row["query_id"]
            gold = set(row["gold_citations"].split(";"))
            pred_all = set(results[qid].get("law_citations", [])) | set(results[qid].get("court_citations", []))
            found = gold & pred_all
            print(f"  {qid}: gold={len(gold)}, pred={len(pred_all)}, found={len(found)}, recall={len(found)/len(gold):.1%}")

    return results


def main():
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "val"

    if target in ("val", "both"):
        print("=== Processing VAL queries with GPT-5.4 ===")
        process_dataset(DATA / "val.csv", "val_full_citations_v2.json", gold_available=True)

    if target in ("test", "both"):
        print("\n=== Processing TEST queries with GPT-5.4 ===")
        process_dataset(DATA / "test.csv", "test_full_citations_v2.json", gold_available=False)


if __name__ == "__main__":
    main()
