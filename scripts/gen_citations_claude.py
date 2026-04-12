#!/usr/bin/env python3
"""Generate citation predictions using Claude models (Opus + Sonnet)."""
import csv, json, time, sys, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import httpx

BASE = Path(__file__).resolve().parent.parent

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_KEY:
    raise ValueError("ANTHROPIC_API_KEY env var required")

API_HEADERS = {
    "x-api-key": ANTHROPIC_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

SYSTEM = """You are an expert Swiss legal researcher with deep knowledge of Swiss federal law, cantonal law, and Federal Supreme Court (BGer/BGE) jurisprudence.

Your task: Given a legal question, predict the COMPLETE list of Swiss legal citations that a Federal Supreme Court decision answering this question would cite.

Swiss court decisions typically cite:
1. SUBSTANTIVE articles: the core legal provisions directly relevant to the issue
2. PROCEDURAL articles: jurisdiction, standing, appeal deadlines, costs, legal aid
3. CONSTITUTIONAL articles: fundamental rights if applicable (Art. 29 BV, Art. 9 BV, etc.)
4. LEADING CASES (BGE): landmark Federal Supreme Court decisions on the topic
5. RECENT CASES: recent unreported decisions (format: 1B_xxx/yyyy, 5A_xxx/yyyy, etc.)

CITATION FORMAT RULES:
- Laws: "Art. 221 Abs. 1 StPO", "Art. 100 Abs. 1 BGG"
- With subsections: "Art. 221 Abs. 1 lit. b StPO"
- BGE: "BGE 137 IV 122 E. 4.2"
- Unreported: "1B_210/2023 E. 4.1"

Return JSON with two arrays:
{"law_citations": ["Art. X Abs. Y Statute", ...], "court_citations": ["BGE X Y Z E. N", ...]}

Be comprehensive — include 15-45 citations total."""


def predict(model, qid, query):
    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers=API_HEADERS,
            timeout=300,
            json={
                "model": model,
                "max_tokens": 8000,
                "system": SYSTEM,
                "messages": [{"role": "user", "content": f"Predict ALL Swiss legal citations for this question:\n\n{query[:3000]}"}],
            },
        )
        data = resp.json()
        if "content" not in data:
            print(f"  API error {model} {qid}: {json.dumps(data)[:200]}", flush=True)
            return qid, {"law_citations": [], "court_citations": [], "error": str(data)}
        text = data["content"][0]["text"]
        # Strip markdown code blocks
        if "```json" in text:
            parts = text.split("```json")
            text = parts[1].split("```")[0].strip() if len(parts) > 1 else text
        elif "```" in text:
            parts = text.split("```")
            text = parts[1].strip() if len(parts) > 2 else text
        # Find JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            json_str = text[start:end + 1]
            result = json.loads(json_str)
        else:
            result = {"law_citations": [], "court_citations": []}
        return qid, result
    except Exception as e:
        import traceback
        print(f"  ERROR {model} {qid}: {e}", flush=True)
        traceback.print_exc()
        return qid, {"law_citations": [], "court_citations": [], "error": str(e)}


def run_model(model, model_tag, splits=("val", "test")):
    print(f"\n=== {model} ({model_tag}) ===", flush=True)
    for split in splits:
        out_path = BASE / f"precompute/{split}_full_citations_{model_tag}.json"
        results = json.load(open(out_path)) if out_path.exists() else {}

        rows = list(csv.DictReader(open(BASE / f"data/{split}.csv")))
        pending = [(r["query_id"], r["query"]) for r in rows if r["query_id"] not in results]
        print(f"  {split}: {len(pending)} pending ({len(results)} cached)", flush=True)

        if not pending:
            continue

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=11) as ex:
            futs = {ex.submit(predict, model, qid, q): qid for qid, q in pending}
            for f in as_completed(futs):
                qid, result = f.result()
                results[qid] = result
                n = len(result.get("law_citations", [])) + len(result.get("court_citations", []))
                print(f"    {qid}: {n} citations", flush=True)

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Wrote {out_path.name} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "claude-sonnet-4-6"
    tag = sys.argv[2] if len(sys.argv) > 2 else "claude_sonnet"
    run_model(model, tag)
