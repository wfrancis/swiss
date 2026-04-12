#!/usr/bin/env python3
"""
Agentic citation retriever: Claude Opus uses tools to SEARCH the law corpus
iteratively, like a real lawyer doing legal research.

Instead of dumping the whole corpus in context, the agent has:
- search_laws(query) -> top 20 matching law articles with text
- search_courts(query) -> top 20 matching court considerations with text
- lookup_statute(statute_code) -> all articles from that statute
- get_article_text(citation) -> full text of a specific article

The agent searches, reads, reasons, and builds its citation list step by step.
"""
import csv, json, time, re, sys, os
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
import numpy as np

BASE = Path(__file__).resolve().parent.parent

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_KEY:
    raise ValueError("ANTHROPIC_API_KEY env var required")

API_HEADERS = {
    "x-api-key": ANTHROPIC_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

# ── Build search indexes ──────────────────────────────────────

print("Building search indexes...", flush=True)

# Law corpus
LAW_TEXTS = {}
LAW_BY_STATUTE = {}
with open(BASE / "data/laws_de.csv") as f:
    for row in csv.DictReader(f):
        LAW_TEXTS[row["citation"]] = row["text"]
        parts = row["citation"].split()
        statute = parts[-1] if len(parts) > 1 else "OTHER"
        LAW_BY_STATUTE.setdefault(statute, []).append(row["citation"])

# Court corpus — just load citations, text on demand
COURT_TEXTS = {}
print("  Loading court corpus (this takes a minute)...", flush=True)
with open(BASE / "data/court_considerations.csv") as f:
    for row in csv.DictReader(f):
        COURT_TEXTS[row["citation"]] = row["text"]

# BM25 for law search
from rank_bm25 import BM25Okapi

def tokenize(text):
    return [t.lower() for t in re.findall(r'[a-zäöüß]+', text.lower()) if len(t) > 2]

law_cites_list = list(LAW_TEXTS.keys())
law_corpus = [tokenize(f"{cite} {LAW_TEXTS[cite]}") for cite in law_cites_list]
law_bm25 = BM25Okapi(law_corpus)

# BM25 for court search — sample 200K for speed
court_cites_list = list(COURT_TEXTS.keys())[:200000]
court_corpus = [tokenize(f"{cite} {COURT_TEXTS[cite][:200]}") for cite in court_cites_list]
court_bm25 = BM25Okapi(court_corpus)

print(f"  {len(LAW_TEXTS)} laws, {len(COURT_TEXTS)} courts indexed", flush=True)

# ── Tool implementations ──────────────────────────────────────

def search_laws(query, top_k=15):
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = law_bm25.get_scores(tokens)
    top_idx = scores.argsort()[-top_k:][::-1]
    results = []
    for idx in top_idx:
        if scores[idx] <= 0:
            break
        cite = law_cites_list[idx]
        results.append({"citation": cite, "text": LAW_TEXTS[cite][:300]})
    return results


def search_courts(query, top_k=15):
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = court_bm25.get_scores(tokens)
    top_idx = scores.argsort()[-top_k:][::-1]
    results = []
    for idx in top_idx:
        if scores[idx] <= 0:
            break
        cite = court_cites_list[idx]
        results.append({"citation": cite, "text": COURT_TEXTS[cite][:300]})
    return results


def lookup_statute(statute_code, max_articles=30):
    articles = LAW_BY_STATUTE.get(statute_code, [])
    return [{"citation": c, "text": LAW_TEXTS[c][:150]} for c in articles[:max_articles]]


def get_article_text(citation):
    text = LAW_TEXTS.get(citation, COURT_TEXTS.get(citation, ""))
    return {"citation": citation, "text": text[:500]} if text else {"citation": citation, "text": "NOT FOUND"}


TOOLS = [
    {
        "name": "search_laws",
        "description": "Search Swiss law articles by German keywords. Returns top matching articles with text.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "German search keywords"}},
            "required": ["query"]
        }
    },
    {
        "name": "search_courts",
        "description": "Search Swiss Federal Supreme Court considerations by German keywords. Returns matching BGE/case entries.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "German search keywords"}},
            "required": ["query"]
        }
    },
    {
        "name": "lookup_statute",
        "description": "List all articles from a specific Swiss statute (e.g., 'StPO', 'ZGB', 'OR', 'BGG').",
        "input_schema": {
            "type": "object",
            "properties": {"statute_code": {"type": "string", "description": "Statute abbreviation like StPO, ZGB, OR"}},
            "required": ["statute_code"]
        }
    },
    {
        "name": "get_article_text",
        "description": "Get the full text of a specific law article or court consideration.",
        "input_schema": {
            "type": "object",
            "properties": {"citation": {"type": "string", "description": "Full citation like 'Art. 221 Abs. 1 StPO'"}},
            "required": ["citation"]
        }
    },
    {
        "name": "submit_citations",
        "description": "Submit your final citation list. Call this when you have completed your research.",
        "input_schema": {
            "type": "object",
            "properties": {
                "law_citations": {"type": "array", "items": {"type": "string"}},
                "court_citations": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["law_citations", "court_citations"]
        }
    },
]

TOOL_FNS = {
    "search_laws": lambda inp: search_laws(inp["query"]),
    "search_courts": lambda inp: search_courts(inp["query"]),
    "lookup_statute": lambda inp: lookup_statute(inp["statute_code"]),
    "get_article_text": lambda inp: get_article_text(inp["citation"]),
}

SYSTEM = """You are a Swiss Federal Supreme Court clerk. Given a legal question, you must research and compile the COMPLETE list of citations (law articles + court decisions) that the court's judgment would include.

You have search tools to find relevant provisions. Use them systematically:

1. First, identify the key legal domains and statutes involved
2. Search for relevant law articles using German legal terms
3. Search for relevant court decisions (BGE and unreported cases)
4. Look up specific statutes to find procedural provisions
5. Include: substantive law, procedural provisions, constitutional rights, costs/legal aid

Be thorough — a typical judgment cites 15-45 provisions. Include:
- Core substantive articles
- Procedural articles (jurisdiction, standing, deadlines)
- Constitutional provisions (BV) if fundamental rights are at stake
- BGE leading cases
- Cost provisions (Art. 66/68 BGG or Art. 422 StPO)

When done researching, call submit_citations with your final list."""


def run_agent(qid, query, model="claude-opus-4-6", max_turns=20):
    """Run the agentic retrieval loop for one query."""
    messages = [{"role": "user", "content": f"Research and compile ALL citations for this Swiss legal question:\n\n{query[:2500]}"}]

    for turn in range(max_turns):
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers=API_HEADERS,
            timeout=120,
            json={
                "model": model,
                "max_tokens": 4096,
                "system": SYSTEM,
                "tools": TOOLS,
                "messages": messages,
            },
        )
        data = resp.json()

        if "content" not in data:
            print(f"    {qid} turn {turn}: API error: {json.dumps(data)[:200]}", flush=True)
            break

        content = data["content"]
        messages.append({"role": "assistant", "content": content})

        # Check stop reason
        if data.get("stop_reason") == "end_turn":
            # Agent finished without submitting — extract from text
            text_blocks = [b.get("text", "") for b in content if b.get("type") == "text"]
            full_text = " ".join(text_blocks)
            if "Art." in full_text or "BGE" in full_text:
                # Try to extract citations from final message
                laws = re.findall(r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+)?(?:\s+lit\.\s+[a-z])?\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü]+', full_text)
                courts = re.findall(r'BGE\s+\d+\s+[IV]+\s+\d+(?:\s+E\.\s*[\d.]+)?', full_text)
                if laws or courts:
                    return {"law_citations": list(set(laws)), "court_citations": list(set(courts))}

        # Check for tool use
        tool_uses = [b for b in content if b.get("type") == "tool_use"]
        if not tool_uses:
            break  # No more tool calls, agent is done

        # Process tool calls
        tool_results = []
        for tu in tool_uses:
            name = tu["name"]
            inp = tu["input"]

            if name == "submit_citations":
                # Final submission
                return inp

            fn = TOOL_FNS.get(name)
            if fn:
                result = fn(inp)
                result_str = json.dumps(result, ensure_ascii=False)[:3000]
            else:
                result_str = json.dumps({"error": f"Unknown tool: {name}"})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result_str,
            })
            print(f"    {qid} turn {turn}: {name}({json.dumps(inp)[:80]})", flush=True)

        # Nudge to submit if running long
        if turn >= 15:
            tool_results.append({"type": "text", "text": "You have done extensive research. Please call submit_citations now with your complete list."})
        messages.append({"role": "user", "content": tool_results})

    return {"law_citations": [], "court_citations": []}


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-6"
    splits = sys.argv[2:] if len(sys.argv) > 2 else ["val", "test"]

    for split in splits:
        print(f"\n=== {split.upper()} ({model}) ===", flush=True)

        queries = {}
        with open(BASE / f"data/{split}.csv") as f:
            for row in csv.DictReader(f):
                queries[row["query_id"]] = row["query"]

        out_path = BASE / f"precompute/{split}_agentic_{model.replace('-', '_')}.json"
        results = json.load(open(out_path)) if out_path.exists() else {}

        pending = [(qid, q) for qid, q in queries.items() if qid not in results]
        print(f"  {len(pending)} pending ({len(results)} cached)", flush=True)

        for qid, query in pending:
            print(f"\n  {qid}:", flush=True)
            t0 = time.time()
            result = run_agent(qid, query, model=model)
            elapsed = time.time() - t0
            n = len(result.get("law_citations", [])) + len(result.get("court_citations", []))
            print(f"  {qid}: {n} citations in {elapsed:.0f}s", flush=True)

            results[qid] = result
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nWrote {out_path}", flush=True)

        # Evaluate if gold available
        if split == "val":
            gold = {}
            with open(BASE / "data/val.csv") as f:
                for row in csv.DictReader(f):
                    gold[row["query_id"]] = set(row["gold_citations"].split(";"))

            corpus = set(LAW_TEXTS.keys()) | set(COURT_TEXTS.keys())

            total_found = 0
            total_gold = 0
            for qid in sorted(gold):
                r = results.get(qid, {})
                predicted = set(r.get("law_citations", []) + r.get("court_citations", []))
                in_corpus = {c for c in predicted if c in corpus}
                found = predicted & gold[qid]
                total_found += len(found)
                total_gold += len(gold[qid])
                print(f"  {qid}: pred={len(predicted)} corpus={len(in_corpus)} gold={len(found)}/{len(gold[qid])}", flush=True)

            print(f"\n  Gold recall: {total_found}/{total_gold} ({total_found/total_gold*100:.0f}%)", flush=True)


if __name__ == "__main__":
    main()
