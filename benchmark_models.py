"""
Benchmark GPT-5.4 vs GPT-5.4 Mini vs DeepSeek V3.2 vs Sonnet 4.6 on val queries.
Measures: recall, precision, F1, cost, latency per model.
"""
import csv
import json
import os
import time
import re
from pathlib import Path
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent
DATA = BASE / "data"

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Model configs
MODELS = {
    "gpt-5.4": {
        "type": "openai",
        "client": OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        "model": "gpt-5.4",
        "pricing": {"input": 2.50, "output": 15.00},
        "use_max_completion_tokens": True,
    },
    "gpt-5.4-mini": {
        "type": "openai",
        "client": OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        "model": "gpt-5.4-mini",
        "pricing": {"input": 0.75, "output": 4.50},
        "use_max_completion_tokens": True,
    },
    "deepseek-v3.2": {
        "type": "openai",
        "client": OpenAI(
            api_key=DEEPSEEK_KEY,
            base_url="https://api.deepseek.com/v1",
        ),
        "model": "deepseek-chat",
        "pricing": {"input": 0.14, "output": 0.28},
        "use_max_completion_tokens": False,
    },
    "sonnet-4.6": {
        "type": "anthropic",
        "client": anthropic.Anthropic(api_key=ANTHROPIC_KEY),
        "model": "claude-sonnet-4-6-20250514",
        "pricing": {"input": 3.00, "output": 15.00},
    },
}

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

Return ONLY a JSON object (no markdown, no code fences) containing:
- "law_citations": list of article citations (strings in exact format)
- "court_citations": list of BGE and unreported case citations (strings in exact format)
- "reasoning": brief one-sentence explanation (keep this SHORT to save tokens)

IMPORTANT: Keep your response concise. List citations only, with minimal reasoning.
"""


def run_query_openai(model_cfg, query):
    """Run a single query against an OpenAI-compatible API."""
    client = model_cfg["client"]
    model = model_cfg["model"]

    start = time.time()
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Predict ALL citations for this legal question:\n\n{query}"}
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }
        if model_cfg.get("use_max_completion_tokens"):
            kwargs["max_completion_tokens"] = 8000
        else:
            kwargs["max_tokens"] = 8000

        response = client.chat.completions.create(**kwargs)
        elapsed = time.time() - start

        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

        content = response.choices[0].message.content
        result = json.loads(content)
        return result, usage, elapsed

    except json.JSONDecodeError as e:
        elapsed = time.time() - start
        # Try to salvage truncated JSON
        content = response.choices[0].message.content
        result = salvage_json(content)
        if result:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
            return result, usage, elapsed
        print(f"    JSON ERROR: {e} (output len={len(content)})")
        return {"law_citations": [], "court_citations": [], "error": str(e)}, {"input_tokens": 0, "output_tokens": 0}, elapsed

    except Exception as e:
        elapsed = time.time() - start
        print(f"    ERROR: {e}")
        return {"law_citations": [], "court_citations": [], "error": str(e)}, {"input_tokens": 0, "output_tokens": 0}, elapsed


def run_query_anthropic(model_cfg, query):
    """Run a single query against Anthropic API."""
    client = model_cfg["client"]
    model = model_cfg["model"]

    start = time.time()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Predict ALL citations for this legal question. Return ONLY a JSON object, no markdown.\n\n{query}"}
            ],
            temperature=0.3,
        )
        elapsed = time.time() - start

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        content = response.content[0].text
        # Strip markdown code fences if present
        content = re.sub(r'^```json\s*', '', content.strip())
        content = re.sub(r'\s*```$', '', content.strip())
        result = json.loads(content)
        return result, usage, elapsed

    except json.JSONDecodeError as e:
        elapsed = time.time() - start
        content = response.content[0].text
        result = salvage_json(content)
        if result:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            return result, usage, elapsed
        print(f"    JSON ERROR: {e}")
        return {"law_citations": [], "court_citations": [], "error": str(e)}, {"input_tokens": 0, "output_tokens": 0}, elapsed

    except Exception as e:
        elapsed = time.time() - start
        print(f"    ERROR: {e}")
        return {"law_citations": [], "court_citations": [], "error": str(e)}, {"input_tokens": 0, "output_tokens": 0}, elapsed


def salvage_json(content):
    """Try to extract citations from truncated/malformed JSON."""
    try:
        # Try to extract law_citations array
        laws = []
        courts = []

        law_match = re.search(r'"law_citations"\s*:\s*\[([^\]]*)', content)
        if law_match:
            raw = law_match.group(1)
            laws = re.findall(r'"([^"]+)"', raw)

        court_match = re.search(r'"court_citations"\s*:\s*\[([^\]]*)', content)
        if court_match:
            raw = court_match.group(1)
            courts = re.findall(r'"([^"]+)"', raw)

        if laws or courts:
            return {"law_citations": laws, "court_citations": courts, "reasoning": "salvaged from truncated JSON"}

    except Exception:
        pass
    return None


def run_query(model_cfg, query):
    """Route to the right API."""
    if model_cfg.get("type") == "anthropic":
        return run_query_anthropic(model_cfg, query)
    else:
        return run_query_openai(model_cfg, query)


def compute_f1(predicted, gold):
    """Compute precision, recall, F1 for citation sets."""
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    tp = len(predicted & gold)
    prec = tp / len(predicted)
    rec = tp / len(gold)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1


def main():
    # Load val queries
    with open(DATA / "val.csv") as f:
        val_rows = list(csv.DictReader(f))

    # Load law corpus for corpus verification
    law_set = set()
    with open(DATA / "laws_de.csv") as f:
        for row in csv.DictReader(f):
            law_set.add(row["citation"])
    print(f"Loaded {len(law_set)} law citations for corpus check\n")

    results = {}

    for model_name, model_cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        model_results = []
        total_input = 0
        total_output = 0
        total_time = 0

        for row in val_rows:
            qid = row["query_id"]
            query = row["query"]
            gold = set(row["gold_citations"].split(";"))

            print(f"\n  {qid} ({len(gold)} gold)...", end="", flush=True)

            result, usage, elapsed = run_query(model_cfg, query)
            total_input += usage["input_tokens"]
            total_output += usage["output_tokens"]
            total_time += elapsed

            # Extract predictions
            pred_laws = set(result.get("law_citations", []))
            pred_courts = set(result.get("court_citations", []))
            pred_all = pred_laws | pred_courts

            # Raw match (exact string match)
            prec, rec, f1 = compute_f1(pred_all, gold)

            # Separate law vs court analysis
            gold_laws = {c for c in gold if not c.startswith("BGE") and not c[0].isdigit()}
            gold_courts = {c for c in gold if c.startswith("BGE") or (c[0].isdigit() and "/" in c)}

            law_found = len(pred_laws & gold)
            court_found = len(pred_courts & gold)

            # Corpus verification
            in_corpus = len(pred_laws & law_set)

            print(f" {elapsed:.1f}s | F1={f1:.2f} P={prec:.2f} R={rec:.2f} | pred={len(pred_all)} (L={len(pred_laws)} C={len(pred_courts)}) | found: L={law_found}/{len(gold_laws)} C={court_found}/{len(gold_courts)} | corpus={in_corpus}/{len(pred_laws)}")

            model_results.append({
                "qid": qid,
                "f1": f1, "precision": prec, "recall": rec,
                "n_pred": len(pred_all), "n_gold": len(gold),
                "n_pred_laws": len(pred_laws), "n_pred_courts": len(pred_courts),
                "law_found": law_found, "court_found": court_found,
                "in_corpus": in_corpus, "elapsed": elapsed,
            })

            time.sleep(0.5)

        # Summary
        avg_f1 = sum(r["f1"] for r in model_results) / len(model_results)
        avg_prec = sum(r["precision"] for r in model_results) / len(model_results)
        avg_rec = sum(r["recall"] for r in model_results) / len(model_results)
        avg_pred = sum(r["n_pred"] for r in model_results) / len(model_results)
        total_law_found = sum(r["law_found"] for r in model_results)
        total_court_found = sum(r["court_found"] for r in model_results)
        total_in_corpus = sum(r["in_corpus"] for r in model_results)
        total_pred_laws = sum(r["n_pred_laws"] for r in model_results)

        input_cost = total_input / 1_000_000 * model_cfg["pricing"]["input"]
        output_cost = total_output / 1_000_000 * model_cfg["pricing"]["output"]
        total_cost = input_cost + output_cost

        print(f"\n  --- {model_name} SUMMARY ---")
        print(f"  Macro F1: {avg_f1:.4f} ({avg_f1*100:.2f}%)")
        print(f"  Avg Precision: {avg_prec:.4f}")
        print(f"  Avg Recall: {avg_rec:.4f}")
        print(f"  Avg predictions: {avg_pred:.1f}")
        print(f"  Gold citations found: law={total_law_found}, court={total_court_found}")
        print(f"  Corpus hit rate: {total_in_corpus}/{total_pred_laws} ({total_in_corpus/max(total_pred_laws,1)*100:.1f}%)")
        print(f"  Tokens: {total_input:,} in + {total_output:,} out")
        print(f"  Cost: ${total_cost:.4f}")
        print(f"  Time: {total_time:.1f}s total, {total_time/len(val_rows):.1f}s/query")

        results[model_name] = {
            "macro_f1": avg_f1, "avg_precision": avg_prec, "avg_recall": avg_rec,
            "avg_predictions": avg_pred,
            "total_law_found": total_law_found, "total_court_found": total_court_found,
            "corpus_hit_rate": total_in_corpus / max(total_pred_laws, 1),
            "total_tokens_in": total_input, "total_tokens_out": total_output,
            "cost_usd": total_cost, "total_time_s": total_time,
            "per_query": model_results,
        }

    # Final comparison
    print(f"\n\n{'='*70}")
    print(f"FINAL COMPARISON (10 val queries, GPT-only citation prediction)")
    print(f"{'='*70}")
    print(f"{'Model':<18} {'F1':>8} {'Prec':>8} {'Rec':>8} {'AvgPred':>8} {'Cost':>8} {'Time':>8}")
    print("-" * 70)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["macro_f1"]):
        print(f"{name:<18} {r['macro_f1']*100:>7.2f}% {r['avg_precision']*100:>7.2f}% {r['avg_recall']*100:>7.2f}% {r['avg_predictions']:>8.1f} ${r['cost_usd']:>6.4f} {r['total_time_s']:>7.1f}s")

    # Save results
    out_path = BASE / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
