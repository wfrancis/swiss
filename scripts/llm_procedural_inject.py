#!/usr/bin/env python3
"""
LLM-based procedural citation injection.

Instead of regex rules, ask DeepSeek to classify each query's proceeding type
and return ONLY the procedural citations it's confident about.

Usage:
    .venv/bin/python scripts/llm_procedural_inject.py \
        --baseline submissions/test_submission_baseline_public_best_30681.csv \
        --split test \
        --output submissions/test_submission_llm_procedural.csv
"""
import argparse, csv, json, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

BASE = Path(__file__).resolve().parent.parent

client = OpenAI(
    api_key='sk-8d5e67b45fe64f43914cfa82e3aab96c',
    base_url='https://api.deepseek.com/v1'
)

SYSTEM = """You are an expert on Swiss Federal Supreme Court (Bundesgericht) citation practice.

Given a legal question and the list of citations already predicted for it, identify which PROCEDURAL and AUXILIARY law articles are missing.

Swiss court decisions routinely cite certain provisions that are NOT topically related to the legal question but are ALWAYS cited in that type of proceeding. These include:

JURISDICTION & PROCEDURE:
- Art. 37/39 StBOG (Federal Criminal Court jurisdiction/procedure)
- Art. 78/80/81 BGG (criminal appeals to BGer)
- Art. 72/74/75 BGG (civil appeals)
- Art. 82/113 BGG (public law / constitutional complaints)
- Art. 393/396 StPO (criminal appeals)
- Art. 384/385/390 StPO (appeal deadlines & requirements)
- Art. 56/60 ATSG (social insurance appeals)

COSTS & LEGAL AID:
- Art. 422 StPO (criminal costs)
- Art. 436 StPO (appeal costs)
- Art. 135 Abs. 3-4 StPO (legal aid in criminal)
- Art. 66/68 BGG (BGer costs)

SENTENCING (only if criminal conviction is at issue):
- Art. 42/44/49 StGB (suspended sentence, probation, aggregate)
- Art. 12 Abs. 1 StGB (intent)
- Art. 25/26 StGB (aiding, special duty)

SUBSTANTIVE BOILERPLATE (only if directly relevant):
- Art. 285 Abs. 1 ZGB (child support criteria - ONLY for maintenance cases)
- Art. 467/505/519/520 ZGB (will validity - ONLY for inheritance cases)
- Art. 18 Abs. 1 OR (contract interpretation - ONLY for contract cases)
- Art. 16 ZGB (capacity - ONLY when capacity is at issue)

CRITICAL RULES:
1. Only suggest citations you are HIGHLY CONFIDENT belong in this specific proceeding type
2. Do NOT suggest substantive law unless the query is clearly about that topic
3. Check the existing predictions — if they already include Art. 221 StPO, this is likely a criminal detention case
4. If unsure about the proceeding type, return EMPTY list — false positives are worse than misses
5. Return ONLY citations in exact Swiss format: "Art. 42 Abs. 1 StGB"

Return JSON: {"proceeding_type": "criminal_appeal|civil_appeal|social_insurance|inheritance|family|contract|other", "confidence": 0.0-1.0, "citations": ["Art. ...", ...], "reasoning": "one sentence"}"""


def classify_query(qid, query, existing_preds):
    pred_list = sorted(existing_preds)[:30]  # send top 30 existing predictions as context
    user_msg = json.dumps({
        "query_id": qid,
        "query": query[:1500],
        "existing_predictions": pred_list,
    })

    resp = client.chat.completions.create(
        model='deepseek-chat',
        temperature=0.0,
        response_format={'type': 'json_object'},
        max_tokens=2000,
        messages=[
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': user_msg}
        ]
    )
    try:
        result = json.loads(resp.choices[0].message.content)
        return qid, result
    except:
        return qid, {"citations": [], "confidence": 0, "proceeding_type": "error"}


def load_preds(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        key = 'predicted_citations' if 'predicted_citations' in reader.fieldnames else 'citations'
        return {row['query_id']: set(c.strip() for c in row[key].split(';') if c.strip()) for row in reader}


def macro_f1(preds, gold):
    f1s = []
    for qid in gold:
        p = preds.get(qid, set())
        g = gold[qid]
        if not p and not g: f1s.append(1.0); continue
        if not p or not g: f1s.append(0.0); continue
        tp = len(p & g)
        pr = tp / len(p)
        rc = tp / len(g)
        f1s.append(2*pr*rc/(pr+rc) if pr+rc > 0 else 0.0)
    return sum(f1s)/len(f1s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--gold', type=Path, default=BASE / 'data/val.csv')
    parser.add_argument('--min-confidence', type=float, default=0.7)
    parser.add_argument('--cache', type=Path, default=BASE / 'precompute/llm_procedural_cache.json')
    args = parser.parse_args()

    predictions = load_preds(args.baseline)

    # Load queries
    queries = {}
    with open(BASE / 'data' / f'{args.split}.csv') as f:
        for row in csv.DictReader(f):
            queries[row['query_id']] = row['query']

    # Load cache
    cache = {}
    if args.cache.exists():
        cache = json.load(open(args.cache))

    # Classify all queries in parallel
    to_classify = []
    for qid in sorted(queries):
        cache_key = f"{args.split}_{qid}"
        if cache_key not in cache:
            to_classify.append(qid)

    print(f"Loaded {len(predictions)} predictions, {len(queries)} queries")
    print(f"Cache: {len(cache)} entries, {len(to_classify)} to classify")

    if to_classify:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=16) as ex:
            futures = {
                ex.submit(classify_query, qid, queries[qid], predictions.get(qid, set())): qid
                for qid in to_classify
            }
            for f in as_completed(futures):
                qid, result = f.result()
                cache_key = f"{args.split}_{qid}"
                cache[cache_key] = result
        elapsed = time.time() - t0
        print(f"Classified {len(to_classify)} queries in {elapsed:.1f}s")

        # Save cache
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        with open(args.cache, 'w') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    # Apply injections
    result = {}
    total_injected = 0
    for qid in sorted(predictions):
        preds = set(predictions[qid])
        cache_key = f"{args.split}_{qid}"
        classification = cache.get(cache_key, {})

        confidence = classification.get('confidence', 0)
        proc_type = classification.get('proceeding_type', 'unknown')
        new_cites = classification.get('citations', [])

        if confidence >= args.min_confidence and new_cites:
            injected = set()
            for c in new_cites:
                if c not in preds and c.startswith('Art.'):
                    injected.add(c)
            if injected:
                total_injected += len(injected)
                print(f"  {qid}: +{len(injected)} ({proc_type}, conf={confidence:.2f})")
                for c in sorted(injected):
                    print(f"    + {c}")
            preds = preds | injected

        result[qid] = preds

    print(f"\nTotal injected: {total_injected}")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id', 'predicted_citations'])
        for qid in sorted(result):
            writer.writerow([qid, ';'.join(sorted(result[qid]))])
    print(f"Wrote: {args.output}")

    # Evaluate on val
    if args.split == 'val' and args.gold.exists():
        gold = {}
        with open(args.gold) as f:
            for row in csv.DictReader(f):
                gold[row['query_id']] = set(row['gold_citations'].split(';'))
        bf1 = macro_f1(predictions, gold)
        rf1 = macro_f1(result, gold)
        print(f"\nBaseline val F1: {bf1:.6f}")
        print(f"Injected val F1: {rf1:.6f} ({(rf1 - bf1)*100:+.2f}pp)")


if __name__ == '__main__':
    main()
