#!/usr/bin/env python3
"""Test DeepSeek as direct citation retriever — all queries in parallel."""
import json, csv, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

BASE = Path(__file__).resolve().parent.parent
client = OpenAI(api_key='sk-8d5e67b45fe64f43914cfa82e3aab96c', base_url='https://api.deepseek.com/v1')

SYSTEM = """You are a Swiss Federal Supreme Court legal expert.
Given a legal question, predict ALL Swiss legal citations (law articles AND court decisions)
that a Federal Supreme Court decision answering this question would cite.

Return JSON: {"citations": ["Art. 221 Abs. 1 StPO", "BGE 137 IV 122 E. 4.2", ...]}

Be comprehensive - include substantive law, procedural law, constitutional provisions,
BGE leading cases, and recent unreported decisions. A typical decision cites 15-45 citations.
Use exact Swiss citation format."""

def query_llm(qid, query):
    resp = client.chat.completions.create(
        model='deepseek-chat', temperature=0.0,
        response_format={'type': 'json_object'}, max_tokens=4000,
        messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': query[:2000]}]
    )
    try:
        return qid, set(json.loads(resp.choices[0].message.content).get('citations', []))
    except:
        return qid, set()

# Load gold + queries
gold, queries = {}, {}
with open(BASE / 'data/val.csv') as f:
    for row in csv.DictReader(f):
        gold[row['query_id']] = set(row['gold_citations'].split(';'))
        queries[row['query_id']] = row['query']

# Load candidate pool
bundles = json.load(open(BASE / 'artifacts/v11/val_v11_strict_v1/judged_bundles.json'))
pool_cites = {b['query_id']: {c['citation'] for c in b['candidates']} for b in bundles['bundles']}

# Fire ALL 10 queries in parallel
t0 = time.time()
results = {}
with ThreadPoolExecutor(max_workers=10) as ex:
    futures = {ex.submit(query_llm, qid, q): qid for qid, q in queries.items()}
    for f in as_completed(futures):
        qid, cites = f.result()
        results[qid] = cites

elapsed = time.time() - t0
print(f'All 10 queries done in {elapsed:.1f}s\n')

total_invisible = total_llm_finds = total_gold = total_llm_correct = 0
for qid in sorted(queries):
    gold_set = gold[qid]
    pool = pool_cites.get(qid, set())
    invisible = gold_set - pool
    llm_cites = results[qid]
    llm_correct = llm_cites & gold_set
    llm_finds_invisible = llm_cites & invisible

    total_invisible += len(invisible)
    total_llm_finds += len(llm_finds_invisible)
    total_gold += len(gold_set)
    total_llm_correct += len(llm_correct)

    print(f'{qid}: gold={len(gold_set)} invisible={len(invisible)} '
          f'LLM_pred={len(llm_cites)} LLM_correct={len(llm_correct)} '
          f'LLM_recovers_invisible={len(llm_finds_invisible)}')
    for c in sorted(llm_finds_invisible):
        print(f'  RECOVERED: {c}')

print(f'\n=== SUMMARY ===')
print(f'Total gold: {total_gold}')
print(f'Invisible (not in any retrieval pool): {total_invisible}')
print(f'LLM finds gold: {total_llm_correct}/{total_gold} = {total_llm_correct/total_gold*100:.1f}%')
print(f'LLM recovers invisible: {total_llm_finds}/{total_invisible} = {total_llm_finds/total_invisible*100:.1f}%')
