#!/usr/bin/env python3
"""Generate citation predictions at multiple temperatures."""
import csv, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

BASE = Path(__file__).resolve().parent.parent
client = OpenAI(
    api_key=os.getenv('LLM_API_KEY', 'sk-8d5e67b45fe64f43914cfa82e3aab96c'),
    base_url=os.getenv('LLM_BASE_URL', 'https://api.deepseek.com/v1')
)
MODEL = os.getenv('FULL_CITATIONS_MODEL', 'deepseek-chat')

# Load system prompt from gen_full_citations_v2.py
with open(BASE / 'precompute/gen_full_citations_v2.py') as f:
    content = f.read()
start = content.find('SYSTEM_PROMPT = """') + len('SYSTEM_PROMPT = """')
end = content.find('"""', start)
SYSTEM_PROMPT = content[start:end]


def predict(query, qid, temperature):
    try:
        resp = client.chat.completions.create(
            model=MODEL, temperature=temperature,
            response_format={'type': 'json_object'}, max_tokens=8000,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f'Predict ALL citations for this Swiss Federal Supreme Court legal question:\n\n{query}'}
            ]
        )
        return qid, json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f'  ERROR {qid}: {e}')
        return qid, {'law_citations': [], 'court_citations': [], 'error': str(e)}


def run_temp(temperature, splits=('val', 'test')):
    tag = f't{int(temperature*10):02d}'
    print(f'\n=== Temperature {temperature} (tag={tag}) ===')

    for split in splits:
        out_path = BASE / f'precompute/{split}_full_citations_{tag}.json'
        results = json.load(open(out_path)) if out_path.exists() else {}

        rows = list(csv.DictReader(open(BASE / f'data/{split}.csv')))
        pending = [(r['query_id'], r['query']) for r in rows if r['query_id'] not in results]
        print(f'  {split}: {len(pending)} to process ({len(results)} cached)')

        if not pending:
            continue

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(predict, q, qid, temperature): qid for qid, q in pending}
            for f in as_completed(futs):
                qid, result = f.result()
                results[qid] = result
                n = len(result.get('law_citations', [])) + len(result.get('court_citations', []))
                print(f'    {qid}: {n} citations')

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'  Wrote {out_path.name} ({time.time()-t0:.1f}s)')


if __name__ == '__main__':
    for temp in [0.5, 0.7, 1.0]:
        run_temp(temp)
    print('\nDone. Analyze with scripts/analyze_multi_temp.py')
