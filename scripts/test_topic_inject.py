#!/usr/bin/env python3
"""Test topic-specific LLM citation injection."""
import csv, json, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

BASE = Path(__file__).resolve().parent.parent
client = OpenAI(api_key='sk-8d5e67b45fe64f43914cfa82e3aab96c', base_url='https://api.deepseek.com/v1')

SYSTEM = (
    "You are a Swiss Federal Supreme Court clerk preparing a draft judgment.\n\n"
    "Given a legal question and the current predicted citations, identify 3-8 SPECIFIC "
    "Swiss law articles that are MISSING from the predictions but would CERTAINLY be "
    "cited in the judgment.\n\n"
    "Focus on:\n"
    "1. The CORE substantive provisions for this specific legal issue (not generic ones)\n"
    "2. Articles that DEFINE key legal concepts mentioned in the question\n"
    "3. Provisions that establish the TEST or STANDARD the court would apply\n"
    "4. Cross-references: if Art. X is cited, Art. Y that X explicitly refers to\n\n"
    "Do NOT suggest:\n"
    "- Generic procedural provisions (BGG, costs) unless specifically relevant\n"
    "- Articles you are not confident about\n"
    "- Broad constitutional provisions\n\n"
    "For each citation, explain in 5 words WHY it must be cited.\n\n"
    'Return JSON: {"missing_citations": [{"citation": "Art. 97 Abs. 1 OR", "why": "employer liability standard"}, ...]}\n\n'
    "Use EXACT Swiss citation format with Abs./lit. when you know the subsection.\n"
    "If unsure of the exact Abs., give the article without Abs."
)


def classify(qid, query, preds):
    resp = client.chat.completions.create(
        model='deepseek-chat', temperature=0.0,
        response_format={'type': 'json_object'}, max_tokens=2000,
        messages=[
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': json.dumps({
                'query': query[:2000],
                'current_predictions': sorted(preds)[:30]
            })}
        ]
    )
    try:
        return qid, json.loads(resp.choices[0].message.content)
    except:
        return qid, {'missing_citations': []}


def load_preds(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        key = 'predicted_citations' if 'predicted_citations' in reader.fieldnames else 'citations'
        return {row['query_id']: set(c.strip() for c in row[key].split(';') if c.strip()) for row in reader}


def macro_f1(preds, gold):
    f1s = []
    for qid in gold:
        p = preds.get(qid, set()); g = gold[qid]
        if not p and not g: f1s.append(1.0); continue
        if not p or not g: f1s.append(0.0); continue
        tp = len(p & g); pr = tp / len(p); rc = tp / len(g)
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0)
    return sum(f1s) / len(f1s)


def main():
    # Load val
    gold, queries = {}, {}
    with open(BASE / 'data/val.csv') as f:
        for row in csv.DictReader(f):
            gold[row['query_id']] = set(row['gold_citations'].split(';'))
            queries[row['query_id']] = row['query']

    current = load_preds(BASE / 'submissions/val_pred_llm_proc_nobgg.csv')
    bf1 = macro_f1(current, gold)

    # Run on val
    print(f"Running topic-specific injection on {len(queries)} val queries...")
    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(classify, qid, queries[qid], current[qid]): qid for qid in queries}
        for f in as_completed(futures):
            qid, result = f.result()
            results[qid] = result
    print(f"Done in {time.time() - t0:.1f}s\n")

    # Apply and evaluate
    injected_preds = {}
    total_inj = 0
    total_correct = 0
    for qid in sorted(results):
        preds = set(current[qid])
        r = results[qid]
        missing = r.get('missing_citations', [])
        new_cites = set()
        for m in missing:
            c = m.get('citation', '')
            if c.startswith('Art.') and c not in preds:
                new_cites.add(c)

        correct = new_cites & gold[qid]
        total_inj += len(new_cites)
        total_correct += len(correct)

        if new_cites:
            print(f"{qid}: +{len(new_cites)} suggested, {len(correct)} correct")
            for m in missing:
                c = m['citation']
                if c in preds:
                    continue
                is_gold = "HIT" if c in gold[qid] else "   "
                print(f"  {is_gold} {c:35s} ({m.get('why', '')})")

        injected_preds[qid] = preds | new_cites

    rf1 = macro_f1(injected_preds, gold)
    prec = total_correct / total_inj * 100 if total_inj else 0
    print(f"\nTotal: {total_inj} injected, {total_correct} correct ({prec:.0f}% precision)")
    print(f"Baseline val F1: {bf1:.6f}")
    print(f"Injected val F1: {rf1:.6f} ({(rf1 - bf1) * 100:+.2f}pp)")

    # Save cache
    with open(BASE / 'precompute/llm_topicspecific_cache_val.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Also run on test
    print("\n\nRunning on test...")
    test_queries = {}
    with open(BASE / 'data/test.csv') as f:
        for row in csv.DictReader(f):
            test_queries[row['query_id']] = row['query']

    test_current = load_preds(BASE / 'submissions/test_submission_llm_proc_nobgg.csv')

    t0 = time.time()
    test_results = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(classify, qid, test_queries[qid], test_current[qid]): qid for qid in test_queries}
        for f in as_completed(futures):
            qid, result = f.result()
            test_results[qid] = result
    print(f"Done in {time.time() - t0:.1f}s")

    # Write test output
    test_injected = {}
    test_total = 0
    for qid in sorted(test_results):
        preds = set(test_current[qid])
        r = test_results[qid]
        for m in r.get('missing_citations', []):
            c = m.get('citation', '')
            if c.startswith('Art.') and c not in preds:
                preds.add(c)
                test_total += 1
        test_injected[qid] = preds

    print(f"Test: {test_total} citations injected")

    with open(BASE / 'submissions/val_pred_topic_inject.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['query_id', 'predicted_citations'])
        for qid in sorted(injected_preds):
            w.writerow([qid, ';'.join(sorted(injected_preds[qid]))])

    with open(BASE / 'submissions/test_submission_topic_inject.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['query_id', 'predicted_citations'])
        for qid in sorted(test_injected):
            w.writerow([qid, ';'.join(sorted(test_injected[qid]))])

    with open(BASE / 'precompute/llm_topicspecific_cache_test.json', 'w') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)

    print("\nWrote val_pred_topic_inject.csv + test_submission_topic_inject.csv")


if __name__ == '__main__':
    main()
