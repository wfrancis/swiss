#!/usr/bin/env python3
"""
Search the full 2.4M court corpus for query-relevant considerations,
then extract the citations mentioned in those considerations.

This is retrieval-augmented citation: find similar REAL court decisions,
copy their citation patterns.
"""
import csv, re, json, time, sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def tokenize(text):
    return [t.lower() for t in re.findall(r'[a-zäöüß]+', text.lower()) if len(t) > 2]


def build_bm25_index(corpus_rows):
    """Build a simple BM25 index over corpus texts."""
    from math import log

    N = len(corpus_rows)
    df = Counter()  # document frequency
    tf = []  # term frequency per doc
    doc_lens = []

    print(f"  Indexing {N:,} documents...", flush=True)
    for i, (cite, text) in enumerate(corpus_rows):
        tokens = tokenize(text)
        doc_lens.append(len(tokens))
        term_counts = Counter(tokens)
        tf.append(term_counts)
        for t in term_counts:
            df[t] += 1
        if (i + 1) % 500000 == 0:
            print(f"    {i+1:,}/{N:,}", flush=True)

    avgdl = sum(doc_lens) / N if N else 1
    return df, tf, doc_lens, avgdl, N


def bm25_search(query_tokens, df, tf, doc_lens, avgdl, N, k1=1.5, b=0.75, top_k=50):
    """Score all documents against query."""
    from math import log
    scores = []
    for i in range(len(tf)):
        score = 0
        dl = doc_lens[i]
        for t in query_tokens:
            if t not in tf[i]:
                continue
            tfi = tf[i][t]
            dfi = df.get(t, 0)
            idf = log((N - dfi + 0.5) / (dfi + 0.5) + 1)
            tf_norm = (tfi * (k1 + 1)) / (tfi + k1 * (1 - b + b * dl / avgdl))
            score += idf * tf_norm
        if score > 0:
            scores.append((score, i))

    scores.sort(reverse=True)
    return scores[:top_k]


# Citation extraction patterns
BGE_PAT = re.compile(r'BGE\s+\d+\s+[IV]+\s+\d+(?:\s+E\.\s*[\d.]+)?')
CASE_PAT = re.compile(r'\d[A-Z]_\d+/\d{4}(?:\s+E\.\s*[\d.]+)?')
LAW_PAT = re.compile(r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+)?(?:\s+lit\.\s+[a-z])?(?:\s+(?:Ziff|Bst)\.\s+\d+)?\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü]+')


def extract_citations(text):
    """Extract all citations mentioned in a court consideration text."""
    cites = set()
    for m in BGE_PAT.finditer(text):
        cites.add(m.group().strip())
    for m in CASE_PAT.finditer(text):
        cites.add(m.group().strip())
    for m in LAW_PAT.finditer(text):
        cites.add(m.group().strip())
    return cites


def main():
    print("Loading court corpus...", flush=True)
    t0 = time.time()
    corpus = []
    with open(BASE / 'data/court_considerations.csv') as f:
        for row in csv.DictReader(f):
            corpus.append((row['citation'], row['text']))
    print(f"  {len(corpus):,} rows in {time.time()-t0:.0f}s", flush=True)

    # Build German keyword queries from the English test queries using DeepSeek
    # Actually, we already have German BM25 queries in query_expansions!
    expansions = {}
    for split in ['val', 'test']:
        path = BASE / f'precompute/{split}_query_expansions.json'
        if path.exists():
            expansions.update(json.load(open(path)))

    # Build BM25 index
    print("\nBuilding BM25 index...", flush=True)
    t0 = time.time()
    df, tf, doc_lens, avgdl, N = build_bm25_index(corpus)
    print(f"  Index built in {time.time()-t0:.0f}s", flush=True)

    # Search for each query
    queries = {}
    for split in ['val', 'test']:
        with open(BASE / f'data/{split}.csv') as f:
            for row in csv.DictReader(f):
                queries[row['query_id']] = row['query']

    print(f"\nSearching {len(queries)} queries...", flush=True)
    results = {}

    for qid in sorted(queries):
        # Get German search terms from expansions
        exp = expansions.get(qid, {})
        german_terms = exp.get('german_terms', [])
        bm25_queries = exp.get('bm25_queries_laws', []) + exp.get('bm25_queries_court', [])

        # Combine all German terms
        search_text = ' '.join(german_terms + bm25_queries)
        if not search_text:
            search_text = queries[qid]  # fallback to English

        tokens = tokenize(search_text)
        hits = bm25_search(tokens, df, tf, doc_lens, avgdl, N, top_k=100)

        # Extract citations from top hits
        cite_votes = Counter()
        for score, idx in hits:
            cite, text = corpus[idx]
            found_cites = extract_citations(text)
            for c in found_cites:
                cite_votes[c] += 1

        results[qid] = {
            'n_hits': len(hits),
            'top_score': hits[0][0] if hits else 0,
            'citations': dict(cite_votes.most_common(100)),
        }

        top_cites = cite_votes.most_common(5)
        print(f"  {qid}: {len(hits)} hits, {len(cite_votes)} unique cites. Top: {[c for c,_ in top_cites[:3]]}", flush=True)

    # Save results
    out_path = BASE / 'precompute/court_corpus_search_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}", flush=True)

    # Evaluate on val
    gold = {}
    with open(BASE / 'data/val.csv') as f:
        for row in csv.DictReader(f):
            gold[row['query_id']] = set(row['gold_citations'].split(';'))

    print("\n=== VAL RECALL FROM CORPUS SEARCH ===", flush=True)
    for min_votes in [1, 2, 3, 5]:
        total_found = 0
        total_gold = 0
        total_candidates = 0
        for qid in gold:
            r = results.get(qid, {})
            candidates = {c for c, v in r.get('citations', {}).items() if v >= min_votes}
            found = candidates & gold[qid]
            total_found += len(found)
            total_gold += len(gold[qid])
            total_candidates += len(candidates)
        recall = total_found / total_gold * 100 if total_gold else 0
        avg_cands = total_candidates / len(gold)
        print(f"  min_votes={min_votes}: recall={recall:.1f}% ({total_found}/{total_gold}) avg_candidates={avg_cands:.0f}", flush=True)


if __name__ == '__main__':
    main()
