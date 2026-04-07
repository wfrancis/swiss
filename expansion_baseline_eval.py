"""
Evaluate GPT-5.4 query expansion + BM25 on val set.
This is the key test: do GPT-generated German search queries find the right citations?
"""
import csv
import json
import pickle
import re
import sys
from pathlib import Path

BASE = Path(__file__).parent


def tokenize(text):
    return [t for t in re.findall(r'[a-zäöüß]+', text.lower()) if len(t) > 1]


def citation_f1(pred, gold):
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main():
    # Load law index
    print("Loading BM25 law index...")
    with open(BASE / "index" / "bm25_laws.pkl", "rb") as f:
        data = pickle.load(f)
    bm25 = data["bm25"]
    citations = data["citations"]

    # Load GPT-5.4 query expansions
    print("Loading GPT-5.4 query expansions...")
    expansions = json.loads((BASE / "precompute" / "val_query_expansions.json").read_text())
    print(f"  {len(expansions)} query expansions loaded")

    # Load val queries
    with open(BASE / "data" / "val.csv", "r") as f:
        val_rows = list(csv.DictReader(f))

    print(f"\n{'='*60}")
    print(f"GPT-5.4 Expansion + BM25 on Val Set (law-only)")
    print(f"{'='*60}")

    f1_scores = []
    for row in val_rows:
        qid = row["query_id"]
        query = row["query"]
        gold = {c.strip() for c in row["gold_citations"].split(";")}
        gold_laws = {c for c in gold if c.startswith("Art.")}

        exp = expansions.get(qid, {})
        predicted = set()

        # 1. Search with each BM25 law query
        for bm25_q in exp.get("bm25_queries_laws", []):
            tokens = tokenize(bm25_q)
            if not tokens:
                continue
            scores_arr = bm25.get_scores(tokens)
            top_idx = scores_arr.argsort()[-30:][::-1]
            for idx in top_idx:
                if scores_arr[idx] > 0:
                    predicted.add(citations[idx])

        # 2. Search with German terms
        if exp.get("german_terms"):
            tokens = tokenize(" ".join(exp["german_terms"]))
            if tokens:
                scores_arr = bm25.get_scores(tokens)
                top_idx = scores_arr.argsort()[-30:][::-1]
                for idx in top_idx:
                    if scores_arr[idx] > 0:
                        predicted.add(citations[idx])

        # 3. Add explicit/identified articles
        for art in exp.get("specific_articles", []):
            predicted.add(art)

        # 4. Extract explicit statute refs from query text
        explicit = set(re.findall(
            r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+(?:\s+lit\.\s+[a-z])?)?\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü]+',
            query
        ))
        predicted |= explicit

        # 5. Filter to only predictions that exist in corpus
        predicted = predicted & set(citations)

        f1_all = citation_f1(predicted, gold)
        f1_law = citation_f1(predicted, gold_laws) if gold_laws else None

        tp_all = len(predicted & gold)
        tp_law = len(predicted & gold_laws)
        print(f"\n{qid}: {len(gold)} gold ({len(gold_laws)} law)")
        print(f"  BM25 queries: {len(exp.get('bm25_queries_laws', []))}")
        print(f"  Specific articles: {exp.get('specific_articles', [])[:5]}")
        print(f"  Predicted: {len(predicted)} | All F1: {f1_all:.4f} (tp={tp_all})")
        if gold_laws:
            print(f"  Law F1: {f1_law:.4f} (tp={tp_law}/{len(gold_laws)})")
            if tp_law > 0:
                hits = predicted & gold_laws
                print(f"  Law hits: {sorted(list(hits))[:10]}")

        f1_scores.append(f1_all)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"\n{'='*60}")
    print(f"Macro F1 (all gold): {macro_f1:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
