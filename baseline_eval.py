"""
Quick baseline evaluation using BM25 on laws_de.csv only.
Tests retrieval quality before the full pipeline is built.
"""
import csv
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

    # Load val queries
    with open(BASE / "data" / "val.csv", "r") as f:
        val_rows = list(csv.DictReader(f))

    print(f"\n{'='*60}")
    print(f"BM25 Law-Only Baseline on Val Set ({len(val_rows)} queries)")
    print(f"{'='*60}")

    f1_scores = []
    for row in val_rows:
        qid = row["query_id"]
        query = row["query"]
        gold = {c.strip() for c in row["gold_citations"].split(";")}
        gold_laws = {c for c in gold if c.startswith("Art.")}
        gold_court = gold - gold_laws

        # Simple approach: tokenize the English query directly
        # (this is intentionally naive — shows the cross-lingual gap)
        tokens = tokenize(query)
        scores = bm25.get_scores(tokens)
        top_indices = scores.argsort()[-50:][::-1]
        predicted = set()
        for idx in top_indices:
            if scores[idx] > 0:
                predicted.add(citations[idx])

        # Also try extracting explicit statute refs from the query
        explicit = set(re.findall(
            r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+(?:\s+lit\.\s+[a-z])?)?\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü]+',
            query
        ))
        predicted |= explicit

        # Evaluate against all gold (unfair since we only search laws)
        f1_all = citation_f1(predicted, gold)
        # Evaluate against law-only gold (fair comparison)
        f1_law = citation_f1(predicted, gold_laws) if gold_laws else None

        tp_all = len(predicted & gold)
        tp_law = len(predicted & gold_laws)
        print(f"\n{qid}: {len(gold)} gold ({len(gold_laws)} law, {len(gold_court)} court)")
        print(f"  Predicted: {len(predicted)} citations")
        print(f"  All gold F1: {f1_all:.4f} (tp={tp_all})")
        if gold_laws:
            print(f"  Law-only F1: {f1_law:.4f} (tp={tp_law}/{len(gold_laws)})")
            if tp_law > 0:
                hits = predicted & gold_laws
                print(f"  Law hits: {list(hits)[:5]}...")

        f1_scores.append(f1_all)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"\n{'='*60}")
    print(f"Macro F1 (all gold): {macro_f1:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
