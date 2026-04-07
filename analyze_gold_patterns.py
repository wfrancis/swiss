"""Analyze gold citation patterns to find boilerplate/procedural articles."""
import csv
from collections import Counter
from pathlib import Path

BASE = Path(__file__).parent

with open(BASE / "data" / "val.csv", "r") as f:
    val_rows = list(csv.DictReader(f))

# Count how often each citation appears across queries
citation_freq = Counter()
query_citations = {}
for row in val_rows:
    qid = row["query_id"]
    gold = row["gold_citations"].split(";")
    query_citations[qid] = gold
    for cit in gold:
        citation_freq[cit] += 1

print("=== Citations appearing in 3+ queries (boilerplate) ===")
boilerplate = []
for cit, count in citation_freq.most_common():
    if count >= 3:
        boilerplate.append(cit)
        print(f"  [{count}/10] {cit}")

print(f"\nTotal boilerplate: {len(boilerplate)} citations")
print(f"If we always predict these: {len(boilerplate)} predictions")

# Check how many gold they'd cover
total_covered = 0
total_gold = 0
for qid, gold in query_citations.items():
    covered = sum(1 for c in gold if c in boilerplate)
    total_covered += covered
    total_gold += len(gold)
    print(f"  {qid}: {covered}/{len(gold)} gold covered by boilerplate")

print(f"\nBoilerplate covers {total_covered}/{total_gold} = {total_covered/total_gold:.1%} of all gold")

# Analyze by statute
print("\n=== Most common statutes in gold ===")
import re
statute_freq = Counter()
for cit, count in citation_freq.items():
    # Extract statute name
    m = re.search(r'((?:Abs\.\s+\d+(?:\s+lit\.\s+[a-z])?\s+)?[A-ZÄÖÜ][A-Za-zÄÖÜäöü]+)$', cit)
    if m:
        statute_freq[m.group(1)] += count

for stat, count in statute_freq.most_common(20):
    print(f"  [{count}] {stat}")

# Analyze what types of articles are commonly missed
print("\n=== Per-query: what's found vs missed by category ===")
import json, pickle
expansions = json.loads((BASE / "precompute" / "val_query_expansions.json").read_text())
case_citations_data = json.loads((BASE / "precompute" / "val_case_citations.json").read_text())

with open(BASE / "index" / "bm25_laws.pkl", "rb") as f:
    law_data = pickle.load(f)
law_set = set(law_data["citations"])

for row in val_rows:
    qid = row["query_id"]
    gold = row["gold_citations"].split(";")
    gold_laws = [c for c in gold if c in law_set]
    gold_court = [c for c in gold if c not in law_set]

    # Categorize missed laws
    missed_laws = []
    for c in gold_laws:
        # Is this a procedural article?
        is_procedural = any(s in c for s in [
            "BGG", "StBOG", "422", "428", "429", "436", "135 Abs. 3", "135 Abs. 4",
            "390", "382", "385", "384", "393", "396", "100 Abs. 1"
        ])
        missed_laws.append((c, "procedural" if is_procedural else "substantive"))

    proc_count = sum(1 for _, t in missed_laws if t == "procedural")
    subst_count = sum(1 for _, t in missed_laws if t == "substantive")
    print(f"\n{qid}: {len(gold_laws)} gold laws ({proc_count} procedural, {subst_count} substantive), {len(gold_court)} court")
