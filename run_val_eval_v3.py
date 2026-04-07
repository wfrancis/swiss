"""
V3: Improved merging of all sources + better pruning.
Key changes:
- Keep more BM25 law candidates (lower threshold)
- Better merging of GPT articles + BM25 hits
- Co-citation expansion with smarter filtering
"""
import csv
import gc
import json
import pickle
import re
import time
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
    t0 = time.time()

    # Load all assets
    expansions = json.loads((BASE / "precompute" / "val_query_expansions.json").read_text())
    case_citations = json.loads((BASE / "precompute" / "val_case_citations.json").read_text())

    with open(BASE / "index" / "bm25_laws.pkl", "rb") as f:
        law_data = pickle.load(f)
    law_bm25 = law_data["bm25"]
    law_cites = law_data["citations"]
    law_set = set(law_cites)

    with open(BASE / "index" / "court_citations.pkl", "rb") as f:
        all_court_cites = pickle.load(f)
    court_set = set(all_court_cites)

    # Build case prefix index
    case_prefix_map = {}
    for cit in all_court_cites:
        base = re.sub(r'\s+E\.\s+.*$', '', cit).strip()
        if base not in case_prefix_map:
            case_prefix_map[base] = []
        case_prefix_map[base].append(cit)

    del all_court_cites
    gc.collect()

    with open(BASE / "data" / "val.csv", "r") as f:
        val_rows = list(csv.DictReader(f))

    print(f"V3 Pipeline: GPT + BM25 + co-citation expansion")
    print(f"{'='*60}\n")

    f1_scores = []
    predictions = {}

    for row in val_rows:
        qid = row["query_id"]
        query = row["query"]
        gold = {c.strip() for c in row["gold_citations"].split(";")}
        gold_laws = {c for c in gold if c.startswith("Art.")}
        gold_court = gold - gold_laws

        exp = expansions.get(qid, {})
        cases = case_citations.get(qid, {})

        scored = {}

        # === SOURCE 1: GPT specific articles (high confidence) ===
        for art in exp.get("specific_articles", []):
            scored[art] = 0.90

        # === SOURCE 2: Explicit refs from query ===
        explicit = set(re.findall(
            r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+(?:\s+lit\.\s+[a-z])?)?\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü]+',
            query
        ))
        for art in explicit:
            scored[art] = max(scored.get(art, 0), 0.95)

        # === SOURCE 3: BM25 law search — keep more candidates ===
        bm25_hits = {}
        for bq in exp.get("bm25_queries_laws", []):
            tokens = tokenize(bq)
            if not tokens:
                continue
            scores_arr = law_bm25.get_scores(tokens)
            top_idx = scores_arr.argsort()[-40:][::-1]
            for idx in top_idx:
                s = scores_arr[idx]
                if s > 0:
                    cit = law_cites[idx]
                    bm25_hits[cit] = max(bm25_hits.get(cit, 0), s)

        # Also search with German terms
        if exp.get("german_terms"):
            tokens = tokenize(" ".join(exp["german_terms"]))
            if tokens:
                scores_arr = law_bm25.get_scores(tokens)
                top_idx = scores_arr.argsort()[-40:][::-1]
                for idx in top_idx:
                    s = scores_arr[idx]
                    if s > 0:
                        cit = law_cites[idx]
                        bm25_hits[cit] = max(bm25_hits.get(cit, 0), s)

        # Normalize BM25 scores and add
        if bm25_hits:
            max_bm25 = max(bm25_hits.values())
            for cit, s in bm25_hits.items():
                norm = (s / max_bm25) * 0.70  # Cap at 0.70
                scored[cit] = max(scored.get(cit, 0), norm)

        # Boost: BM25 hit that matches a key statute from GPT
        key_statutes = exp.get("key_statutes", [])
        for cit in bm25_hits:
            for stat in key_statutes:
                if stat in cit:
                    scored[cit] = min(scored.get(cit, 0) * 1.3, 0.92)

        # === SOURCE 4: GPT case citations ===
        gpt_court = cases.get("expanded", [])
        for cit in gpt_court:
            if cit in court_set:
                scored[cit] = max(scored.get(cit, 0), 0.85)
            else:
                # Still include — might be a valid citation format
                scored[cit] = max(scored.get(cit, 0), 0.45)

        # === SOURCE 5: Co-citation expansion ===
        seen_cases = set()
        for cit in gpt_court:
            base = re.sub(r'\s+E\.\s+.*$', '', cit).strip()
            if base in seen_cases:
                continue
            seen_cases.add(base)
            siblings = case_prefix_map.get(base, [])
            parent_score = scored.get(cit, 0.5)
            for sib in siblings[:20]:  # Limit siblings
                if sib not in scored:
                    scored[sib] = parent_score * 0.35

        # === PRUNE ===
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        estimated = exp.get("estimated_citation_count", 25)

        # Filter to corpus-verified only (except explicit refs)
        verified_ranked = []
        for cit, s in ranked:
            if cit in law_set or cit in court_set or cit in explicit:
                verified_ranked.append((cit, s))

        # Dynamic cutoff with score gap detection
        if verified_ranked:
            max_score = verified_ranked[0][1]
            # Use a score threshold that adapts to the estimated count
            target = max(estimated, 15)
            cutoff = target

            # Extend if scores are still high
            for i in range(target, len(verified_ranked)):
                if verified_ranked[i][1] < max_score * 0.12:
                    cutoff = i
                    break
                cutoff = i + 1

            # But cap to avoid too much noise
            cutoff = min(cutoff, 55)
        else:
            cutoff = 0

        selected = set(cit for cit, _ in verified_ranked[:cutoff])
        predictions[qid] = selected

        # === EVALUATE ===
        f1 = citation_f1(selected, gold)
        tp = len(selected & gold)
        tp_law = len(selected & gold_laws)
        tp_court = len(selected & gold_court)

        print(f"{qid}: {len(gold)} gold ({len(gold_laws)}L/{len(gold_court)}C) | "
              f"Pred: {len(selected)} | F1: {f1:.4f} | "
              f"tp: {tp} ({tp_law}L/{tp_court}C)")

        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"\n{'='*60}")
    print(f"MACRO F1: {macro_f1:.4f} ({time.time()-t0:.0f}s)")
    print(f"{'='*60}")

    # Save
    Path(BASE / "submissions").mkdir(exist_ok=True)
    with open(BASE / "submissions" / "val_pred_v3.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions.keys()):
            writer.writerow([qid, ";".join(sorted(predictions[qid]))])


if __name__ == "__main__":
    main()
