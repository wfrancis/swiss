"""
V2 evaluation: GPT articles + GPT cases + co-citation expansion + BM25 law search.
Skips slow court BM25 shards — GPT + co-citation is more effective.
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

    # Load all precomputed assets
    print("Loading assets...", flush=True)
    expansions = json.loads((BASE / "precompute" / "val_query_expansions.json").read_text())
    case_citations = json.loads((BASE / "precompute" / "val_case_citations.json").read_text())

    # Load law BM25 index
    with open(BASE / "index" / "bm25_laws.pkl", "rb") as f:
        law_data = pickle.load(f)
    law_bm25 = law_data["bm25"]
    law_cites = law_data["citations"]
    law_set = set(law_cites)

    # Load court citation index for co-citation expansion
    with open(BASE / "index" / "court_citations.pkl", "rb") as f:
        all_court_cites = pickle.load(f)
    court_set = set(all_court_cites)

    # Build court case prefix index for fast co-citation lookup
    print("Building case prefix index...", flush=True)
    case_prefix_map = {}
    for cit in all_court_cites:
        base = re.sub(r'\s+E\.\s+.*$', '', cit).strip()
        if base not in case_prefix_map:
            case_prefix_map[base] = []
        case_prefix_map[base].append(cit)
    print(f"  {len(case_prefix_map)} unique cases, {len(all_court_cites)} total considerations")

    del all_court_cites
    gc.collect()

    # Load val
    with open(BASE / "data" / "val.csv", "r") as f:
        val_rows = list(csv.DictReader(f))

    print(f"\n{'='*60}")
    print(f"V2 Pipeline: GPT articles + GPT cases + co-citation + BM25")
    print(f"{'='*60}")

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

        # === SOURCE 1: GPT-identified specific articles (highest confidence) ===
        for art in exp.get("specific_articles", []):
            if art in law_set:
                scored[art] = 0.95
            else:
                scored[art] = 0.85  # May still be valid, just not in our index

        # === SOURCE 2: Explicit statute refs from query text ===
        explicit = set(re.findall(
            r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+(?:\s+lit\.\s+[a-z])?)?\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü]+',
            query
        ))
        for art in explicit:
            if art in law_set:
                scored[art] = max(scored.get(art, 0), 0.98)

        # === SOURCE 3: BM25 law search with GPT German queries ===
        for bq in exp.get("bm25_queries_laws", []):
            tokens = tokenize(bq)
            if not tokens:
                continue
            scores_arr = law_bm25.get_scores(tokens)
            top_idx = scores_arr.argsort()[-20:][::-1]
            for idx in top_idx:
                if scores_arr[idx] > 0:
                    cit = law_cites[idx]
                    norm = min(scores_arr[idx] / 15.0, 1.0)
                    scored[cit] = max(scored.get(cit, 0), norm * 0.7)

        # BM25 with German terms
        if exp.get("german_terms"):
            tokens = tokenize(" ".join(exp["german_terms"]))
            if tokens:
                scores_arr = law_bm25.get_scores(tokens)
                top_idx = scores_arr.argsort()[-20:][::-1]
                for idx in top_idx:
                    if scores_arr[idx] > 0:
                        cit = law_cites[idx]
                        norm = min(scores_arr[idx] / 15.0, 1.0)
                        scored[cit] = max(scored.get(cit, 0), norm * 0.6)

        # === SOURCE 4: GPT-identified court cases ===
        gpt_court = cases.get("expanded", [])
        for cit in gpt_court:
            if cit in court_set:
                scored[cit] = max(scored.get(cit, 0), 0.85)
            else:
                scored[cit] = max(scored.get(cit, 0), 0.5)  # Not verified

        # === SOURCE 5: Co-citation expansion ===
        # For each GPT-identified case that exists, grab sibling Erwägungen
        seen_cases = set()
        for cit in gpt_court:
            base = re.sub(r'\s+E\.\s+.*$', '', cit).strip()
            if base in seen_cases:
                continue
            seen_cases.add(base)

            siblings = case_prefix_map.get(base, [])
            if siblings:
                # Score siblings based on parent score
                parent_score = scored.get(cit, 0.5)
                for sib in siblings:
                    if sib not in scored:
                        scored[sib] = parent_score * 0.4

        # === PRUNE ===
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        estimated = exp.get("estimated_citation_count", 25)

        # Dynamic cutoff: take estimated count, extend if scores stay high
        if ranked:
            max_score = ranked[0][1]
            threshold = max_score * 0.15
            cutoff = max(estimated, 15)
            for i in range(cutoff, len(ranked)):
                if ranked[i][1] < threshold:
                    cutoff = i
                    break
                cutoff = i + 1
            cutoff = min(cutoff, 55)
        else:
            cutoff = 0

        selected = set(cit for cit, _ in ranked[:cutoff])

        # Filter: only include citations that exist in corpus
        # (unless they're explicit from query text)
        verified = set()
        for cit in selected:
            if cit in law_set or cit in court_set or cit in explicit:
                verified.add(cit)
        selected = verified

        predictions[qid] = selected

        # === EVALUATE ===
        f1 = citation_f1(selected, gold)
        tp = len(selected & gold)
        tp_law = len(selected & gold_laws)
        tp_court = len(selected & gold_court)

        print(f"\n{qid}: {len(gold)} gold ({len(gold_laws)} law, {len(gold_court)} court)")
        print(f"  Predicted: {len(selected)} | F1: {f1:.4f}")
        print(f"  Law: {tp_law}/{len(gold_laws)} | Court: {tp_court}/{len(gold_court)} | tp: {tp}")
        if tp > 0:
            hits = sorted(selected & gold)[:10]
            print(f"  Hits: {hits}")

        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"\n{'='*60}")
    print(f"MACRO F1: {macro_f1:.4f}")
    print(f"Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")

    # Save submission
    Path(BASE / "submissions").mkdir(exist_ok=True)
    with open(BASE / "submissions" / "val_pred_v2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions.keys()):
            writer.writerow([qid, ";".join(sorted(predictions[qid]))])
    print(f"Saved to submissions/val_pred_v2.csv")


if __name__ == "__main__":
    main()
