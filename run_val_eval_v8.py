"""
V8 pipeline: V7 + dense court retrieval via OpenAI text-embedding-3-small.
Key changes from V7:
1. Add court FAISS index search (top 200 court candidates per query)
2. Score court dense hits as moderate confidence (0.55-0.65)
3. Co-citation expansion at 0.50
"""
import csv
import gc
import json
import os
import pickle
import re
import time
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE = Path(__file__).parent

# Court dense retrieval settings
USE_COURT_DENSE = True
COURT_TOP_K = 200
ENABLE_LIVE_RERANK = False  # disabled — hurt performance in V7

def tokenize(text):
    return [t for t in re.findall(r'[a-zäöüß]+', text.lower()) if len(t) > 1]


def build_fuzzy_index(law_set):
    """Build lookup structures for fuzzy matching GPT predictions to corpus."""
    # Index by statute abbreviation + article number
    # "Art. 221 Abs. 1 lit. b StPO" -> key: ("StPO", "221")
    statute_article_map = defaultdict(list)
    for cit in law_set:
        m = re.match(r'Art\.\s+(\d+[a-z]?)\b.*?([A-ZÄÖÜ][A-Za-zÄÖÜäöü]+)\s*$', cit)
        if m:
            art_num, statute = m.group(1), m.group(2)
            statute_article_map[(statute, art_num)].append(cit)
    return statute_article_map


def fuzzy_match_citation(cit, law_set, statute_article_map):
    """Try to find the closest corpus match for a GPT-predicted citation."""
    if cit in law_set:
        return cit  # Exact match

    # Parse the citation
    m = re.match(r'Art\.\s+(\d+[a-z]?)\b(.*?)([A-ZÄÖÜ][A-Za-zÄÖÜäöü]+)\s*$', cit)
    if not m:
        return None
    art_num, middle, statute = m.group(1), m.group(2).strip(), m.group(3)

    # Look up by statute + article number
    candidates = statute_article_map.get((statute, art_num), [])
    if not candidates:
        return None

    if len(candidates) == 1:
        # Only one candidate with same statute+article — likely a match
        return candidates[0]

    # Multiple candidates — find closest by edit distance
    best_match = None
    best_ratio = 0
    for candidate in candidates:
        ratio = SequenceMatcher(None, cit, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = candidate

    if best_ratio >= 0.75:
        return best_match

    return None


def fuzzy_match_court(cit, court_set, case_prefix_map):
    """Try to match GPT court prediction to corpus."""
    if cit in court_set:
        return cit

    # Extract base (without Erwägung)
    base_cit = re.sub(r'\s+E\.\s+.*$', '', cit).strip()

    # Try exact base match
    siblings = case_prefix_map.get(base_cit, [])
    if siblings:
        # Try to find the exact Erwägung
        for sib in siblings:
            if sib == cit:
                return sib
        # Return siblings as partial matches
        return siblings  # Return list of related citations

    # Try BGE format variations
    # "BGE 137 IV 122" → try as-is
    if base_cit in case_prefix_map:
        return case_prefix_map[base_cit]

    return None


def live_rerank_candidates(query, candidates_with_text, model="gpt-5.4"):
    """Call GPT-5.4 to review retrieval candidates. Returns set of approved citations."""
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Format candidates
    candidate_lines = []
    for i, (cit, text) in enumerate(candidates_with_text[:60]):  # Max 60 candidates
        candidate_lines.append(f"{i+1}. [{cit}]: {text[:200]}")

    candidates_text = "\n".join(candidate_lines)

    prompt = f"""Given this Swiss legal question, review each candidate citation below.
For each, respond with ONLY "Y" (relevant to this question) or "N" (not relevant).

QUESTION:
{query}

CANDIDATE CITATIONS:
{candidates_text}

Respond as a JSON object: {{"ratings": [{{"id": 1, "r": "Y"}}, ...]}}
Only include "Y" ratings to save tokens. List the IDs of relevant candidates.
Respond with: {{"relevant_ids": [1, 3, 5, ...]}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert Swiss legal researcher. Review citations for relevance to the given question."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_completion_tokens=2000,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        relevant_ids = set(result.get("relevant_ids", []))

        approved = set()
        for i, (cit, text) in enumerate(candidates_with_text[:60]):
            if (i + 1) in relevant_ids:
                approved.add(cit)
        return approved
    except Exception as e:
        print(f"    Rerank error: {e}")
        return set()


def main():
    t0 = time.time()

    print("Loading assets...", flush=True)
    expansions = json.loads((BASE / "precompute" / "val_query_expansions.json").read_text())
    case_citations = json.loads((BASE / "precompute" / "val_case_citations.json").read_text())
    # MERGE V1 (GPT-4.1) + V2 (GPT-5.4) + V3 (GPT-5.4 diverse) for maximum recall
    runs = []
    for name in ["val_full_citations.json", "val_full_citations_v2.json", "val_full_citations_v3.json"]:
        path = BASE / "precompute" / name
        if path.exists():
            runs.append((name, json.loads(path.read_text())))

    # Merge: union of citations from all runs, track frequency
    full_citations = {}
    all_qids = set()
    for _, data in runs:
        all_qids.update(data.keys())

    for qid in all_qids:
        law_freq = defaultdict(int)  # citation -> how many runs predicted it
        court_freq = defaultdict(int)
        for _, data in runs:
            d = data.get(qid, {})
            for c in d.get("law_citations", []):
                law_freq[c] += 1
            for c in d.get("court_citations", []):
                court_freq[c] += 1

        full_citations[qid] = {
            "law_citations": list(law_freq.keys()),
            "court_citations": list(court_freq.keys()),
            "law_freq": dict(law_freq),    # How many runs predicted each citation
            "court_freq": dict(court_freq),
        }

    n_total = sum(len(f["law_citations"]) + len(f["court_citations"]) for f in full_citations.values())
    print(f"  Merged {len(runs)} GPT runs → {n_total} total unique citations")

    # BM25
    with open(BASE / "index" / "bm25_laws.pkl", "rb") as f:
        law_data = pickle.load(f)
    law_bm25 = law_data["bm25"]
    law_cites_bm25 = law_data["citations"]
    law_set = set(law_cites_bm25)

    # Load law text for reranking
    law_text = {}
    if ENABLE_LIVE_RERANK:
        with open(BASE / "data" / "laws_de.csv") as f:
            for row in csv.DictReader(f):
                law_text[row["citation"]] = row.get("title", "") + " " + row["text"]

    # Build fuzzy matching index
    statute_article_map = build_fuzzy_index(law_set)

    # Court
    with open(BASE / "index" / "court_citations.pkl", "rb") as f:
        all_court_cites = pickle.load(f)
    court_set = set(all_court_cites)

    case_prefix_map = {}
    for cit in all_court_cites:
        base_cit = re.sub(r'\s+E\.\s+.*$', '', cit).strip()
        if base_cit not in case_prefix_map:
            case_prefix_map[base_cit] = []
        case_prefix_map[base_cit].append(cit)
    del all_court_cites
    gc.collect()

    # Dense
    print("Loading FAISS...", flush=True)
    faiss_index = faiss.read_index(str(BASE / "index" / "faiss_laws.index"))
    with open(BASE / "index" / "faiss_laws_citations.pkl", "rb") as f:
        faiss_law_cites = pickle.load(f)
    embed_model = SentenceTransformer("intfloat/multilingual-e5-large")

    # Court dense retrieval (V8)
    court_faiss = None
    court_faiss_cites = None
    if USE_COURT_DENSE:
        court_idx_path = BASE / "index" / "faiss_court_openai.index"
        court_cites_path = BASE / "index" / "faiss_court_openai_citations.pkl"
        if court_idx_path.exists():
            print("Loading court FAISS...", flush=True)
            court_faiss = faiss.read_index(str(court_idx_path))
            with open(court_cites_path, "rb") as f:
                court_faiss_cites = pickle.load(f)
            print(f"  Court index: {court_faiss.ntotal:,} citations")
        else:
            print("  Court FAISS not yet built — skipping court dense retrieval")

    # Val
    with open(BASE / "data" / "val.csv", "r") as f:
        val_rows = list(csv.DictReader(f))
    gold_map = {row["query_id"]: set(row["gold_citations"].split(";")) for row in val_rows}

    print(f"\nProcessing {len(val_rows)} queries (live_rerank={ENABLE_LIVE_RERANK})...\n", flush=True)
    predictions = {}

    for row in val_rows:
        qid = row["query_id"]
        query = row["query"]
        exp = expansions.get(qid, {})
        cases = case_citations.get(qid, {})
        full = full_citations.get(qid, {})

        scored = {}
        source_tracker = defaultdict(set)  # Track which sources contributed each citation

        # === 1. GPT full citation predictions (highest signal) ===
        law_freq = full.get("law_freq", {})   # How many runs predicted each
        court_freq = full.get("court_freq", {})

        for art in full.get("law_citations", []):
            freq = law_freq.get(art, 1)
            # Score scales with frequency: 1 run=0.85, 2 runs=0.90, 3 runs=0.95
            base_score = min(0.80 + freq * 0.05, 0.95)
            if art in law_set:
                scored[art] = base_score
                source_tracker[art].add("gpt_full")
            else:
                # FUZZY MATCH
                match = fuzzy_match_citation(art, law_set, statute_article_map)
                if match:
                    scored[match] = max(scored.get(match, 0), base_score - 0.05)
                    source_tracker[match].add("gpt_full_fuzzy")

        for cit in full.get("court_citations", []):
            freq = court_freq.get(cit, 1)
            base_score = min(0.78 + freq * 0.05, 0.93)
            if cit in court_set:
                scored[cit] = base_score
                source_tracker[cit].add("gpt_full")
            else:
                # Try co-citation matching
                result = fuzzy_match_court(cit, court_set, case_prefix_map)
                if isinstance(result, str):
                    scored[result] = max(scored.get(result, 0), 0.85)
                    source_tracker[result].add("gpt_full_fuzzy")
                elif isinstance(result, list):
                    for sib in result[:8]:
                        scored[sib] = max(scored.get(sib, 0), 0.70)
                        source_tracker[sib].add("gpt_court_sibling")

        # === 2. GPT specific articles ===
        for art in exp.get("specific_articles", []):
            if art in law_set:
                scored[art] = max(scored.get(art, 0), 0.92)
                source_tracker[art].add("gpt_specific")
            else:
                match = fuzzy_match_citation(art, law_set, statute_article_map)
                if match:
                    scored[match] = max(scored.get(match, 0), 0.88)
                    source_tracker[match].add("gpt_specific_fuzzy")

        # === 3. Explicit refs from query ===
        explicit = set(re.findall(
            r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+(?:\s+lit\.\s+[a-z])?)?\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü]+',
            query
        ))
        for art in explicit:
            if art in law_set:
                scored[art] = max(scored.get(art, 0), 0.95)
                source_tracker[art].add("explicit")
            else:
                match = fuzzy_match_citation(art, law_set, statute_article_map)
                if match:
                    scored[match] = max(scored.get(match, 0), 0.93)
                    source_tracker[match].add("explicit_fuzzy")

        # === 4. Dense retrieval top-200 (laws) ===
        q_emb = embed_model.encode([f"query: {query}"], normalize_embeddings=True)
        d_scores, d_indices = faiss_index.search(q_emb.astype(np.float32), 200)
        for rank, (score, idx) in enumerate(zip(d_scores[0], d_indices[0])):
            cit = faiss_law_cites[idx]
            norm = float(score) * 0.65
            if rank < 10:
                norm *= 1.3
            scored[cit] = max(scored.get(cit, 0), norm)
            source_tracker[cit].add("dense")

        # === 4b. Court dense retrieval (V8: use GERMAN keywords, not English query) ===
        if court_faiss is not None:
            # Use GPT-expanded German court search queries, not the English question
            court_queries = exp.get("bm25_queries_court", [])[:8]
            if not court_queries:
                # Fallback: use german_terms
                german = exp.get("german_terms", [])
                if german:
                    court_queries = [" ".join(german[:20])]

            if court_queries:
                try:
                    resp = oai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=court_queries,
                        dimensions=512,
                    )
                    # Search with each query, merge results with rank-based scoring
                    court_rank_best = {}  # cit -> best rank across queries
                    for emb in resp.data:
                        qv = np.array([emb.embedding], dtype=np.float32)
                        qv /= np.linalg.norm(qv, axis=1, keepdims=True)
                        c_scores, c_indices = court_faiss.search(qv, COURT_TOP_K)
                        for rank, (score, idx) in enumerate(zip(c_scores[0], c_indices[0])):
                            cit = court_faiss_cites[idx]
                            if cit not in court_rank_best or rank < court_rank_best[cit]:
                                court_rank_best[cit] = rank

                    # Score based on best rank
                    for cit, best_rank in court_rank_best.items():
                        # rank 0-20: high, 20-100: med, 100+: low
                        if best_rank < 20:
                            norm = 0.68
                        elif best_rank < 50:
                            norm = 0.60
                        elif best_rank < 100:
                            norm = 0.52
                        else:
                            norm = 0.42
                        scored[cit] = max(scored.get(cit, 0), norm)
                        source_tracker[cit].add("court_dense")
                except Exception as e:
                    print(f"    Court dense error: {e}")

        # === 5. BM25 ===
        bm25_hits = {}
        for bq in exp.get("bm25_queries_laws", []):
            tokens = tokenize(bq)
            if not tokens:
                continue
            scores_arr = law_bm25.get_scores(tokens)
            for idx in scores_arr.argsort()[-80:][::-1]:
                s = scores_arr[idx]
                if s > 0:
                    cit = law_cites_bm25[idx]
                    bm25_hits[cit] = max(bm25_hits.get(cit, 0), s)

        if exp.get("german_terms"):
            tokens = tokenize(" ".join(exp["german_terms"]))
            if tokens:
                scores_arr = law_bm25.get_scores(tokens)
                for idx in scores_arr.argsort()[-80:][::-1]:
                    s = scores_arr[idx]
                    if s > 0:
                        cit = law_cites_bm25[idx]
                        bm25_hits[cit] = max(bm25_hits.get(cit, 0), s)

        if bm25_hits:
            max_bm25 = max(bm25_hits.values())
            for cit, s in bm25_hits.items():
                norm = (s / max_bm25) * 0.65
                scored[cit] = max(scored.get(cit, 0), norm)
                source_tracker[cit].add("bm25")

        # === 6. GPT case citations (old) ===
        for cit in cases.get("expanded", []):
            if cit in court_set:
                scored[cit] = max(scored.get(cit, 0), 0.85)
                source_tracker[cit].add("gpt_case")

        # === 7. Co-citation expansion (V8: higher score, fewer siblings) ===
        gpt_court = list(set(cases.get("expanded", []) + full.get("court_citations", [])))
        seen = set()
        for cit in gpt_court:
            base_cit = re.sub(r'\s+E\.\s+.*$', '', cit).strip()
            if base_cit in seen:
                continue
            seen.add(base_cit)
            siblings = case_prefix_map.get(base_cit, [])
            # Give siblings enough score to survive the cutoff
            # but not so much they displace high-quality law predictions
            for sib in siblings[:3]:  # Fewer siblings, but higher score
                if sib not in scored:
                    scored[sib] = 0.50
                    source_tracker[sib].add("cocitation")

        # === 8. Multi-source agreement boost ===
        for cit in scored:
            n_sources = len(source_tracker[cit])
            if n_sources >= 2:
                scored[cit] = min(scored[cit] * 1.25, 0.96)
            if n_sources >= 3:
                scored[cit] = min(scored[cit] * 1.35, 0.98)

        # === 9. Always include Art. 100 Abs. 1 BGG ===
        if "Art. 100 Abs. 1 BGG" in law_set:
            scored["Art. 100 Abs. 1 BGG"] = max(scored.get("Art. 100 Abs. 1 BGG", 0), 0.80)

        # === 10. Optional live GPT reranking ===
        reranked = set()
        if ENABLE_LIVE_RERANK:
            # Get top retrieval candidates that aren't already high-confidence
            retrieval_only = [(cit, s) for cit, s in scored.items()
                             if cit in law_set and s < 0.80 and s > 0.20]
            retrieval_only.sort(key=lambda x: -x[1])
            candidates_for_review = [(cit, law_text.get(cit, "")) for cit, s in retrieval_only[:60]]
            if candidates_for_review:
                reranked = live_rerank_candidates(query, candidates_for_review)
                for cit in reranked:
                    scored[cit] = max(scored.get(cit, 0), 0.82)
                    source_tracker[cit].add("gpt_rerank")
                print(f"    Rerank: {len(reranked)} approved from {len(candidates_for_review)} candidates")

        # === 11. Filter to corpus-verified and select ===
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        verified = [(c, s) for c, s in ranked if c in law_set or c in court_set or c in explicit]

        # === 12. V8 HARD LAW/COURT CAPS ===
        gpt_estimate = exp.get("estimated_citation_count", 25)
        total_cap = min(max(gpt_estimate, 15), 32)  # 15-32 range

        # Hard caps: never exceed these regardless of scores
        law_cap = int(total_cap * 0.55)  # ~55% laws
        court_cap = total_cap - law_cap  # ~45% courts

        verified_laws = [(c, s) for c, s in verified if c in law_set]
        verified_courts = [(c, s) for c, s in verified if c in court_set]

        selected_laws = set(c for c, _ in verified_laws[:law_cap])
        selected_courts = set(c for c, _ in verified_courts[:court_cap])

        explicit_verified = {c for c in explicit if c in law_set or c in court_set}

        selected = selected_laws | selected_courts | explicit_verified
        predictions[qid] = selected

        # Stats
        gold = gold_map[qid]
        tp = len(selected & gold)
        prec = tp / len(selected) if selected else 0
        rec = tp / len(gold) if gold else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        # Count fuzzy matches that hit gold
        fuzzy_hits = sum(1 for c in selected & gold if "fuzzy" in str(source_tracker.get(c, set())))

        print(f"  {qid}: pred={len(selected)} (est={gpt_estimate}), gold={len(gold)}, TP={tp}, P={prec:.2f}, R={rec:.2f}, F1={f1:.2f} | fuzzy_hits={fuzzy_hits}")

    # Macro F1
    f1_scores = []
    for qid in predictions:
        gold = gold_map[qid]
        pred = predictions[qid]
        tp = len(pred & gold)
        prec = tp / len(pred) if pred else 0
        rec = tp / len(gold) if gold else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"\n=== V7 MACRO F1: {macro_f1:.4f} ({macro_f1*100:.2f}%) ===")
    print(f"Total time: {time.time()-t0:.0f}s")

    # Compare to V6
    print(f"V6 was: 20.80% | Diff: {(macro_f1 - 0.208)*100:+.2f}%")

    out_path = BASE / "submissions" / "val_pred_v7.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions.keys()):
            writer.writerow([qid, ";".join(predictions[qid])])
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
