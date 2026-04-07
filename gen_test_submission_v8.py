"""
V8 test submission generator.
V7 + court dense retrieval via OpenAI embeddings.
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
USE_COURT_DENSE = True
COURT_TOP_K = 200


def tokenize(text):
    return [t for t in re.findall(r'[a-zäöüß]+', text.lower()) if len(t) > 1]


def build_fuzzy_index(law_set):
    statute_article_map = defaultdict(list)
    for cit in law_set:
        m = re.match(r'Art\.\s+(\d+[a-z]?)\b.*?([A-ZÄÖÜ][A-Za-zÄÖÜäöü]+)\s*$', cit)
        if m:
            art_num, statute = m.group(1), m.group(2)
            statute_article_map[(statute, art_num)].append(cit)
    return statute_article_map


def fuzzy_match_citation(cit, law_set, statute_article_map):
    if cit in law_set:
        return cit
    m = re.match(r'Art\.\s+(\d+[a-z]?)\b(.*?)([A-ZÄÖÜ][A-Za-zÄÖÜäöü]+)\s*$', cit)
    if not m:
        return None
    art_num, middle, statute = m.group(1), m.group(2).strip(), m.group(3)
    candidates = statute_article_map.get((statute, art_num), [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
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


def main():
    t0 = time.time()
    print("Loading assets...", flush=True)

    # Load test expansions
    expansions = json.loads((BASE / "precompute" / "test_query_expansions.json").read_text())
    case_citations = json.loads((BASE / "precompute" / "test_case_citations.json").read_text())

    # MERGE all available test full citation runs
    runs = []
    for name in ["test_full_citations.json", "test_full_citations_v2.json", "test_full_citations_v3.json"]:
        path = BASE / "precompute" / name
        if path.exists():
            runs.append((name, json.loads(path.read_text())))
            print(f"  Loaded {name}")

    full_citations = {}
    all_qids = set()
    for _, data in runs:
        all_qids.update(data.keys())

    for qid in all_qids:
        law_freq = defaultdict(int)
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
            "law_freq": dict(law_freq),
            "court_freq": dict(court_freq),
        }
    n_total = sum(len(f["law_citations"]) + len(f["court_citations"]) for f in full_citations.values())
    print(f"  Merged {len(runs)} runs → {n_total} total unique citations")

    # BM25
    with open(BASE / "index" / "bm25_laws.pkl", "rb") as f:
        law_data = pickle.load(f)
    law_bm25 = law_data["bm25"]
    law_cites_bm25 = law_data["citations"]
    law_set = set(law_cites_bm25)

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

    # Court dense (V8)
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

    # Test queries
    with open(BASE / "data" / "test.csv", "r") as f:
        test_rows = list(csv.DictReader(f))

    print(f"\nProcessing {len(test_rows)} test queries...\n", flush=True)
    predictions = {}

    for row in test_rows:
        qid = row["query_id"]
        query = row["query"]
        exp = expansions.get(qid, {})
        cases = case_citations.get(qid, {})
        full = full_citations.get(qid, {})

        scored = {}
        source_tracker = defaultdict(set)

        # 1. GPT full citations with frequency-based scoring
        law_freq = full.get("law_freq", {})
        court_freq = full.get("court_freq", {})

        for art in full.get("law_citations", []):
            freq = law_freq.get(art, 1)
            base_score = min(0.80 + freq * 0.05, 0.95)
            if art in law_set:
                scored[art] = base_score
                source_tracker[art].add("gpt_full")
            else:
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
                base_cit = re.sub(r'\s+E\.\s+.*$', '', cit).strip()
                siblings = case_prefix_map.get(base_cit, [])
                for sib in siblings[:8]:
                    scored[sib] = max(scored.get(sib, 0), 0.70)
                    source_tracker[sib].add("gpt_court_sibling")

        # 2. GPT specific articles
        for art in exp.get("specific_articles", []):
            if art in law_set:
                scored[art] = max(scored.get(art, 0), 0.92)
                source_tracker[art].add("gpt_specific")
            else:
                match = fuzzy_match_citation(art, law_set, statute_article_map)
                if match:
                    scored[match] = max(scored.get(match, 0), 0.88)
                    source_tracker[match].add("gpt_specific_fuzzy")

        # 3. Explicit refs from query
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

        # 4. Dense retrieval (laws)
        q_emb = embed_model.encode([f"query: {query}"], normalize_embeddings=True)
        d_scores, d_indices = faiss_index.search(q_emb.astype(np.float32), 200)
        for rank, (score, idx) in enumerate(zip(d_scores[0], d_indices[0])):
            cit = faiss_law_cites[idx]
            norm = float(score) * 0.65
            if rank < 10:
                norm *= 1.3
            scored[cit] = max(scored.get(cit, 0), norm)
            source_tracker[cit].add("dense")

        # 4b. Court dense retrieval (V8)
        if court_faiss is not None:
            try:
                resp = oai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[query],
                    dimensions=512,
                )
                q_court = np.array([resp.data[0].embedding], dtype=np.float32)
                q_court /= np.linalg.norm(q_court, axis=1, keepdims=True)

                c_scores, c_indices = court_faiss.search(q_court, COURT_TOP_K)
                for rank, (score, idx) in enumerate(zip(c_scores[0], c_indices[0])):
                    cit = court_faiss_cites[idx]
                    norm = 0.40 + float(score) * 0.30
                    if rank < 20:
                        norm += 0.05
                    scored[cit] = max(scored.get(cit, 0), norm)
                    source_tracker[cit].add("court_dense")
            except Exception as e:
                print(f"    Court dense error: {e}")

        # 5. BM25
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

        # 6. GPT case citations
        for cit in cases.get("expanded", []):
            if cit in court_set:
                scored[cit] = max(scored.get(cit, 0), 0.85)
                source_tracker[cit].add("gpt_case")

        # 7. Co-citation expansion
        gpt_court = list(set(cases.get("expanded", []) + full.get("court_citations", [])))
        seen = set()
        for cit in gpt_court:
            base_cit = re.sub(r'\s+E\.\s+.*$', '', cit).strip()
            if base_cit in seen:
                continue
            seen.add(base_cit)
            siblings = case_prefix_map.get(base_cit, [])
            parent_score = scored.get(cit, 0.5)
            for sib in siblings[:15]:
                if sib not in scored:
                    scored[sib] = parent_score * 0.30
                    source_tracker[sib].add("cocitation")

        # 8. Multi-source agreement
        for cit in scored:
            n_sources = len(source_tracker[cit])
            if n_sources >= 2:
                scored[cit] = min(scored[cit] * 1.25, 0.96)
            if n_sources >= 3:
                scored[cit] = min(scored[cit] * 1.35, 0.98)

        # 9. Art. 100 Abs. 1 BGG
        if "Art. 100 Abs. 1 BGG" in law_set:
            scored["Art. 100 Abs. 1 BGG"] = max(scored.get("Art. 100 Abs. 1 BGG", 0), 0.80)

        # 10. Filter + smart cutoff
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        verified = [(c, s) for c, s in ranked if c in law_set or c in court_set or c in explicit]

        gpt_estimate = exp.get("estimated_citation_count", 25)
        high_conf_count = sum(1 for c, s in verified if s >= 0.80)
        target = gpt_estimate
        if high_conf_count > target:
            target = high_conf_count
        target = max(target, 10)
        target = min(target, gpt_estimate + 8)
        target = min(target, 40)
        cutoff = min(target, len(verified))

        selected = set(c for c, _ in verified[:cutoff])
        predictions[qid] = selected
        print(f"  {qid}: pred={len(selected)} (est={gpt_estimate}, high_conf={high_conf_count})")

    # Write submission
    out_path = BASE / "submissions" / "test_submission_v7.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions.keys()):
            writer.writerow([qid, ";".join(predictions[qid])])

    avg_pred = sum(len(p) for p in predictions.values()) / len(predictions)
    print(f"\nSaved to {out_path}")
    print(f"Total queries: {len(predictions)}, avg predictions: {avg_pred:.1f}")
    print(f"Time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
