"""
V11 pipeline: preserve V7b retrieval, add a strict candidate-judging stage.

Design goals:
1. Keep the V7b retrieval stack as the control.
2. Judge only uncertain candidates, not the whole pool.
3. Never let the LLM invent citations.
4. Cache every judge batch so val/test runs are reproducible and cheap to iterate.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import pickle
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zäöüß]+", text.lower()) if len(t) > 1]


def compact_text(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def build_fuzzy_index(law_set: set[str]) -> dict[tuple[str, str], list[str]]:
    statute_article_map: dict[tuple[str, str], list[str]] = defaultdict(list)
    for cit in sorted(law_set):
        match = re.match(r"Art\.\s+(\d+[a-z]?)\b.*?([A-ZÄÖÜ][A-Za-zÄÖÜäöü]+)\s*$", cit)
        if match:
            art_num, statute = match.group(1), match.group(2)
            statute_article_map[(statute, art_num)].append(cit)
    return statute_article_map


def fuzzy_match_citation(
    cit: str,
    law_set: set[str],
    statute_article_map: dict[tuple[str, str], list[str]],
) -> str | None:
    if cit in law_set:
        return cit

    match = re.match(r"Art\.\s+(\d+[a-z]?)\b(.*?)([A-ZÄÖÜ][A-Za-zÄÖÜäöü]+)\s*$", cit)
    if not match:
        return None
    art_num, statute = match.group(1), match.group(3)

    candidates = statute_article_map.get((statute, art_num), [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    best_match = None
    best_ratio = 0.0
    for candidate in candidates:
        ratio = SequenceMatcher(None, cit, candidate).ratio()
        if ratio > best_ratio or (ratio == best_ratio and (best_match is None or candidate < best_match)):
            best_ratio = ratio
            best_match = candidate

    if best_ratio >= 0.75:
        return best_match
    return None


def fuzzy_match_court(cit: str, court_set: set[str], case_prefix_map: dict[str, list[str]]) -> str | list[str] | None:
    if cit in court_set:
        return cit

    base_cit = re.sub(r"\s+E\.\s+.*$", "", cit).strip()
    siblings = case_prefix_map.get(base_cit, [])
    if siblings:
        for sibling in siblings:
            if sibling == cit:
                return sibling
        return siblings

    if base_cit in case_prefix_map:
        return case_prefix_map[base_cit]
    return None


def load_full_citation_runs(split: str) -> dict[str, dict[str, Any]]:
    runs = []
    for name in [f"{split}_full_citations.json", f"{split}_full_citations_v2.json", f"{split}_full_citations_v3.json"]:
        path = BASE / "precompute" / name
        if path.exists():
            runs.append((name, json.loads(path.read_text())))

    merged: dict[str, dict[str, Any]] = {}
    all_qids = set()
    for _, data in runs:
        all_qids.update(data.keys())

    for qid in all_qids:
        law_freq: dict[str, int] = defaultdict(int)
        court_freq: dict[str, int] = defaultdict(int)
        for _, data in runs:
            row = data.get(qid, {})
            for citation in row.get("law_citations", []):
                law_freq[citation] += 1
            for citation in row.get("court_citations", []):
                court_freq[citation] += 1
        merged[qid] = {
            "law_citations": list(law_freq.keys()),
            "court_citations": list(court_freq.keys()),
            "law_freq": dict(law_freq),
            "court_freq": dict(court_freq),
        }
    return merged


def load_law_text_map() -> dict[str, str]:
    law_text = {}
    with open(BASE / "data" / "laws_de.csv") as f:
        for row in csv.DictReader(f):
            text = " ".join(part for part in [row.get("title", ""), row.get("text", "")] if part)
            law_text[row["citation"]] = compact_text(text, limit=320)
    return law_text


class CourtTextStore:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        if cache_path.exists():
            try:
                self.cache = json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                self.cache = {}

    def ensure(self, citations: set[str]) -> None:
        missing = {citation for citation in citations if citation not in self.cache}
        if not missing:
            return

        found = 0
        with open(BASE / "data" / "court_considerations.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                citation = row["citation"]
                if citation in missing:
                    self.cache[citation] = compact_text(row.get("text", ""), limit=320)
                    missing.remove(citation)
                    found += 1
                    if not missing:
                        break

        for citation in missing:
            self.cache[citation] = ""

        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2))
        print(
            f"  Court text cache: found {found}, missing {len(missing)}, total cached {len(self.cache)}",
            flush=True,
        )

    def get(self, citation: str) -> str:
        return self.cache.get(citation, "")


class JudgeCache:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.entries: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if cache_path.exists():
            with open(cache_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = entry.get("key")
                    if key:
                        self.entries[key] = entry

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self.entries.get(key)
        if not entry:
            return None
        return entry.get("response")

    def add(self, key: str, payload: dict[str, Any], response: dict[str, Any]) -> None:
        entry = {
            "key": key,
            "created_at": time.time(),
            "payload": payload,
            "response": response,
        }
        with self._lock:
            with open(self.cache_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.entries[key] = entry


@dataclass
class Candidate:
    citation: str
    kind: str
    raw_score: float
    final_score: float
    baseline_rank: int
    sources: list[str]
    gpt_full_freq: int = 0
    dense_rank: int | None = None
    court_dense_rank: int | None = None
    bm25_rank: int | None = None
    is_explicit: bool = False
    is_query_case: bool = False
    snippet: str = ""
    judge_label: str | None = None
    judge_confidence: float = 0.0
    judge_reason: str = ""
    auto_bucket: str | None = None

    def source_set(self) -> set[str]:
        return set(self.sources)

    def evidence_lines(self) -> list[str]:
        evidence = []
        if self.gpt_full_freq:
            evidence.append(f"predicted_by_{self.gpt_full_freq}_gpt_runs")
        if self.is_explicit:
            evidence.append("citation_appears_in_question")
        if self.is_query_case:
            evidence.append("gpt_case_citation")
        if self.dense_rank is not None:
            evidence.append(f"law_dense_rank={self.dense_rank}")
        if self.court_dense_rank is not None:
            evidence.append(f"court_dense_rank={self.court_dense_rank}")
        if self.bm25_rank is not None:
            evidence.append(f"bm25_rank={self.bm25_rank}")
        for source in self.sources:
            if source not in {"dense", "bm25"}:
                evidence.append(source)
        return evidence[:8]


@dataclass
class QueryBundle:
    query_id: str
    query: str
    estimated_count: int
    candidates: list[Candidate]
    auto_keep: list[Candidate] = field(default_factory=list)
    judge_laws: list[Candidate] = field(default_factory=list)
    judge_courts: list[Candidate] = field(default_factory=list)
    auto_drop: list[Candidate] = field(default_factory=list)
    selected: list[Candidate] = field(default_factory=list)


@dataclass
class Assets:
    expansions: dict[str, Any]
    case_citations: dict[str, Any]
    full_citations: dict[str, Any]
    law_bm25: Any
    law_cites_bm25: list[str]
    law_set: set[str]
    law_text: dict[str, str]
    statute_article_map: dict[tuple[str, str], list[str]]
    court_set: set[str]
    case_prefix_map: dict[str, list[str]]
    faiss_index: faiss.Index
    faiss_law_cites: list[str]
    court_faiss: faiss.Index | None
    court_faiss_cites: list[str] | None
    embed_model: SentenceTransformer


@dataclass
class V11Config:
    split: str
    use_judge: bool
    judge_model: str
    prompt_version: str
    law_judge_topk: int
    court_judge_topk: int
    law_batch_size: int
    court_batch_size: int
    use_court_dense: bool
    court_dense_query_limit: int
    court_dense_topk: int
    max_output: int
    min_output: int
    court_fraction: float
    min_courts_if_any: int
    must_keep_confidence: float
    cache_path: Path
    court_text_cache_path: Path
    court_dense_cache_path: Path
    query_offset: int
    max_queries: int | None

    @classmethod
    def from_env(cls, split: str) -> "V11Config":
        max_queries = os.getenv("V11_MAX_QUERIES")
        query_offset = os.getenv("V11_QUERY_OFFSET")
        return cls(
            split=split,
            use_judge=os.getenv("V11_USE_JUDGE", "1") == "1",
            judge_model=os.getenv("V11_JUDGE_MODEL", "gpt-5.4"),
            prompt_version=os.getenv("V11_PROMPT_VERSION", "v11_strict_v1"),
            law_judge_topk=int(os.getenv("V11_LAW_JUDGE_TOPK", "60")),
            court_judge_topk=int(os.getenv("V11_COURT_JUDGE_TOPK", "36")),
            law_batch_size=int(os.getenv("V11_LAW_BATCH_SIZE", "20")),
            court_batch_size=int(os.getenv("V11_COURT_BATCH_SIZE", "12")),
            use_court_dense=os.getenv("V11_USE_COURT_DENSE", "1") == "1",
            court_dense_query_limit=int(os.getenv("V11_COURT_DENSE_QUERY_LIMIT", "8")),
            court_dense_topk=int(os.getenv("V11_COURT_DENSE_TOPK", "160")),
            max_output=int(os.getenv("V11_MAX_OUTPUT", "40")),
            min_output=int(os.getenv("V11_MIN_OUTPUT", "10")),
            court_fraction=float(os.getenv("V11_COURT_FRACTION", "0.25")),
            min_courts_if_any=int(os.getenv("V11_MIN_COURTS_IF_ANY", "4")),
            must_keep_confidence=float(os.getenv("V11_MUST_KEEP_CONFIDENCE", "0.86")),
            cache_path=BASE / "precompute" / f"judge_cache_{split}_v11.jsonl",
            court_text_cache_path=BASE / "precompute" / f"court_text_cache_{split}_v11.json",
            court_dense_cache_path=BASE / "precompute" / f"court_dense_hits_{split}_v11.json",
            query_offset=int(query_offset) if query_offset else 0,
            max_queries=int(max_queries) if max_queries else None,
        )


class CourtDenseCache:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.entries: dict[str, dict[str, Any]] = {}
        if cache_path.exists():
            try:
                self.entries = json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                self.entries = {}

    def get(self, query_id: str, signature: str) -> dict[str, int] | None:
        entry = self.entries.get(query_id)
        if not entry or entry.get("signature") != signature:
            return None
        hits = entry.get("hits", [])
        return {
            item["citation"]: int(item["rank"])
            for item in hits
            if isinstance(item, dict) and "citation" in item and "rank" in item
        }

    def put(self, query_id: str, signature: str, hits: dict[str, int]) -> None:
        ordered_hits = [
            {"citation": citation, "rank": rank}
            for citation, rank in sorted(hits.items(), key=lambda item: item[1])
        ]
        self.entries[query_id] = {
            "signature": signature,
            "hits": ordered_hits,
        }
        self.cache_path.write_text(json.dumps(self.entries, ensure_ascii=False, indent=2))


def load_assets(split: str) -> Assets:
    print("Loading assets...", flush=True)
    expansions = json.loads((BASE / "precompute" / f"{split}_query_expansions.json").read_text())
    case_citations = json.loads((BASE / "precompute" / f"{split}_case_citations.json").read_text())
    full_citations = load_full_citation_runs(split)
    print(f"  Loaded {len(full_citations)} merged GPT citation rows", flush=True)

    with open(BASE / "index" / "bm25_laws.pkl", "rb") as f:
        law_data = pickle.load(f)
    law_bm25 = law_data["bm25"]
    law_cites_bm25 = law_data["citations"]
    law_set = set(law_cites_bm25)
    law_text = load_law_text_map()
    statute_article_map = build_fuzzy_index(law_set)

    with open(BASE / "index" / "court_citations.pkl", "rb") as f:
        all_court_cites = pickle.load(f)
    court_set = set(all_court_cites)
    case_prefix_map: dict[str, list[str]] = defaultdict(list)
    for citation in all_court_cites:
        base_cit = re.sub(r"\s+E\.\s+.*$", "", citation).strip()
        case_prefix_map[base_cit].append(citation)
    del all_court_cites
    gc.collect()

    print("Loading FAISS...", flush=True)
    faiss_index = faiss.read_index(str(BASE / "index" / "faiss_laws.index"))
    with open(BASE / "index" / "faiss_laws_citations.pkl", "rb") as f:
        faiss_law_cites = pickle.load(f)

    court_faiss = None
    court_faiss_cites = None
    court_idx_path = BASE / "index" / "faiss_court_openai.index"
    court_cites_path = BASE / "index" / "faiss_court_openai_citations.pkl"
    if court_idx_path.exists() and court_cites_path.exists():
        print("Loading court FAISS...", flush=True)
        court_faiss = faiss.read_index(str(court_idx_path))
        with open(court_cites_path, "rb") as f:
            court_faiss_cites = pickle.load(f)
        print(f"  Court index: {court_faiss.ntotal:,} citations", flush=True)

    embed_model = SentenceTransformer("intfloat/multilingual-e5-large")

    return Assets(
        expansions=expansions,
        case_citations=case_citations,
        full_citations=full_citations,
        law_bm25=law_bm25,
        law_cites_bm25=law_cites_bm25,
        law_set=law_set,
        law_text=law_text,
        statute_article_map=statute_article_map,
        court_set=court_set,
        case_prefix_map=case_prefix_map,
        faiss_index=faiss_index,
        faiss_law_cites=faiss_law_cites,
        court_faiss=court_faiss,
        court_faiss_cites=court_faiss_cites,
        embed_model=embed_model,
    )


def load_rows(
    split: str,
    max_queries: int | None = None,
    query_offset: int = 0,
) -> list[dict[str, str]]:
    with open(BASE / "data" / f"{split}.csv") as f:
        rows = list(csv.DictReader(f))
    query_ids_path = os.getenv("V11_QUERY_IDS_PATH")
    if query_ids_path:
        query_ids = {
            line.strip()
            for line in Path(query_ids_path).read_text().splitlines()
            if line.strip()
        }
        rows = [row for row in rows if row["query_id"] in query_ids]
    if query_offset:
        rows = rows[query_offset:]
    if max_queries is not None:
        return rows[:max_queries]
    return rows


def get_candidate_meta(meta_map: dict[str, dict[str, Any]], citation: str) -> dict[str, Any]:
    if citation not in meta_map:
        meta_map[citation] = {
            "sources": set(),
            "gpt_full_freq": 0,
            "dense_rank": None,
            "court_dense_rank": None,
            "bm25_rank": None,
            "is_explicit": False,
            "is_query_case": False,
        }
    return meta_map[citation]


def build_court_dense_queries(expansion: dict[str, Any], config: V11Config) -> list[str]:
    court_queries = expansion.get("bm25_queries_court", [])[: config.court_dense_query_limit]
    court_queries = [query.strip() for query in court_queries if query.strip()]
    if court_queries:
        return court_queries

    german_terms = [term.strip() for term in expansion.get("german_terms", []) if term.strip()]
    if german_terms:
        return [" ".join(german_terms[:20])]
    return []


def score_court_dense_rank(rank: int) -> float:
    if rank <= 10:
        return 0.72
    if rank <= 25:
        return 0.66
    if rank <= 50:
        return 0.58
    if rank <= 100:
        return 0.50
    return 0.44


def retrieve_court_dense_hits(
    query_id: str,
    expansion: dict[str, Any],
    assets: Assets,
    config: V11Config,
    cache: CourtDenseCache | None = None,
) -> dict[str, int]:
    if not config.use_court_dense or assets.court_faiss is None or not assets.court_faiss_cites:
        return {}

    court_queries = build_court_dense_queries(expansion, config)
    if not court_queries:
        return {}

    signature_payload = {
        "queries": court_queries,
        "topk": config.court_dense_topk,
    }
    signature = hashlib.sha1(
        json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if cache is not None:
        cached = cache.get(query_id, signature)
        if cached is not None:
            return cached

    from openai import OpenAI

    load_dotenv(BASE / ".env")
    client = OpenAI(
        api_key=os.getenv("V11_EMBED_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("V11_EMBED_BASE_URL"),
    )
    response = client.embeddings.create(
        model=os.getenv("V11_EMBED_MODEL", "text-embedding-3-small"),
        input=court_queries,
        dimensions=512,
    )

    best_ranks: dict[str, int] = {}
    for embedding in response.data:
        vector = np.array([embedding.embedding], dtype=np.float32)
        vector /= np.linalg.norm(vector, axis=1, keepdims=True)
        scores, indices = assets.court_faiss.search(vector, config.court_dense_topk)
        for zero_rank, idx in enumerate(indices[0]):
            citation = assets.court_faiss_cites[idx]
            rank = zero_rank + 1
            if citation not in best_ranks or rank < best_ranks[citation]:
                best_ranks[citation] = rank

    if cache is not None:
        cache.put(query_id, signature, best_ranks)
    return best_ranks


def generate_candidates_for_row(
    row: dict[str, str],
    assets: Assets,
    config: V11Config,
    court_dense_cache: CourtDenseCache | None = None,
) -> QueryBundle:
    qid = row["query_id"]
    query = row["query"]
    expansion = assets.expansions.get(qid, {})
    cases = assets.case_citations.get(qid, {})
    full = assets.full_citations.get(qid, {})

    scored: dict[str, float] = {}
    meta_map: dict[str, dict[str, Any]] = {}
    source_tracker: dict[str, set[str]] = defaultdict(set)

    def update_candidate(
        citation: str,
        score: float,
        source: str,
        *,
        gpt_full_freq: int | None = None,
        dense_rank: int | None = None,
        court_dense_rank: int | None = None,
        bm25_rank: int | None = None,
        is_explicit: bool = False,
        is_query_case: bool = False,
    ) -> None:
        scored[citation] = max(scored.get(citation, 0.0), score)
        source_tracker[citation].add(source)
        meta = get_candidate_meta(meta_map, citation)
        meta["sources"].add(source)
        if gpt_full_freq is not None:
            meta["gpt_full_freq"] = max(meta["gpt_full_freq"], gpt_full_freq)
        if dense_rank is not None:
            current = meta["dense_rank"]
            meta["dense_rank"] = dense_rank if current is None else min(current, dense_rank)
        if court_dense_rank is not None:
            current = meta["court_dense_rank"]
            meta["court_dense_rank"] = court_dense_rank if current is None else min(current, court_dense_rank)
        if bm25_rank is not None:
            current = meta["bm25_rank"]
            meta["bm25_rank"] = bm25_rank if current is None else min(current, bm25_rank)
        if is_explicit:
            meta["is_explicit"] = True
        if is_query_case:
            meta["is_query_case"] = True

    law_freq = full.get("law_freq", {})
    court_freq = full.get("court_freq", {})

    for citation in full.get("law_citations", []):
        freq = law_freq.get(citation, 1)
        base_score = min(0.80 + freq * 0.05, 0.95)
        if citation in assets.law_set:
            update_candidate(citation, base_score, "gpt_full", gpt_full_freq=freq)
        else:
            match = fuzzy_match_citation(citation, assets.law_set, assets.statute_article_map)
            if match:
                update_candidate(match, base_score - 0.05, "gpt_full_fuzzy", gpt_full_freq=freq)

    for citation in full.get("court_citations", []):
        freq = court_freq.get(citation, 1)
        base_score = min(0.78 + freq * 0.05, 0.93)
        if citation in assets.court_set:
            update_candidate(citation, base_score, "gpt_full", gpt_full_freq=freq)
        else:
            result = fuzzy_match_court(citation, assets.court_set, assets.case_prefix_map)
            if isinstance(result, str):
                update_candidate(result, 0.85, "gpt_full_fuzzy", gpt_full_freq=freq)
            elif isinstance(result, list):
                for sibling in result[:8]:
                    update_candidate(sibling, 0.70, "gpt_court_sibling", gpt_full_freq=freq)

    for citation in expansion.get("specific_articles", []):
        if citation in assets.law_set:
            update_candidate(citation, 0.92, "gpt_specific")
        else:
            match = fuzzy_match_citation(citation, assets.law_set, assets.statute_article_map)
            if match:
                update_candidate(match, 0.88, "gpt_specific_fuzzy")

    explicit = set(
        re.findall(
            r"Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+(?:\s+lit\.\s+[a-z])?)?\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü]+",
            query,
        )
    )
    for citation in sorted(explicit):
        if citation in assets.law_set:
            update_candidate(citation, 0.95, "explicit", is_explicit=True)
        else:
            match = fuzzy_match_citation(citation, assets.law_set, assets.statute_article_map)
            if match:
                update_candidate(match, 0.93, "explicit_fuzzy", is_explicit=True)
            else:
                update_candidate(citation, 0.95, "explicit", is_explicit=True)

    q_emb = assets.embed_model.encode([f"query: {query}"], normalize_embeddings=True)
    dense_scores, dense_indices = assets.faiss_index.search(q_emb.astype(np.float32), 200)
    for rank, (score, idx) in enumerate(zip(dense_scores[0], dense_indices[0]), start=1):
        citation = assets.faiss_law_cites[idx]
        norm = float(score) * 0.65
        if rank <= 10:
            norm *= 1.3
        update_candidate(citation, norm, "dense", dense_rank=rank)

    court_dense_hits = retrieve_court_dense_hits(qid, expansion, assets, config, court_dense_cache)
    for citation, best_rank in court_dense_hits.items():
        update_candidate(
            citation,
            score_court_dense_rank(best_rank),
            "court_dense",
            court_dense_rank=best_rank,
        )

    bm25_hits: dict[str, float] = {}
    bm25_ranks: dict[str, int] = {}
    for query_text in expansion.get("bm25_queries_laws", []):
        tokens = tokenize(query_text)
        if not tokens:
            continue
        scores_arr = assets.law_bm25.get_scores(tokens)
        ordered = scores_arr.argsort()[-80:][::-1]
        for local_rank, idx in enumerate(ordered, start=1):
            score = scores_arr[idx]
            if score <= 0:
                continue
            citation = assets.law_cites_bm25[idx]
            bm25_hits[citation] = max(bm25_hits.get(citation, 0.0), float(score))
            if citation not in bm25_ranks or local_rank < bm25_ranks[citation]:
                bm25_ranks[citation] = local_rank

    if expansion.get("german_terms"):
        tokens = tokenize(" ".join(expansion["german_terms"]))
        if tokens:
            scores_arr = assets.law_bm25.get_scores(tokens)
            ordered = scores_arr.argsort()[-80:][::-1]
            for local_rank, idx in enumerate(ordered, start=1):
                score = scores_arr[idx]
                if score <= 0:
                    continue
                citation = assets.law_cites_bm25[idx]
                bm25_hits[citation] = max(bm25_hits.get(citation, 0.0), float(score))
                if citation not in bm25_ranks or local_rank < bm25_ranks[citation]:
                    bm25_ranks[citation] = local_rank

    if bm25_hits:
        max_bm25 = max(bm25_hits.values())
        for citation, score in bm25_hits.items():
            norm = (score / max_bm25) * 0.65
            update_candidate(citation, norm, "bm25", bm25_rank=bm25_ranks.get(citation))

    for citation in cases.get("expanded", []):
        if citation in assets.court_set:
            update_candidate(citation, 0.85, "gpt_case", is_query_case=True)

    gpt_court = unique_preserve_order(cases.get("expanded", []) + full.get("court_citations", []))
    seen = set()
    for citation in gpt_court:
        base_cit = re.sub(r"\s+E\.\s+.*$", "", citation).strip()
        if base_cit in seen:
            continue
        seen.add(base_cit)
        siblings = assets.case_prefix_map.get(base_cit, [])
        parent_score = scored.get(citation, 0.5)
        for sibling in siblings[:15]:
            if sibling not in scored:
                update_candidate(sibling, parent_score * 0.30, "cocitation")

    raw_scores = dict(scored)
    for citation in list(scored):
        source_count = len(source_tracker[citation])
        if source_count >= 2:
            scored[citation] = min(scored[citation] * 1.25, 0.96)
        if source_count >= 3:
            scored[citation] = min(scored[citation] * 1.35, 0.98)

    if "Art. 100 Abs. 1 BGG" in assets.law_set:
        update_candidate("Art. 100 Abs. 1 BGG", max(scored.get("Art. 100 Abs. 1 BGG", 0.0), 0.80), "boilerplate")
        raw_scores["Art. 100 Abs. 1 BGG"] = max(raw_scores.get("Art. 100 Abs. 1 BGG", 0.0), 0.80)

    ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    verified: list[Candidate] = []
    for _, (citation, final_score) in enumerate(ranked, start=1):
        if not (citation in assets.law_set or citation in assets.court_set or citation in explicit):
            continue
        meta = get_candidate_meta(meta_map, citation)
        if citation in assets.court_set:
            kind = "court"
        else:
            kind = "law"
        verified.append(
            Candidate(
                citation=citation,
                kind=kind,
                raw_score=raw_scores.get(citation, final_score),
                final_score=final_score,
                baseline_rank=len(verified) + 1,
                sources=sorted(meta["sources"]),
                gpt_full_freq=meta["gpt_full_freq"],
                dense_rank=meta["dense_rank"],
                court_dense_rank=meta["court_dense_rank"],
                bm25_rank=meta["bm25_rank"],
                is_explicit=meta["is_explicit"],
                is_query_case=meta["is_query_case"],
            )
        )

    return QueryBundle(
        query_id=qid,
        query=query,
        estimated_count=expansion.get("estimated_citation_count", 25),
        candidates=verified,
    )


def is_auto_keep(candidate: Candidate) -> bool:
    sources = candidate.source_set()
    if candidate.is_explicit:
        return True
    if candidate.kind == "law":
        if candidate.gpt_full_freq >= 3 and candidate.raw_score >= 0.95:
            return True
        if candidate.gpt_full_freq >= 2 and "gpt_specific" in sources and candidate.raw_score >= 0.92:
            return True
        if candidate.gpt_full_freq >= 2 and "gpt_specific_fuzzy" in sources and candidate.raw_score >= 0.88:
            return True
    else:
        if candidate.gpt_full_freq >= 3 and candidate.raw_score >= 0.90:
            return True
        if candidate.gpt_full_freq >= 2 and "gpt_case" in sources and candidate.raw_score >= 0.85:
            return True
    return False


def is_auto_drop(candidate: Candidate) -> bool:
    sources = candidate.source_set()
    if candidate.final_score < 0.20:
        return True
    if sources == {"cocitation"}:
        return True
    if candidate.kind == "court" and sources <= {"cocitation", "gpt_court_sibling"} and candidate.final_score < 0.72:
        return True
    if candidate.kind == "court" and sources == {"court_dense"} and candidate.final_score < 0.50:
        return True
    if candidate.kind == "court" and sources <= {"court_dense", "cocitation", "gpt_court_sibling"} and candidate.final_score < 0.58:
        return True
    if candidate.kind == "law" and sources <= {"dense"} and candidate.final_score < 0.35:
        return True
    if candidate.kind == "law" and sources <= {"bm25"} and candidate.final_score < 0.35:
        return True
    if candidate.kind == "law" and sources <= {"dense", "bm25"} and candidate.final_score < 0.40:
        return True
    return False


def bucket_candidates(bundle: QueryBundle, config: V11Config) -> None:
    uncertain_laws = []
    uncertain_courts = []
    for candidate in bundle.candidates:
        if is_auto_keep(candidate):
            candidate.auto_bucket = "auto_keep"
            bundle.auto_keep.append(candidate)
            continue
        if is_auto_drop(candidate):
            candidate.auto_bucket = "auto_drop"
            bundle.auto_drop.append(candidate)
            continue
        if candidate.kind == "law":
            uncertain_laws.append(candidate)
        else:
            uncertain_courts.append(candidate)

    bundle.judge_laws = uncertain_laws[: config.law_judge_topk]
    bundle.judge_courts = uncertain_courts[: config.court_judge_topk]
    for candidate in uncertain_laws[config.law_judge_topk :] + uncertain_courts[config.court_judge_topk :]:
        candidate.auto_bucket = "auto_drop"
        bundle.auto_drop.append(candidate)


def gather_court_citations_for_judge(bundles: list[QueryBundle]) -> set[str]:
    citations = set()
    for bundle in bundles:
        for candidate in bundle.judge_courts:
            citations.add(candidate.citation)
    return citations


def annotate_snippets(bundles: list[QueryBundle], assets: Assets, court_text_store: CourtTextStore) -> None:
    for bundle in bundles:
        for candidate in bundle.judge_laws:
            candidate.snippet = assets.law_text.get(candidate.citation, candidate.citation)
        for candidate in bundle.judge_courts:
            snippet = court_text_store.get(candidate.citation)
            candidate.snippet = snippet or candidate.citation


def judge_prompt(kind: str) -> str:
    if kind == "law":
        kind_rules = (
            "For laws, the exact article matters. Reject topically related articles that are not likely to be cited. "
            "Common procedural provisions should only be accepted if they would realistically appear in the judgment."
        )
    else:
        kind_rules = (
            "For courts, the exact case and consideration matter. If the base case seems relevant but this exact Erwägung "
            "looks doubtful, prefer plausible or reject."
        )

    return (
        "You are reviewing Swiss legal citations for a Federal Supreme Court judgment.\n"
        "Decide whether the court would ACTUALLY cite each exact candidate citation in an opinion answering the question.\n"
        "Be conservative: many candidates are topically related but should still be rejected.\n\n"
        "Labels:\n"
        "- must_include: strong likelihood this exact citation would be cited\n"
        "- plausible: defensible supporting citation, but not strong enough for must_include\n"
        "- reject: unlikely, wrong granularity, wrong authority, or only loosely related\n\n"
        f"{kind_rules}\n"
        "Never invent new citations. Only judge the candidates provided.\n"
        'Return JSON: {"decisions":[{"id":1,"label":"must_include|plausible|reject","confidence":0.0,"reason":"short"}]}'
    )


def build_batch_payload(bundle: QueryBundle, kind: str, candidates: list[Candidate], config: V11Config) -> dict[str, Any]:
    payload_candidates = []
    for idx, candidate in enumerate(candidates, start=1):
        payload_candidates.append(
            {
                "id": idx,
                "citation": candidate.citation,
                "evidence": candidate.evidence_lines(),
                "snippet": compact_text(candidate.snippet, limit=260),
            }
        )
    return {
        "prompt_version": config.prompt_version,
        "query_id": bundle.query_id,
        "kind": kind,
        "query": bundle.query,
        "candidates": payload_candidates,
    }


def parse_json_response(message: Any) -> dict[str, Any]:
    raw_candidates: list[str] = []
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        raw_candidates.append(content.strip())

    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        raw_candidates.append(reasoning_content.strip())

    for raw in raw_candidates:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    pass

    raise json.JSONDecodeError(
        "Unable to parse JSON object from model response",
        raw_candidates[0] if raw_candidates else "",
        0,
    )


def judge_batch(
    bundle: QueryBundle,
    kind: str,
    candidates: list[Candidate],
    config: V11Config,
    judge_cache: JudgeCache,
) -> dict[str, tuple[str, float, str]]:
    if not candidates:
        return {}

    payload = build_batch_payload(bundle, kind, candidates, config)
    key = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    cached = judge_cache.get(key)
    if cached is not None:
        response_data = cached
    elif not config.use_judge:
        response_data = {
            "decisions": [
                {
                    "id": idx,
                    "label": "plausible" if candidate.final_score >= 0.50 else "reject",
                    "confidence": 0.0,
                    "reason": "judge_disabled",
                }
                for idx, candidate in enumerate(candidates, start=1)
            ]
        }
        judge_cache.add(key, payload, response_data)
    else:
        from openai import OpenAI

        load_dotenv(BASE / ".env")
        client = OpenAI(
            api_key=(
                os.getenv("V11_API_KEY")
                or os.getenv("LLM_API_KEY")
                or os.getenv("DEEPSEEK_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            ),
            base_url=os.getenv("V11_BASE_URL") or os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL"),
        )
        kwargs = {
            "model": config.judge_model,
            "messages": [
                {"role": "system", "content": judge_prompt(kind)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        use_max_tokens = os.getenv("V11_USE_MAX_TOKENS", "0") == "1" or config.judge_model.startswith("deepseek")
        if use_max_tokens:
            kwargs["max_tokens"] = int(os.getenv("V11_MAX_TOKENS", "8000"))
        else:
            kwargs["max_completion_tokens"] = int(os.getenv("V11_MAX_COMPLETION_TOKENS", "2400"))

        max_attempts = max(1, int(os.getenv("V11_JUDGE_MAX_ATTEMPTS", "3")))
        response_data = None
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.chat.completions.create(**kwargs)
                response_data = parse_json_response(response.choices[0].message)
                break
            except Exception as exc:  # noqa: BLE001 — we want to recover from ANY judge failure
                last_error = exc
                if attempt < max_attempts:
                    sleep_seconds = min(30.0, 2.0 ** attempt)
                    print(
                        f"    [judge retry] {bundle.query_id} {kind} "
                        f"attempt {attempt}/{max_attempts} failed: "
                        f"{type(exc).__name__}: {exc}; sleeping {sleep_seconds:.1f}s",
                        flush=True,
                    )
                    time.sleep(sleep_seconds)
                else:
                    print(
                        f"    [judge fallback] {bundle.query_id} {kind} "
                        f"all {max_attempts} attempts failed: "
                        f"{type(exc).__name__}: {exc}; labelling batch as reject",
                        flush=True,
                    )

        if response_data is None:
            response_data = {
                "decisions": [
                    {
                        "id": idx,
                        "label": "reject",
                        "confidence": 0.0,
                        "reason": (
                            f"judge_unavailable: "
                            f"{type(last_error).__name__ if last_error else 'unknown'}"
                        ),
                    }
                    for idx, candidate in enumerate(candidates, start=1)
                ],
                "_fallback": True,
            }

        judge_cache.add(key, payload, response_data)

    by_id: dict[int, tuple[str, float, str]] = {}
    for item in response_data.get("decisions", []):
        try:
            idx = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        label = str(item.get("label", "reject")).strip().lower()
        if label not in {"must_include", "plausible", "reject"}:
            label = "reject"
        confidence = item.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        reason = str(item.get("reason", "")).strip()
        by_id[idx] = (label, confidence, reason)

    decisions = {}
    for idx, candidate in enumerate(candidates, start=1):
        label, confidence, reason = by_id.get(idx, ("reject", 0.0, "missing_from_response"))
        decisions[candidate.citation] = (label, confidence, reason)
    return decisions


def apply_judgments(bundle: QueryBundle, config: V11Config, judge_cache: JudgeCache) -> None:
    for candidate in bundle.auto_keep:
        candidate.judge_label = "must_include"
        candidate.judge_confidence = 1.0
        candidate.judge_reason = "auto_keep"

    for kind, candidates, batch_size in [
        ("law", bundle.judge_laws, config.law_batch_size),
        ("court", bundle.judge_courts, config.court_batch_size),
    ]:
        for offset in range(0, len(candidates), batch_size):
            batch = candidates[offset : offset + batch_size]
            decisions = judge_batch(bundle, kind, batch, config, judge_cache)
            for candidate in batch:
                label, confidence, reason = decisions.get(candidate.citation, ("reject", 0.0, "missing"))
                candidate.judge_label = label
                candidate.judge_confidence = confidence
                candidate.judge_reason = reason

    for candidate in bundle.auto_drop:
        candidate.judge_label = "reject"
        candidate.judge_confidence = 0.0
        candidate.judge_reason = "auto_drop"


def candidate_priority(candidate: Candidate) -> tuple[float, float, float]:
    label_bonus = {"must_include": 2.0, "plausible": 1.0}.get(candidate.judge_label or "reject", 0.0)
    return (label_bonus, candidate.judge_confidence, candidate.final_score)


def candidate_priority_sort_key(candidate: Candidate) -> tuple[float, float, float, str]:
    label_bonus = {"must_include": 2.0, "plausible": 1.0}.get(candidate.judge_label or "reject", 0.0)
    return (-label_bonus, -candidate.judge_confidence, -candidate.final_score, candidate.citation)


def select_candidates(bundle: QueryBundle, config: V11Config) -> list[Candidate]:
    must_candidates = []
    plausible_candidates = []
    for candidate in bundle.candidates:
        if candidate.judge_label == "must_include":
            must_candidates.append(candidate)
        elif candidate.judge_label == "plausible":
            plausible_candidates.append(candidate)

    must_candidates.sort(key=candidate_priority_sort_key)
    plausible_candidates.sort(key=candidate_priority_sort_key)

    locked_keep = [
        candidate
        for candidate in must_candidates
        if candidate in bundle.auto_keep or candidate.judge_confidence >= config.must_keep_confidence
    ]
    locked_keep.sort(key=candidate_priority_sort_key)

    target = bundle.estimated_count
    target = max(target, len(locked_keep))
    target = max(target, config.min_output)
    target = min(target, bundle.estimated_count + 8)
    target = min(target, config.max_output)

    selected: list[Candidate] = []
    selected_ids = set()
    for candidate in locked_keep:
        if len(selected) >= target:
            break
        if candidate.citation in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.citation)

    positive_courts = [
        candidate
        for candidate in must_candidates + plausible_candidates
        if candidate.kind == "court" and candidate.citation not in selected_ids
    ]
    positive_courts.sort(key=candidate_priority_sort_key)
    selected_courts = sum(1 for candidate in selected if candidate.kind == "court")
    if positive_courts:
        soft_court_target = max(config.min_courts_if_any, round(target * config.court_fraction))
        while positive_courts and selected_courts < soft_court_target and len(selected) < target:
            candidate = positive_courts.pop(0)
            selected.append(candidate)
            selected_ids.add(candidate.citation)
            selected_courts += 1

    remaining_positive = [
        candidate
        for candidate in must_candidates + plausible_candidates
        if candidate.citation not in selected_ids
    ]
    remaining_positive.sort(key=candidate_priority_sort_key)
    for candidate in remaining_positive:
        if len(selected) >= target:
            break
        selected.append(candidate)
        selected_ids.add(candidate.citation)

    bundle.selected = selected
    return selected


def evaluate_predictions(predictions: dict[str, set[str]], gold_map: dict[str, set[str]]) -> tuple[float, dict[str, float]]:
    f1_scores = {}
    for qid, pred in predictions.items():
        gold = gold_map[qid]
        tp = len(pred & gold)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores[qid] = f1
    macro_f1 = sum(f1_scores.values()) / len(f1_scores) if f1_scores else 0.0
    return macro_f1, f1_scores


def run_pipeline(split: str, output_path: Path, evaluate: bool = False) -> dict[str, set[str]]:
    t0 = time.time()
    config = V11Config.from_env(split)
    print(
        (
            f"V11 config: split={split}, use_judge={config.use_judge}, model={config.judge_model}, "
            f"prompt={config.prompt_version}, law_topk={config.law_judge_topk}, court_topk={config.court_judge_topk}"
        ),
        flush=True,
    )

    assets = load_assets(split)
    rows = load_rows(split, max_queries=config.max_queries, query_offset=config.query_offset)
    print(f"Processing {len(rows)} {split} queries...\n", flush=True)

    court_dense_cache = CourtDenseCache(config.court_dense_cache_path)
    bundles = []
    for row in rows:
        bundle = generate_candidates_for_row(row, assets, config, court_dense_cache)
        bucket_candidates(bundle, config)
        bundles.append(bundle)

    court_text_store = CourtTextStore(config.court_text_cache_path)
    needed_courts = gather_court_citations_for_judge(bundles)
    if needed_courts:
        print(f"Preparing {len(needed_courts)} court snippets for judge batches...", flush=True)
        court_text_store.ensure(needed_courts)
    annotate_snippets(bundles, assets, court_text_store)

    judge_cache = JudgeCache(config.cache_path)
    predictions: dict[str, set[str]] = {}

    gold_map = {}
    if evaluate:
        gold_map = {
            row["query_id"]: set(row["gold_citations"].split(";"))
            for row in rows
        }

    for bundle in bundles:
        apply_judgments(bundle, config, judge_cache)
        selected = select_candidates(bundle, config)
        predictions[bundle.query_id] = {candidate.citation for candidate in selected}

        must_count = sum(1 for candidate in bundle.candidates if candidate.judge_label == "must_include")
        plausible_count = sum(1 for candidate in bundle.candidates if candidate.judge_label == "plausible")
        reject_count = sum(1 for candidate in bundle.candidates if candidate.judge_label == "reject")

        line = (
            f"  {bundle.query_id}: auto_keep={len(bundle.auto_keep)}, judge="
            f"{len(bundle.judge_laws) + len(bundle.judge_courts)}, selected={len(selected)} "
            f"(must={must_count}, plausible={plausible_count}, reject={reject_count})"
        )

        if evaluate:
            gold = gold_map[bundle.query_id]
            pred = predictions[bundle.query_id]
            tp = len(pred & gold)
            precision = tp / len(pred) if pred else 0.0
            recall = tp / len(gold) if gold else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            line += f" | gold={len(gold)}, TP={tp}, P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}"
        print(line, flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions):
            writer.writerow([qid, ";".join(sorted(predictions[qid]))])

    if evaluate:
        macro_f1, _ = evaluate_predictions(predictions, gold_map)
        print(f"\n=== V11 MACRO F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%) ===")

    avg_predictions = sum(len(pred) for pred in predictions.values()) / len(predictions) if predictions else 0.0
    print(f"Saved to {output_path}")
    print(f"Average predictions: {avg_predictions:.1f}")
    print(f"Total time: {time.time() - t0:.0f}s")
    return predictions
