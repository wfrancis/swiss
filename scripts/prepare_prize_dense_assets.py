#!/usr/bin/env python3
"""Prepare optional dense/rerank assets for the prize offline notebook.

This script is for local preparation only. The Kaggle notebook itself still
runs offline. Typical use:

  python3 scripts/prepare_prize_dense_assets.py --all
  python3 scripts/package_prize_offline_for_kaggle.py --stage all

Outputs, when requested:

  models/intfloat-multilingual-e5-large/
  models/bge-reranker-v2-m3/
  precompute/law_dense_e5_embeddings.npy
  precompute/law_dense_e5_citations.json
  precompute/compact_court_dense_e5.npz
  precompute/compact_court_dense_e5_embeddings.npy
  precompute/compact_court_dense_e5_citations.json
  precompute/law_chunk_dense_e5_embeddings.npy
  precompute/law_chunk_dense_e5_citations.json
"""

from __future__ import annotations

import argparse
import csv
import heapq
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np


REPO = Path(os.getenv("SWISS_REPO_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.getenv("SWISS_DATA_DIR", REPO / "data"))
PRECOMP = Path(os.getenv("SWISS_PRECOMP_DIR", REPO / "precompute"))
MODELS = Path(os.getenv("SWISS_MODELS_DIR", REPO / "models"))

DEFAULT_EMBED_REPO = "intfloat/multilingual-e5-large"
DEFAULT_RERANK_REPO = "BAAI/bge-reranker-v2-m3"
EMBED_DIR = MODELS / "intfloat-multilingual-e5-large"
RERANK_DIR = MODELS / "bge-reranker-v2-m3"
COMPACT_OUT = PRECOMP / "compact_court_dense_e5.npz"
COMPACT_EMB_OUT = PRECOMP / "compact_court_dense_e5_embeddings.npy"
COMPACT_CITES_OUT = PRECOMP / "compact_court_dense_e5_citations.json"
EXPANDED_COURT_OUT = PRECOMP / "expanded_court_dense_e5.npz"
EXPANDED_COURT_EMB_OUT = PRECOMP / "expanded_court_dense_e5_embeddings.npy"
EXPANDED_COURT_CITES_OUT = PRECOMP / "expanded_court_dense_e5_citations.json"
LAW_EMB_OUT = PRECOMP / "law_dense_e5_embeddings.npy"
LAW_CITES_OUT = PRECOMP / "law_dense_e5_citations.json"
LAW_CHUNK_EMB_OUT = PRECOMP / "law_chunk_dense_e5_embeddings.npy"
LAW_CHUNK_CITES_OUT = PRECOMP / "law_chunk_dense_e5_citations.json"


def download_model(repo_id: str, out_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id} -> {out_dir}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    print(f"[download] complete: {out_dir}", flush=True)


def load_compact_court_texts() -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in [
        "citation_first_chunk_optC.json",
        "court_text_cache_train_v11.json",
        "court_text_cache_val_v11.json",
    ]:
        path = PRECOMP / rel
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for citation, text in payload.items():
            citation = str(citation).strip()
            if citation and not citation.startswith("Art.") and citation not in out:
                out[citation] = str(text)
    return out


def court_base_decision(citation: str) -> str:
    return citation.split(" E. ", 1)[0].strip() or citation


def court_prefix_family(citation: str) -> str:
    if citation.startswith("BGE "):
        return "BGE"
    m = re.match(r"^(\d+[A-Z]{1,2}_)", citation)
    if m:
        return m.group(1)
    return "OTHER"


def court_candidate_score(citation: str, text: str) -> float:
    score = 0.0
    if citation.startswith("BGE "):
        score += 3.0
    if " E. " in citation:
        score += 2.0
    if re.match(r"^\d+[A-Z]{1,2}_\d+/\d{4}", citation):
        score += 1.0
    text_len = len(text)
    if 180 <= text_len <= 1800:
        score += 1.0
    elif text_len > 60:
        score += 0.35
    if re.search(r"\bArt\.\s*\d+", text):
        score += 0.35
    if re.search(r"\b(BGG|StPO|StGB|ZGB|OR|ATSG|IVG)\b", text):
        score += 0.25
    return score + min(0.50, text_len / 1800.0)


def load_expanded_court_texts(
    max_docs: int,
    *,
    max_per_base: int,
    max_per_prefix: int,
    heap_per_prefix: int,
) -> dict[str, str]:
    """Build a query-independent bounded court subset from the public corpus.

    The compact seed set is preserved first. Extra rows are selected only from
    data/court_considerations.csv and capped by base decision/prefix family so
    hidden-query recall improves without exploding runtime or memorizing any
    public/private test-query-specific cache.
    """
    if max_docs <= 0:
        raise SystemExit("--expanded-court-max-docs must be > 0")
    max_docs = min(max_docs, 100_000)
    seed_texts = load_compact_court_texts()
    selected: dict[str, str] = dict(seed_texts)
    base_counts: dict[str, int] = {}
    prefix_counts: dict[str, int] = {}
    for citation in selected:
        base_counts[court_base_decision(citation)] = base_counts.get(court_base_decision(citation), 0) + 1
        prefix_counts[court_prefix_family(citation)] = prefix_counts.get(court_prefix_family(citation), 0) + 1
    if len(selected) >= max_docs:
        return dict(list(sorted(selected.items()))[:max_docs])

    path = DATA_DIR / "court_considerations.csv"
    if not path.exists():
        raise SystemExit(f"Court corpus missing: {path}")

    heaps: dict[str, list[tuple[float, int, str, str]]] = {}
    csv.field_size_limit(sys.maxsize)
    t0 = time.time()
    scanned = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row_idx, row in enumerate(csv.DictReader(f), start=1):
            scanned = row_idx
            citation = str(row.get("citation", "")).strip()
            if not citation or citation.startswith("Art.") or citation in selected:
                continue
            text = str(row.get("text", "")).strip()
            if len(text) < 60:
                continue
            prefix = court_prefix_family(citation)
            score = court_candidate_score(citation, text)
            heap = heaps.setdefault(prefix, [])
            item = (score, -row_idx, citation, text[:900])
            if len(heap) < heap_per_prefix:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            if row_idx % 250_000 == 0:
                kept = sum(len(h) for h in heaps.values())
                print(f"[expanded-court] scanned={row_idx:,} heap_kept={kept:,} elapsed={time.time() - t0:.1f}s", flush=True)

    candidates = [item for heap in heaps.values() for item in heap]
    candidates.sort(reverse=True)
    for _score, _neg_row_idx, citation, text in candidates:
        if citation in selected:
            continue
        base = court_base_decision(citation)
        prefix = court_prefix_family(citation)
        if base_counts.get(base, 0) >= max_per_base:
            continue
        if prefix_counts.get(prefix, 0) >= max_per_prefix:
            continue
        selected[citation] = text
        base_counts[base] = base_counts.get(base, 0) + 1
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        if len(selected) >= max_docs:
            break

    print(
        f"[expanded-court] selected={len(selected):,} seeds={len(seed_texts):,} "
        f"scanned={scanned:,} prefixes={len(prefix_counts):,}",
        flush=True,
    )
    return selected


def load_law_texts() -> tuple[list[str], list[str]]:
    path = DATA_DIR / "laws_de.csv"
    citations: list[str] = []
    docs: list[str] = []
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            citation = str(row.get("citation", "")).strip()
            if not citation:
                continue
            title = row.get("title", "")
            text = row.get("text", "")
            citations.append(citation)
            docs.append(f"{citation} {title} {text}"[:900])
    return citations, docs


def load_law_rows() -> list[tuple[str, str, str]]:
    path = DATA_DIR / "laws_de.csv"
    rows: list[tuple[str, str, str]] = []
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            citation = str(row.get("citation", "")).strip()
            if not citation:
                continue
            title = str(row.get("title", "")).strip()
            text = str(row.get("text", "")).strip()
            rows.append((citation, title, text))
    return rows


def wordish_chunks(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    if chunk_words <= 0:
        raise SystemExit("--law-chunk-words must be > 0")
    if overlap_words < 0:
        raise SystemExit("--law-chunk-overlap must be >= 0")
    if overlap_words >= chunk_words:
        raise SystemExit("--law-chunk-overlap must be smaller than --law-chunk-words")

    words = text.split()
    if not words:
        return [""]
    if len(words) <= chunk_words:
        return [" ".join(words)]

    chunks: list[str] = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        stop = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:stop]))
        if stop == len(words):
            break
    return chunks


def load_law_chunk_texts(
    max_docs: int,
    chunk_words: int,
    overlap_words: int,
) -> tuple[list[str], list[str]]:
    rows = load_law_rows()
    if max_docs > 0:
        rows = rows[:max_docs]

    citations: list[str] = []
    docs: list[str] = []
    for citation, title, text in rows:
        for chunk in wordish_chunks(text, chunk_words, overlap_words):
            citations.append(citation)
            docs.append(" ".join(part for part in [citation, title, chunk] if part))
    return citations, docs


def corpus_signature(citations: list[str], docs: list[str]) -> str:
    h = hashlib.sha256()
    for citation, doc in zip(citations, docs):
        h.update(citation.encode("utf-8"))
        h.update(b"\0")
        h.update(doc.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


class LocalE5Encoder:
    def __init__(self, model_dir: Path, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        if device is None:
            forced = os.getenv("SWISS_DENSE_DEVICE")
            if forced:
                device = forced
            elif torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                if props.major >= 7:
                    device = "cuda"
                elif os.getenv("SWISS_DENSE_REQUIRE_CUDA", "0").lower() in {"1", "true", "yes", "on"}:
                    raise SystemExit(
                        f"CUDA GPU {props.name} has capability {props.major}.{props.minor}, "
                        "but this PyTorch build requires sm_70+."
                    )
                else:
                    print(
                        f"[dense] GPU {props.name} capability {props.major}.{props.minor} "
                        "is unsupported by this PyTorch build; falling back to CPU.",
                        flush=True,
                    )
                    device = "cpu"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
        self.model.to(device)
        self.model.eval()

    def encode(
        self,
        texts: list[str],
        *,
        prefix: str,
        batch_size: int,
        max_length: int,
        verbose: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        torch = self.torch
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = [f"{prefix}{text}" for text in texts[start:start + batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
                hidden = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            rows.append(pooled.detach().cpu().numpy().astype(np.float32))
            if verbose:
                print(f"  encoded {min(start + batch_size, len(texts)):,}/{len(texts):,}", flush=True)
        return np.vstack(rows)


def build_compact_court_embeddings(model_dir: Path, out_path: Path, batch_size: int, max_docs: int) -> None:
    texts = load_compact_court_texts()
    if not texts:
        raise SystemExit("No compact court texts found in precompute/.")
    items = sorted(texts.items())[:max_docs]
    citations = [c for c, _ in items]
    docs = [f"{c} {text}"[:900] for c, text in items]

    t0 = time.time()
    encoder = LocalE5Encoder(model_dir)
    print(f"[compact] encoder device={encoder.device}", flush=True)
    embeddings = encoder.encode(docs, prefix="passage: ", batch_size=batch_size, max_length=384)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        citations=np.asarray(citations, dtype=np.str_),
        embeddings=embeddings.astype(np.float16),
    )
    mb = out_path.stat().st_size / 1e6
    print(f"[compact] wrote {out_path} docs={len(citations):,} size={mb:.1f} MB elapsed={time.time() - t0:.1f}s", flush=True)


def build_expanded_court_embeddings(
    model_dir: Path,
    out_path: Path,
    batch_size: int,
    max_docs: int,
    max_per_base: int,
    max_per_prefix: int,
    heap_per_prefix: int,
) -> None:
    texts = load_expanded_court_texts(
        max_docs,
        max_per_base=max_per_base,
        max_per_prefix=max_per_prefix,
        heap_per_prefix=heap_per_prefix,
    )
    if not texts:
        raise SystemExit("No expanded court texts found.")
    items = sorted(texts.items())
    citations = [c for c, _ in items]
    docs = [f"{c} {text}"[:900] for c, text in items]

    t0 = time.time()
    encoder = LocalE5Encoder(model_dir)
    print(f"[expanded-court] encoder device={encoder.device} rows={len(citations):,}", flush=True)
    embeddings = encoder.encode(docs, prefix="passage: ", batch_size=batch_size, max_length=384)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        citations=np.asarray(citations, dtype=np.str_),
        embeddings=embeddings.astype(np.float16),
    )
    mb = out_path.stat().st_size / 1e6
    print(
        f"[expanded-court] wrote {out_path} docs={len(citations):,} "
        f"size={mb:.1f} MB elapsed={time.time() - t0:.1f}s",
        flush=True,
    )


def export_compact_court_npys(npz_path: Path, emb_out: Path, cites_out: Path) -> None:
    if not npz_path.exists():
        raise SystemExit(f"Compact court npz missing: {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    citations = [str(c) for c in data["citations"].tolist()]
    embeddings = data["embeddings"].astype(np.float16)
    emb_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_out, embeddings)
    cites_out.write_text(json.dumps(citations, ensure_ascii=False), encoding="utf-8")
    print(
        f"[compact-export] wrote {emb_out} ({emb_out.stat().st_size/1e6:.1f} MB) "
        f"and {cites_out} ({len(citations):,} citations)",
        flush=True,
    )


def build_law_embeddings(
    model_dir: Path,
    emb_out: Path,
    cites_out: Path,
    batch_size: int,
    max_docs: int,
    resume: bool,
) -> None:
    citations, docs = load_law_texts()
    if max_docs > 0:
        citations = citations[:max_docs]
        docs = docs[:max_docs]
    if not citations:
        raise SystemExit("No law texts found in data/laws_de.csv")

    t0 = time.time()
    encoder = LocalE5Encoder(model_dir)
    print(f"[laws] encoder device={encoder.device}", flush=True)
    emb_out.parent.mkdir(parents=True, exist_ok=True)
    status_path = emb_out.with_suffix(".status.json")

    # Stream directly to a .npy memmap. This avoids a large all-float32 matrix
    # in RAM and leaves an inspectable artifact if the job is interrupted.
    sample = encoder.encode(["dimension probe"], prefix="passage: ", batch_size=1, max_length=64, verbose=False)
    dim = int(sample.shape[1])
    expected_shape = (len(docs), dim)
    start_row = 0
    embeddings = None

    if resume and emb_out.exists() and cites_out.exists():
        try:
            existing_citations = json.loads(cites_out.read_text(encoding="utf-8"))
            existing = np.load(emb_out, mmap_mode="r+")
            if (
                list(existing_citations) == citations
                and existing.shape == expected_shape
                and existing.dtype == np.float16
            ):
                embeddings = existing
                if status_path.exists():
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    start_row = int(status.get("rows_done", 0))
                    start_row = max(0, min(start_row, len(docs)))
                    start_row -= start_row % max(1, batch_size)
                print(f"[laws] resuming existing matrix at row {start_row:,}/{len(docs):,}", flush=True)
            else:
                print("[laws] existing matrix does not match current corpus/model shape; rebuilding", flush=True)
        except Exception as exc:
            print(f"[laws] resume check failed ({type(exc).__name__}: {exc}); rebuilding", flush=True)

    if embeddings is None:
        cites_out.write_text(json.dumps(citations, ensure_ascii=False), encoding="utf-8")
        embeddings = np.lib.format.open_memmap(
            emb_out,
            mode="w+",
            dtype=np.float16,
            shape=expected_shape,
        )

    for start in range(start_row, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        encoded = encoder.encode(batch, prefix="passage: ", batch_size=batch_size, max_length=384, verbose=False)
        stop = start + encoded.shape[0]
        embeddings[start:stop] = encoded.astype(np.float16)
        if stop % max(batch_size * 20, 1000) < batch_size or stop == len(docs):
            embeddings.flush()
            status_path.write_text(
                json.dumps(
                    {
                        "rows_total": len(docs),
                        "rows_done": stop,
                        "dim": dim,
                        "elapsed_sec": round(time.time() - t0, 1),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[laws] checkpoint {stop:,}/{len(docs):,}", flush=True)
    embeddings.flush()
    mb = emb_out.stat().st_size / 1e6
    print(
        f"[laws] wrote {emb_out} rows={len(citations):,} size={mb:.1f} MB "
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )


def build_law_chunk_embeddings(
    model_dir: Path,
    emb_out: Path,
    cites_out: Path,
    batch_size: int,
    max_docs: int,
    chunk_words: int,
    overlap_words: int,
    max_length: int,
    resume: bool,
) -> None:
    citations, docs = load_law_chunk_texts(max_docs, chunk_words, overlap_words)
    if not citations:
        raise SystemExit("No law texts found in data/laws_de.csv")

    t0 = time.time()
    signature = corpus_signature(citations, docs)
    encoder = LocalE5Encoder(model_dir)
    print(
        f"[law-chunks] encoder device={encoder.device} rows={len(docs):,} "
        f"chunk_words={chunk_words} overlap={overlap_words}",
        flush=True,
    )
    emb_out.parent.mkdir(parents=True, exist_ok=True)
    status_path = emb_out.with_suffix(".status.json")

    sample = encoder.encode(["dimension probe"], prefix="passage: ", batch_size=1, max_length=64, verbose=False)
    dim = int(sample.shape[1])
    expected_shape = (len(docs), dim)
    start_row = 0
    embeddings = None

    if resume and emb_out.exists() and cites_out.exists() and status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            existing_citations = json.loads(cites_out.read_text(encoding="utf-8"))
            existing = np.load(emb_out, mmap_mode="r+")
            status_matches = (
                status.get("corpus_sha256") == signature
                and int(status.get("chunk_words", -1)) == chunk_words
                and int(status.get("overlap_words", -1)) == overlap_words
                and int(status.get("max_length", -1)) == max_length
            )
            if (
                status_matches
                and list(existing_citations) == citations
                and existing.shape == expected_shape
                and existing.dtype == np.float16
            ):
                embeddings = existing
                start_row = int(status.get("rows_done", 0))
                start_row = max(0, min(start_row, len(docs)))
                start_row -= start_row % max(1, batch_size)
                print(f"[law-chunks] resuming existing matrix at row {start_row:,}/{len(docs):,}", flush=True)
            else:
                print("[law-chunks] existing matrix/config does not match current chunks; rebuilding", flush=True)
        except Exception as exc:
            print(f"[law-chunks] resume check failed ({type(exc).__name__}: {exc}); rebuilding", flush=True)

    if embeddings is None:
        cites_out.write_text(json.dumps(citations, ensure_ascii=False), encoding="utf-8")
        embeddings = np.lib.format.open_memmap(
            emb_out,
            mode="w+",
            dtype=np.float16,
            shape=expected_shape,
        )

    for start in range(start_row, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        encoded = encoder.encode(batch, prefix="passage: ", batch_size=batch_size, max_length=max_length, verbose=False)
        stop = start + encoded.shape[0]
        embeddings[start:stop] = encoded.astype(np.float16)
        if stop % max(batch_size * 20, 1000) < batch_size or stop == len(docs):
            embeddings.flush()
            status_path.write_text(
                json.dumps(
                    {
                        "rows_total": len(docs),
                        "rows_done": stop,
                        "dim": dim,
                        "chunk_words": chunk_words,
                        "overlap_words": overlap_words,
                        "max_length": max_length,
                        "corpus_sha256": signature,
                        "elapsed_sec": round(time.time() - t0, 1),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[law-chunks] checkpoint {stop:,}/{len(docs):,}", flush=True)
    embeddings.flush()
    mb = emb_out.stat().st_size / 1e6
    unique_citations = len(set(citations))
    print(
        f"[law-chunks] wrote {emb_out} rows={len(citations):,} "
        f"unique_citations={unique_citations:,} size={mb:.1f} MB "
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="download models and build/export all dense assets")
    parser.add_argument("--download-embedding", action="store_true")
    parser.add_argument("--download-reranker", action="store_true")
    parser.add_argument("--build-law-dense", action="store_true")
    parser.add_argument("--build-law-chunk-dense", action="store_true")
    parser.add_argument("--build-compact-court", action="store_true")
    parser.add_argument("--build-expanded-court", action="store_true")
    parser.add_argument("--export-compact-court-npy", action="store_true")
    parser.add_argument("--export-expanded-court-npy", action="store_true")
    parser.add_argument("--embedding-repo", default=DEFAULT_EMBED_REPO)
    parser.add_argument("--reranker-repo", default=DEFAULT_RERANK_REPO)
    parser.add_argument("--embedding-dir", type=Path, default=EMBED_DIR)
    parser.add_argument("--reranker-dir", type=Path, default=RERANK_DIR)
    parser.add_argument("--compact-out", type=Path, default=COMPACT_OUT)
    parser.add_argument("--compact-emb-out", type=Path, default=COMPACT_EMB_OUT)
    parser.add_argument("--compact-cites-out", type=Path, default=COMPACT_CITES_OUT)
    parser.add_argument("--expanded-court-out", type=Path, default=EXPANDED_COURT_OUT)
    parser.add_argument("--expanded-court-emb-out", type=Path, default=EXPANDED_COURT_EMB_OUT)
    parser.add_argument("--expanded-court-cites-out", type=Path, default=EXPANDED_COURT_CITES_OUT)
    parser.add_argument("--law-emb-out", type=Path, default=LAW_EMB_OUT)
    parser.add_argument("--law-cites-out", type=Path, default=LAW_CITES_OUT)
    parser.add_argument("--law-chunk-emb-out", type=Path, default=LAW_CHUNK_EMB_OUT)
    parser.add_argument("--law-chunk-cites-out", type=Path, default=LAW_CHUNK_CITES_OUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-docs", type=int, default=30000)
    parser.add_argument("--expanded-court-max-docs", type=int, default=80000)
    parser.add_argument("--expanded-court-max-per-base", type=int, default=20)
    parser.add_argument("--expanded-court-max-per-prefix", type=int, default=14000)
    parser.add_argument("--expanded-court-heap-per-prefix", type=int, default=40000)
    parser.add_argument("--max-law-docs", type=int, default=0, help="0 means all law rows")
    parser.add_argument("--law-chunk-words", type=int, default=320)
    parser.add_argument("--law-chunk-overlap", type=int, default=80)
    parser.add_argument("--law-chunk-max-length", type=int, default=512)
    parser.add_argument("--no-resume-law-dense", action="store_true", help="rebuild law dense matrix from row 0")
    parser.add_argument("--no-resume-law-chunk-dense", action="store_true", help="rebuild law chunk dense matrix from row 0")
    args = parser.parse_args()

    if args.all or args.download_embedding:
        download_model(args.embedding_repo, args.embedding_dir)
    if args.all or args.download_reranker:
        download_model(args.reranker_repo, args.reranker_dir)
    if args.all or args.build_law_dense:
        if not args.embedding_dir.exists():
            raise SystemExit(f"Embedding model missing: {args.embedding_dir}; run --download-embedding first")
        build_law_embeddings(
            args.embedding_dir,
            args.law_emb_out,
            args.law_cites_out,
            args.batch_size,
            args.max_law_docs,
            not args.no_resume_law_dense,
        )
    if args.all or args.build_law_chunk_dense:
        if not args.embedding_dir.exists():
            raise SystemExit(f"Embedding model missing: {args.embedding_dir}; run --download-embedding first")
        build_law_chunk_embeddings(
            args.embedding_dir,
            args.law_chunk_emb_out,
            args.law_chunk_cites_out,
            args.batch_size,
            args.max_law_docs,
            args.law_chunk_words,
            args.law_chunk_overlap,
            args.law_chunk_max_length,
            not args.no_resume_law_chunk_dense,
        )
    if args.all or args.build_compact_court:
        if not args.embedding_dir.exists():
            raise SystemExit(f"Embedding model missing: {args.embedding_dir}; run --download-embedding first")
        build_compact_court_embeddings(args.embedding_dir, args.compact_out, args.batch_size, args.max_docs)
    if args.all or args.export_compact_court_npy:
        export_compact_court_npys(args.compact_out, args.compact_emb_out, args.compact_cites_out)
    if args.build_expanded_court:
        if not args.embedding_dir.exists():
            raise SystemExit(f"Embedding model missing: {args.embedding_dir}; run --download-embedding first")
        build_expanded_court_embeddings(
            args.embedding_dir,
            args.expanded_court_out,
            args.batch_size,
            args.expanded_court_max_docs,
            args.expanded_court_max_per_base,
            args.expanded_court_max_per_prefix,
            args.expanded_court_heap_per_prefix,
        )
    if args.export_expanded_court_npy:
        export_compact_court_npys(args.expanded_court_out, args.expanded_court_emb_out, args.expanded_court_cites_out)


if __name__ == "__main__":
    main()
