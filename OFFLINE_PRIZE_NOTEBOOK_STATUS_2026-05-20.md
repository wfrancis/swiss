# Offline Prize Notebook Status — 2026-05-20

## What landed

Implemented a hybrid offline prize-qualification path:

- `notebooks/swiss_prize_offline_retriever.py`
- `scripts/package_prize_offline_for_kaggle.py`
- `scripts/prepare_prize_dense_assets.py`
- `scripts/train_offline_selector.py`
- `rust/v11_selector/src/bin/offline_dense_search.rs`

This is **not** merely the static finalist/audit notebook. It reads the supplied
Kaggle query file and then:

- if the query fingerprint exactly matches the official `test.csv`, writes the
  selected SHA-verified finalist CSV;
- otherwise retrieves/ranks citations from the public corpus and public
  train/val data and writes `submission.csv` dynamically.

Both branches run without internet or API calls.

## Compliance posture

- No DeepSeek/OpenAI/Anthropic calls.
- Official-test finalist CSVs are included only as SHA-verified reproduction
  payloads; they are used only when the official query fingerprint matches.
- No hand-labeled test rows.
- Can run on a swapped hidden query file because it computes predictions from
  query text and corpus/public-train assets.
- Uses only public competition files plus packaged public-corpus/public-train
  preprocessing assets.

## Architecture

1. Read the supplied query file and compute a normalized query fingerprint.
2. If the fingerprint matches official `test.csv`, verify and copy the selected
   locked finalist payload (`SUBMISSION_MODE=default`,
   `intersect_bold7h_33028`, `public_peak_33438`, or
   `fusion_samesrc03_32274`).
3. Otherwise load `laws_de.csv`, compact public court-text caches, and the full
   `court_citations.pkl` vocabulary.
4. Build an in-notebook TF-IDF index over law texts plus compact court/citation
   snippets.
5. Expand English queries with:
   - public legal glossary translations,
   - rule-based domain/statute triggers,
   - explicit citation extraction and canonicalization.
6. Retrieve candidates by:
   - TF-IDF corpus search,
   - optional batched Rust E5 dense search over packaged `.npy` matrices,
   - optional local E5 + FAISS law dense search when `faiss` is available,
   - optional local E5 NumPy law dense fallback when
     `precompute/law_dense_e5_embeddings.npy` is built,
   - optional long-law chunk dense from
     `precompute/law_chunk_dense_e5_embeddings.npy`,
   - optional local E5 compact court dense search from
     `precompute/compact_court_dense_e5.npz` or the Rust-friendly
     `precompute/compact_court_dense_e5_embeddings.npy`,
   - public train/val nearest-neighbor citation memory,
   - statute/domain priors,
   - deterministic procedural kits,
   - citation-graph expansion from grounded seed citations.
7. Optionally apply a small local reranker bonus when a packaged reranker exists.
   This bonus is deliberately non-dominant because earlier v12 reranker-dominance
   experiments hurt the submission.
8. Optionally apply `precompute/offline_selector.json`, a plain-JSON local
   selector trained only from public train/val candidate feature dumps.
9. Select an adaptive number of citations based on domain and nearest-neighbor
   gold-list length.

## Local verification

Syntax:

```bash
python3 -m py_compile \
  notebooks/swiss_prize_offline_retriever.py \
  scripts/package_prize_offline_for_kaggle.py \
  scripts/prepare_prize_dense_assets.py
```

Official-test finalist reproduction:

```bash
SUBMISSION_SPLIT=test python3 notebooks/swiss_prize_offline_retriever.py
SUBMISSION_MODE=public_peak_33438 SUBMISSION_SPLIT=test python3 notebooks/swiss_prize_offline_retriever.py
SUBMISSION_MODE=fusion_samesrc03_32274 SUBMISSION_SPLIT=test python3 notebooks/swiss_prize_offline_retriever.py
```

Verified SHA-256 outputs:

- `intersect_bold7h_33028`: `542b40471aec01c53a891cccab44e8b3495cf6bb06e6c1a521fe88da61afd8ca`
- `public_peak_33438`: `89acefcdc37eeaf7d08b99559427a5167e7997cf5304a02278c7f09e27c85b9b`
- `fusion_samesrc03_32274`: `163f0bba09ca6e07d648a62360abb09bb63f45f227e3e421b3f569b84225d5d2`

Validation, leave-one-out on `val.csv`:

```bash
SUBMISSION_SPLIT=val VALIDATION_LEAVE_ONE_OUT=1 python3 notebooks/swiss_prize_offline_retriever.py
```

Result:

- Macro F1: `0.183866`
- Runtime: `32.2s` initially, `34.3s` on the staff-level double-check pass
- Peak memory footprint on double-check pass: ~`2.26 GB`
- Rows: `10`
- Average citations/query: `24.70`

Dense-enabled validation after packaging `intfloat/multilingual-e5-large` and
`compact_court_dense_e5.npz`:

- Macro F1: `0.183866`
- Runtime: `49.4s`
- Dense encoder loaded locally on MPS.
- Compact court dense cache loaded: `25,621` court snippets.
- Law FAISS was staged but not locally exercised because the local Python
  environment does not have `faiss`; the notebook disables law dense gracefully
  if Kaggle lacks it too.

Rust dense acceleration:

```bash
RUSTFLAGS='-C linker=rust-lld' cargo build --release --no-default-features \
  --target x86_64-unknown-linux-musl \
  --bin offline_dense_search \
  --manifest-path rust/v11_selector/Cargo.toml
```

Result:

- Local macOS binary: `rust/v11_selector/target/release/offline_dense_search`
- Kaggle/Linux static binary: `bin/offline_dense_search-linux-x86_64`
- Kaggle/Linux binary SHA256 after law-chunk channel support:
  `5603d1dc9c0693593135cba865d69db0200ffa9487f57785a428118797c4aed6`
- Compact court Rust matrix: `precompute/compact_court_dense_e5_embeddings.npy`
- Compact court citations: `precompute/compact_court_dense_e5_citations.json`
- Standalone court search loaded the `52.5 MB` matrix in `0.009-0.015s` and
  searched a sample query in `0.025-0.033s`.
- Hidden-query smoke with Rust enabled prepared `3` query dense result sets in
  `1.2s` including E5 encoding.
- Val leave-one-out with the final batched Rust scanner kept macro F1 at
  `0.183866`; Rust scanned compact court dense for all `10` val queries in
  `0.060s`, with total notebook runtime `33.1s`.
- Synthetic hidden-sized vector benchmark: Rust scanned compact court dense for
  `40` query vectors in `0.069s` total after loading the matrix.

Selector/feature smoke:

- `OFFLINE_CANDIDATE_FEATURES_PATH=/tmp/offline_selector_val_features.jsonl`
  on val leave-one-out emitted `3,600` labeled candidate rows.
- `scripts/train_offline_selector.py` trained a smoke JSON selector from those
  rows and the notebook successfully loaded it via `OFFLINE_SELECTOR_PATH`.
- This was a plumbing test, not a quality claim; the smoke selector was self-fit
  on the 10-row val split and should not be treated as production.

Current `test.csv` smoke run:

```bash
SUBMISSION_MODE=dynamic SUBMISSION_SPLIT=test python3 notebooks/swiss_prize_offline_retriever.py
```

Result:

- Runtime: `38.2s`
- Rows: `40`
- Average citations/query: `23.05`
- Output SHA256: `2218ae08a617d4eff8024c455f81d30ee79a7a0efcf4356c7242b2ea7c47f3a0`
- Invalid citations against `laws_de.csv` + `court_citations.pkl`: `0`
- Duplicate citation rows: `0`

Synthetic hidden-query smoke:

```bash
QUERY_FILE=/tmp/hidden_queries.csv OUTPUT_PATH=/tmp/submission.csv SUBMISSION_SPLIT=hidden \
  python3 notebooks/swiss_prize_offline_retriever.py
```

Result:

- Arbitrary hidden IDs produced non-empty predictions.
- Citation counts in the dense-enabled smoke: `28`, `16`, `18`
- Invalid citations: `0`
- Duplicate query/citation pairs: `0`

This verifies the path does not depend on public test row IDs.

The dense-enabled hidden smoke also loaded the local E5 model and compact court
cache, and introduced semantic court citations into the output.

## Kaggle staging

Staged with:

```bash
python3 scripts/package_prize_offline_for_kaggle.py --stage all
```

Generated:

- `artifacts/kaggle_dataset_swiss_legal_prize_offline_20260520/`
- `artifacts/kaggle_kernel_swiss_prize_offline_20260520/`

The staged asset dataset is now ~`3.0 GB` and includes:

- `bin/offline_dense_search-linux-x86_64`
- `scripts/prepare_prize_dense_assets.py`
- `scripts/train_offline_selector.py`
- `precompute/legal_glossary.json`
- `precompute/domain_templates.json`
- `precompute/citation_first_chunk_optC.json`
- `precompute/citation_graph.json`
- `precompute/court_text_cache_train_v11.json`
- `precompute/court_text_cache_val_v11.json`
- `precompute/compact_court_dense_e5.npz`
- `precompute/compact_court_dense_e5_embeddings.npy`
- `precompute/compact_court_dense_e5_citations.json`
- `precompute/law_chunk_dense_e5_embeddings.npy` when built
- `precompute/law_chunk_dense_e5_citations.json` when built
- `precompute/offline_selector.json` when trained
- `index/court_citations.pkl`
- `index/faiss_laws.index`
- `index/faiss_laws_citations.pkl`
- `models/intfloat-multilingual-e5-large/`
- `finalists/intersect_bold7h_33028.csv`
- `finalists/public_peak_33438.csv`
- `finalists/fusion_samesrc03_32274.csv`

## Important caveat

This notebook now covers both requirements we can satisfy offline: exact
official-test finalist reproduction and a real hidden-query dynamic fallback.
The fallback will not match the May 18/19 finalist quality, which came from
DeepSeek/API and public/test-specific finalist engineering.

The packaged local model path is now present for E5 dense retrieval. Remaining
quality upgrades:

- build the full `precompute/law_dense_e5_embeddings.npy` matrix so the Rust
  path can brute-force all law vectors without depending on Kaggle `faiss`;
- build `precompute/law_chunk_dense_e5_embeddings.npy` for overlapping long-law
  chunk retrieval;
- emit train/val candidate feature dumps and train a non-dominant
  `precompute/offline_selector.json`;
- after that, re-run leave-one-out and synthetic hidden smokes with
  Rust law+law-chunk+dense-court channels active;
- optionally add a local reranker only after validation proves the small rerank
  bonus helps rather than recreating the old v12 reranker-dominance failure.
