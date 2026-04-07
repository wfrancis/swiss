# Swiss Legal Retrieval — Kaggle Competition

Competing in the [LLM Agentic Legal Information Retrieval](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval) Kaggle competition. The task: given English legal questions, predict the Swiss law articles and Federal Supreme Court (BGer/BGE) decisions a Swiss court would cite when answering.

## Current Status

| Version | Val F1 | Kaggle Public LB | Notes |
|---------|--------|------------------|-------|
| V6 | 20.80% | **15.86%** | Baseline: dense + BM25 + GPT-4.1 full citations |
| V7 | 23.63% | 24.39% | + Fuzzy matching + smart cutoff + GPT-5.4 |
| **V7b** | 23.63% | **24.46%** | + 3-run GPT-5.4 ensemble (V1+V2+V3) |
| V8 (WIP) | 22.80% | — | + Court FAISS index (1.99M records) — needs cutoff tuning |

**Leader:** 35.9% • **Our rank:** ~11/301 • **Deadline:** 2026-05-24

## Pipeline Architecture

```
English query
    │
    ├─ GPT-5.4 multi-run ensemble (V1+V2+V3) → law + court citations
    ├─ Dense retrieval (multilingual-e5-large, FAISS) → top 200 laws
    ├─ BM25 (German tokenization) → top 80 laws per query
    ├─ Court dense (text-embedding-3-small, FAISS, 1.99M records)
    ├─ Explicit citation extraction from query (regex)
    └─ Co-citation expansion via case_prefix_map
        │
        ▼
    Score merging + multi-source agreement boost
        │
        ▼
    Fuzzy matching (statute + article number lookup)
        │
        ▼
    Smart cutoff (estimated_citation_count guided)
        │
        ▼
    Submission CSV
```

## Key Files

- `run_val_eval_v8.py` — current pipeline (V8 with court dense)
- `run_val_eval_v7.py` — V7b best Kaggle (24.46%)
- `gen_test_submission_v8.py` — V8 test submission generator
- `precompute/gen_full_citations_v2.py` — GPT-5.4 multi-round citation generator
- `index/embed_court_openai.py` — Court corpus embedding (with checkpointing)
- `benchmark_models.py` — GPT-5.4 vs Mini vs DeepSeek vs Sonnet 4.6 head-to-head

## Setup

```bash
# Install dependencies
pip install openai anthropic faiss-cpu sentence-transformers rank-bm25 python-dotenv kaggle

# Add API keys to .env (NEVER commit this file)
echo "OPENAI_API_KEY=sk-..." > .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
echo "DEEPSEEK_API_KEY=sk-..." >> .env

# Download data from Kaggle
kaggle competitions download -c llm-agentic-legal-information-retrieval -p data/
unzip data/llm-agentic-legal-information-retrieval.zip -d data/
```

## Building Indices

```bash
# Law corpus indices (BM25 + FAISS, ~10 min, free)
python3 index/build_bm25.py
python3 index/embed_corpus.py

# Court corpus FAISS (~45 min, ~$4 in OpenAI credits)
python3 index/embed_court_openai.py
```

## Running

```bash
# Generate GPT precompute (~$0.75 per run × 3 runs = ~$2.25)
python3 precompute/gen_query_expansions.py
python3 precompute/gen_case_citations.py
python3 precompute/gen_full_citations_v2.py val
python3 precompute/gen_full_citations_v2.py test

# Run val eval
python3 run_val_eval_v8.py

# Generate test submission
python3 gen_test_submission_v8.py

# Submit to Kaggle
kaggle competitions submit -c llm-agentic-legal-information-retrieval \
  -f submissions/test_submission_v8.csv -m "V8 description"
```

## Key Learnings

1. **GPT-5.4 dominates** — 16.24% F1 single-shot on val (vs Mini 13.17%, Sonnet 15.57%, DeepSeek 3.59%)
2. **Corpus hit rate matters** — GPT-5.4 gets 82% of citation formats right vs 38% for Mini
3. **Multi-run ensemble** — Union of 3 GPT runs finds 38% of gold citations (vs 25% single run)
4. **Fuzzy matching** — Recovers ~20% of GPT predictions that fail exact corpus match
5. **Court dense retrieval** — English query embedding gives 0/102 court recall; German keywords give 22/102
6. **Cutoff is everything** — Predicting 28 vs 40 jumped Kaggle from 15.86% → 24.46%
7. **Live GPT reranking failed** — Topical relevance ≠ citation relevance

## Cost Tracking

- Court corpus embedding: $4.19 (one-time)
- GPT-5.4 precompute (3 runs × val + test): ~$2.25
- Per Kaggle submission iteration: ~$0
- Total spent: ~$10
