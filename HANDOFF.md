# Handoff: Swiss Legal Retrieval — Path to First Place

**Audience:** Codex (or any engineer) picking up this project
**Date:** 2026-04-07
**Author:** Previous AI collaborator + wfrancis

---

## 1. TL;DR

- **Competition:** [LLM Agentic Legal Information Retrieval](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval)
- **Deadline:** 2026-05-24 (~7 weeks remaining)
- **Prize:** $10,000
- **Current best:** **V7b at 24.46%** public LB, rank ~11/301
- **Leader:** 35.9% (Kanak Raj)
- **Gap to first:** 11.4 points
- **User:** `wbfranci` on Kaggle
- **Project root:** `~/swiss-legal-retrieval/`

**Critical fact:** This is **NOT a code competition** (`isKernelsSubmissionsOnly=False`). Submissions are CSV files, not notebooks. This means:
- No GPU/runtime constraints
- No internet restrictions
- You can use GPT-5.4 as much as you want
- Most competitors are probably self-limiting with smaller local models → **this is our moat**

---

## 2. Task Definition

**Input:** 40 test queries in English. Each is a detailed legal question (1000-1800 chars) about Swiss law.

**Output:** For each query, a set of citations a Swiss Federal Supreme Court (BGer) decision would cite. Format:
```
query_id,predicted_citations
test_001,Art. 221 Abs. 1 StPO;BGE 137 IV 122 E. 4.2;1B_210/2023 E. 4.1
```

**Citation types:**
1. Law articles: `Art. 221 Abs. 1 lit. b StPO` (statute abbreviations: StPO, StGB, ZGB, OR, BGG, BV, ATSG, IVG, SchKG, ZPO, IPRG, StBOG, etc.)
2. Leading cases: `BGE 137 IV 122 E. 4.2` (volume, roman, page, consideration)
3. Unreported cases: `1B_210/2023 E. 4.1`

**Metric:** Macro F1. For each query, compute precision and recall over predicted vs gold citation sets, F1 per query, then average across queries.

**Corpus:**
- `laws_de.csv` — 175,933 Swiss law articles in German (73MB)
- `court_considerations.csv` — 2,476,315 court considerations in German (2.4GB, 1.99M unique citations after dedup)
- `train.csv` — 1,139 queries with gold citations (median 2 per query, max 44)
- `val.csv` — 10 queries with gold (avg **25.1** per query — very different from train!)
- `test.csv` — 40 queries, no gold (avg ~25 based on top team's 35.9% score)

**Gold citation distribution on val:**
- ~60% law articles (mostly ZGB, OR, StGB, BV)
- ~40% court decisions (mix of BGE and unreported)

---

## 3. Current Pipeline (V7b, the best)

File: `run_val_eval_v7.py` and `gen_test_submission_v7.py`

```
English query
    │
    ├─ GPT-5.4 ensemble (3 runs, union of predictions)
    │   ├─ val_full_citations.json (gpt-4.1, temp=0.3)
    │   ├─ val_full_citations_v2.json (gpt-5.4, temp=0.2)
    │   └─ val_full_citations_v3.json (gpt-5.4, temp=0.5)
    │
    ├─ Query expansions (GPT-5.4): German keywords, BM25 queries, estimated_citation_count
    │   └─ val_query_expansions.json
    │
    ├─ Law dense retrieval: multilingual-e5-large + FAISS top-200 (full English query)
    ├─ Law BM25: German keyword tokenization, 80 hits per BM25 query
    ├─ Explicit citation regex from query text
    └─ Co-citation expansion (base case → siblings at 0.30)
        │
        ▼
    Merge scores (take max per citation)
        │
        ▼
    Multi-source agreement boost (2+ sources: ×1.25; 3+: ×1.35)
        │
        ▼
    Fuzzy matching (non-corpus GPT predictions → closest corpus match)
        │
        ▼
    Smart cutoff (GPT estimated_citation_count, 15-40 range)
        │
        ▼
    Output: ~28 predictions per query avg
```

**Scoring scale:**
- Explicit query refs: 0.95
- GPT specific articles (from expansions): 0.92
- GPT full citation (in corpus): 0.88-0.95 (freq-weighted)
- GPT fuzzy matched: 0.85-0.90
- Dense top-10: up to 0.84
- BM25 normalized: up to 0.65
- Dense other: up to 0.65
- Co-citation siblings: 0.30 (too low in V7b — V8 attempts to fix)

**Cutoff logic:**
```python
target = gpt_estimate  # from query_expansions (coarse: usually 18 or 35)
if high_conf_count > target:
    target = high_conf_count
target = max(target, 10)
target = min(target, gpt_estimate + 8)
target = min(target, 40)
```

---

## 4. Version History & Key Learnings

### V1-V5 (Abandoned)
- BM25-only, dense-only, with cross-encoder reranker
- Reranker (`bge-reranker-v2-m3`) **did not work** for Swiss legal text — scored wrong things
- V5 best: unclear, discarded

### V6 — 15.86% Kaggle (Baseline)
- Combined dense + BM25 + GPT-4.1 full citations + case citations + explicit refs
- Always predicted 40 citations (hit cap on every query)
- Used `gen_full_citations.py` with `gpt-4.1` (not 5.4!)

### V7 — 24.39% Kaggle (+8.53)
**Massive jump.** Three changes:
1. **Fuzzy matching** for GPT predictions. If GPT says `Art. 221 Abs. 1 lit. b StPO` but corpus has `Art. 221 Abs. 1 StPO`, match them via `(statute, article_number)` lookup.
2. **Multi-run GPT ensemble**: V1 (GPT-4.1) + V2 (GPT-5.4). Union recall is ~38% vs single run ~25%.
3. **Smart cutoff**: Use `estimated_citation_count` from query expansions instead of always 40. Dropped avg predictions from 40 → 28.

### V7b — 24.46% Kaggle (+0.07)
- Added V3 (GPT-5.4 temp=0.5) to the ensemble. Diminishing returns.
- Current best public LB score.

### V8 (WIP, regressed to 22.80% val)
- **Court FAISS dense retrieval** via OpenAI `text-embedding-3-small` (512-dim, 1.99M records, 4GB)
- **Bug 1:** First used full English query → 0/102 gold court recall. Language mismatch.
- **Bug 2:** Then used German keywords → 22/102 recall (working!) but pipeline didn't benefit
- **Bug 3:** Court dense hits score 0.42-0.68 but laws score 0.80+, so courts get pushed out of cutoff
- **Bug 4:** Budget approach (law cap + court cap) made it worse (50-100 predictions per query)
- **Status:** Court FAISS index built and working. Pipeline integration needs a proper rethink.

---

## 5. Model Benchmarks (Already Run)

Tested on 10 val queries with same one-shot prompt (no retrieval augmentation):

| Model | Macro F1 | Corpus Hit Rate | Cost/10q | Notes |
|-------|---------|-----------------|----------|-------|
| **GPT-5.4** | **16.24%** | **82.1%** | $0.12 | **Best** |
| Sonnet 4.6 | 15.57% | 51.2% | $0.18 | Close 2nd |
| GPT-5.4 Mini | 13.17% | 38.1% | $0.03 | Poor corpus format |
| DeepSeek V3.2 | 3.59% | 39.1% | $0.02 | Generated 568 hallucinations/query |

**Use GPT-5.4 for anything accuracy-critical.** Mini and DeepSeek are cheaper but fall apart on Swiss legal citation format. Sonnet 4.6 is close to GPT-5.4 in F1 but much worse on corpus hit rate (citation format accuracy).

**Pricing (as of 2026-04):**
- GPT-5.4: $2.50/M input, $15.00/M output
- GPT-5.4 Mini: $0.75/M input, $4.50/M output
- text-embedding-3-small: $0.02/M tokens
- Budget spent so far: ~$10 total

---

## 6. Error Analysis (What Makes Up the 11.4-Point Gap)

Counterfactual analysis on val (current pipeline = 21.9%):

| Scenario | Val F1 | Gain vs current |
|----------|--------|-----------------|
| **Current (V7-ish)** | 21.9% | — |
| + Perfect court recall | **42.8%** | **+20.9%** |
| + Zero false positives | **42.8%** | **+20.9%** |
| + Both | 73.9% | +52.0% |

**Translation:** We have TWO equal-value levers, each worth ~20 F1 points:
1. **Find all gold court citations** (court recall problem)
2. **Stop predicting things that aren't gold** (false positive problem)

### Specific misses on val (from error analysis):
- **val_001** (gold=42, TP=8): missed 10 procedural laws (Art. 422 StPO, Art. 37/39 StBOG) + 21 court citations
- **val_002** (gold=36, TP=10): missed 10 substantive laws (ATSG/IVG) + 14 court citations
- **val_008** (gold=29, TP=5): missed 7 procedural + 10 substantive + 7 court
- **val_010** (gold=25, TP=5): missed 4 procedural + 6 substantive + 10 court

### Court citation stats:
- 102 gold court citations total on val
- V7b finds: ~16/102 (~16%)
- GPT exact match (all 3 runs union): 23.6% (22/102 unique gold found exact)
- Base case match (GPT predicted BGE 137 IV 122, gold is E. 4.2): 60% of GPT predictions are exact, 17% base-match
- **Max theoretical court recall from GPT alone even with optimal sibling expansion: 37.6%** (35/93)

### Court dense retrieval stats:
- English query embedding: **0/102** gold court recall at top-1000
- German keywords (bm25_queries_court): **22/102** at top-500 (21.6%)
- Multi-query union: **24/102** at top-500 (23.5%)

**The dense retrieval finds gold courts but the pipeline isn't including them.**

---

## 7. Three Concrete Paths to First Place

### Path A: Fix V8 Court Integration (Most likely to work, ~1-2 days)

The court FAISS is already built (`index/faiss_court_openai.index`, 4GB, 1.99M records, $4 sunk cost). German keyword search finds 22/102 gold courts at top-500. The pipeline needs to use these properly without displacing correct laws.

**Specific fix:**
1. Keep V7b's law scoring exactly as-is
2. Add court dense retrieval as a SEPARATE "court-only" candidate set, not merged into main score
3. Use GPT `estimated_citation_count` to target N total predictions
4. Allocate: top `0.55*N` laws from V7b score + top `0.45*N` courts from COURT-ONLY scoring
5. Court-only scoring = max(GPT exact match, GPT sibling expansion, court dense rank score)

**Pitfalls to avoid:**
- DO NOT boost court sibling scores above 0.80 — they flood predictions
- DO NOT use the full English query for court dense — use `bm25_queries_court` from `val_query_expansions.json` (already precomputed, key `bm25_queries_court`)
- DO NOT merge court and law scores into one sorted list — laws will always win
- DO use a hard total cap (32-40 predictions max)

**Expected outcome:** If this works, court recall goes from 16% → 30-35%, adding ~3-5 F1 points. New Kaggle score: ~28-30%.

### Path B: LLM-as-Judge with Retrieval Context (Higher ceiling, more work)

The insight: GPT-5.4 alone (one-shot) gets 16%. Adding retrieval bumps to 24%. The leap to 35%+ probably requires **GPT-5.4 looking at actual retrieval candidates and picking**, not just predicting from memory.

**Critical:** We tried this in V7 as a "reranker" and it HURT (topical match ≠ citation relevance). But the failure was in how we asked. The right question is not "is this topically relevant?" but **"would a Federal Supreme Court decision answering the question cite this specific article?"**

**Architecture:**
```python
For each query:
    1. Retrieve top 100 law candidates (dense + BM25)
    2. Retrieve top 100 court candidates (court FAISS with German keywords)
    3. Get GPT precompute (3-run ensemble)
    4. Build a "menu" of ~200 candidates with FULL TEXT
    5. Send to GPT-5.4:
       "Here is a Swiss legal question. Here are candidate citations with their
        actual corpus text. Return the list of citations that a Swiss Federal
        Supreme Court decision answering this question would cite. Be strict -
        only include if you are confident. Also estimate total citations needed."
    6. Parse response as JSON {citations: [...], count: N}
    7. Use GPT's response as final predictions
```

**Cost:** ~40 queries × ~50K input tokens × $2.50/M + ~2K output × $15/M ≈ **$5.25 per full run**. Negligible.

**Key design decisions:**
- Candidate text: truncate to 150-200 chars each to fit 200 in one context window
- Prompt MUST emphasize "what would ACTUALLY be cited" not "what's related"
- Include a few-shot example from training data showing a question and its actual citations
- Temperature 0.1-0.3 for consistency

**Expected outcome:** If prompt is tuned right, this could reach 30-35% F1. The difference between this and the failed V7 attempt is that GPT is now the PRIMARY decision-maker with retrieval as context, not a rubber stamp on retrieval's top-K.

### Path C: Training Data Few-Shot (Cheap, incremental)

We have 1,139 training queries with gold citations. Barely used them.

**Approach:**
1. Embed all 1,139 training queries
2. For each test query, find top 5 most similar training queries by embedding
3. Include those similar queries + their gold citations as few-shot examples in the GPT prompt
4. This teaches GPT "queries LIKE this usually cite THESE patterns"

**Caveat:** Train distribution is different — median 2 citations per query vs val's 25. Filter training data to only use queries with ≥10 citations (~38 queries total with 21+) to match the complexity.

**Expected outcome:** +2-4% F1. Cheap to try.

---

## 8. Recommended Sequence for Codex

**Week 1 — Path A (V8 court fix)**
1. Clone repo, setup env (see Section 9)
2. Read V7 pipeline carefully (`run_val_eval_v7.py`)
3. Re-read V8 and understand why it regressed (`run_val_eval_v8.py`)
4. Rebuild V8 from V7 base with the "separate court scoring track" approach
5. Iterate on val until val F1 > 25%
6. Generate test submission, submit to Kaggle
7. **Goal: 28%+ public LB**

**Week 2 — Path B (LLM-as-Judge)**
1. Build a new `run_val_eval_v9.py` that sends candidates to GPT-5.4 for judgment
2. Iterate on prompt (this is where the real work is)
3. Few-shot with 2-3 training examples showing complex queries
4. **Goal: 32%+ public LB**

**Week 3-4 — Path C + ensembling**
1. Add training data few-shot to V9
2. Ensemble V8 + V9 predictions (intersection for high confidence, union for recall)
3. Fine-tune cutoffs per-query using GPT to estimate exact count
4. **Goal: 36%+ (first place)**

---

## 9. Environment Setup

### Clone and install
```bash
git clone https://github.com/wfrancis/swiss.git swiss-legal-retrieval
cd swiss-legal-retrieval

# Python 3.9+ required
pip install openai anthropic faiss-cpu sentence-transformers rank-bm25 \
    python-dotenv kaggle numpy
```

### Create `.env` (never commit this!)
```bash
cat > .env <<EOF
OPENAI_API_KEY=sk-proj-...your-key...
ANTHROPIC_API_KEY=sk-ant-...your-key...   # Only needed if benchmarking
DEEPSEEK_API_KEY=sk-...                    # Only needed if benchmarking
KAGGLE_USERNAME=wbfranci
KAGGLE_KEY=your-kaggle-key
EOF
```

### Download competition data (~2.3GB)
```bash
mkdir -p data
kaggle competitions download -c llm-agentic-legal-information-retrieval -p data/
unzip data/llm-agentic-legal-information-retrieval.zip -d data/
# Expected files: train.csv, val.csv, test.csv, laws_de.csv, court_considerations.csv
```

### Build law indices (~10 min, free)
```bash
python3 index/build_bm25.py         # → index/bm25_laws.pkl (186MB)
python3 index/embed_corpus.py       # → index/faiss_laws.index (720MB)
# Also needs: index/court_citations.pkl (unique court citation list, ~56MB)
# If missing, regenerate from court_considerations.csv
```

### Build court FAISS index (~45 min, $4)
```bash
python3 index/embed_court_openai.py
# Output: index/faiss_court_openai.index (4GB), faiss_court_openai_citations.pkl
# Has checkpointing — safe to resume if killed (every 50k records)
```

### Run GPT precompute (~5 min, $2-3)
```bash
# These already exist in precompute/ but if you need to regenerate:
python3 precompute/gen_query_expansions.py     # → val_/test_query_expansions.json
python3 precompute/gen_case_citations.py       # → val_/test_case_citations.json
python3 precompute/gen_full_citations_v2.py val   # GPT-5.4 run
python3 precompute/gen_full_citations_v2.py test
# Edit the script to change temperature for additional runs (v3)
```

---

## 10. How to Submit to Kaggle

### Validate the pipeline on val first
```bash
python3 run_val_eval_v7.py
# Check that val F1 is ~23-24%
# Reads: precompute/val_*.json, index/*.{index,pkl}, data/val.csv
# Writes: submissions/val_pred_v7.csv
```

### Generate test submission
```bash
python3 gen_test_submission_v7.py
# Writes: submissions/test_submission_v7.csv
# 40 rows, avg 28 predictions per query
```

### Verify format
```bash
head -2 submissions/test_submission_v7.csv
# Expected:
# query_id,predicted_citations
# test_001,Art. 221 Abs. 1 StPO;BGE 137 IV 122 E. 4.2;...

wc -l submissions/test_submission_v7.csv   # Should be 41 (header + 40 rows)
```

### Submit via Kaggle API
```bash
# Method 1: CLI (requires kaggle.json at ~/.kaggle/ OR env vars)
kaggle competitions submit \
    -c llm-agentic-legal-information-retrieval \
    -f submissions/test_submission_v7.csv \
    -m "V7b: 3-run GPT-5.4 ensemble + fuzzy match"

# Method 2: Python SDK (what we've been using — kaggle.json auth is broken in new SDK)
KAGGLE_API_TOKEN=<your-kaggle-key> python3 -c "
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.competition_submit(
    'submissions/test_submission_v7.csv',
    'Description of submission',
    'llm-agentic-legal-information-retrieval'
)
print('Submitted!')
"
```

### Check submission score (after ~30-60s)
```bash
KAGGLE_API_TOKEN=<your-kaggle-key> python3 -c "
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
subs = api.competition_submissions('llm-agentic-legal-information-retrieval')
for s in subs[:5]:
    d = vars(s)
    # NEW SDK uses _public_score, not publicScore
    print(f'{d.get(\"_description\",\"?\")[:60]:60s} | {d.get(\"_public_score\",\"pending\"):<10} | {d.get(\"_status\",\"?\")}')"
```

### Check leaderboard
```bash
KAGGLE_API_TOKEN=<your-kaggle-key> python3 -c "
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
lb = api.competition_leaderboard_view('llm-agentic-legal-information-retrieval')
for i, e in enumerate(lb[:15]):
    print(f'{i+1:3d}. {e.teamName:30s} {e.score}')
"
```

### Submission limits
- **10 submissions per day**
- No limit on total submissions
- Each submission takes ~30-60 seconds to score
- Public LB score is what you see; private LB (final ranking) is on different subset

---

## 11. Key File Reference

### Code (all in repo)
```
run_val_eval_v7.py              # V7b pipeline (23.63% val, 24.46% Kaggle) — BEST
run_val_eval_v8.py              # V8 WIP (22.80% val, regression from V7b)
gen_test_submission_v7.py       # V7b test submission generator
gen_test_submission_v8.py       # V8 test submission generator
benchmark_models.py             # GPT-5.4 vs Mini vs DeepSeek vs Sonnet
evaluate.py                     # Metric implementation
analyze_gold_patterns.py        # Training data analysis

precompute/
├── gen_query_expansions.py     # GPT-5.4: German keywords, BM25 queries, citation count estimate
├── gen_case_citations.py       # GPT-5.4: predict BGE/case citations with Erwägung expansion
├── gen_full_citations.py       # GPT-4.1: complete citation prediction (V1 run)
├── gen_full_citations_v2.py    # GPT-5.4: complete citation prediction (V2, V3 runs)
├── val_query_expansions.json   # 10 val queries, includes estimated_citation_count
├── val_case_citations.json     # GPT-predicted case citations for val
├── val_full_citations.json     # V1 run (GPT-4.1)
├── val_full_citations_v2.json  # V2 run (GPT-5.4, temp=0.2)
├── val_full_citations_v3.json  # V3 run (GPT-5.4, temp=0.5)
├── test_*.json                 # Same structure for test queries
├── domain_templates.json       # Generic Swiss legal domain templates
└── legal_glossary.json         # EN→DE legal term glossary

index/
├── build_bm25.py                    # Builds bm25_laws.pkl from laws_de.csv
├── embed_corpus.py                  # Builds faiss_laws.index with multilingual-e5-large
├── embed_court_openai.py            # NEW: Builds court FAISS with text-embedding-3-small
├── build_court_sharded.py           # OLD: Court BM25 (abandoned, shards found 0 gold)
└── [generated, not in repo]
    ├── bm25_laws.pkl                # 186MB, laws BM25 index
    ├── faiss_laws.index             # 720MB, law dense index
    ├── faiss_laws_citations.pkl     # Parallel citation list
    ├── court_citations.pkl          # 56MB, unique court citation strings
    ├── faiss_court_openai.index     # 4GB, court dense index
    └── faiss_court_openai_citations.pkl  # Parallel citation list

submissions/
├── test_submission.csv          # V2? old
├── test_submission_v6.csv       # V6 — 15.86% Kaggle
├── test_submission_v7.csv       # V7/V7b — 24.46% Kaggle  ← BEST
└── val_pred_v*.csv              # Per-version val predictions
```

### Data (not in repo, download from Kaggle)
```
data/
├── train.csv                    # 1,139 queries, avg 4.1 citations (MEDIAN 2)
├── val.csv                      # 10 queries, avg 25.1 citations
├── test.csv                     # 40 queries, no gold
├── laws_de.csv                  # 175,933 law articles (73MB)
└── court_considerations.csv     # 2,476,315 rows (2.4GB, 1.99M unique citations)
```

---

## 12. Specific Code Gotchas

1. **`gen_full_citations.py` uses `gpt-4.1` not `gpt-5.4`.** This is V1 precompute. Don't "fix" it — we intentionally keep it for ensemble diversity. Use `gen_full_citations_v2.py` for GPT-5.4 runs.

2. **`max_tokens` vs `max_completion_tokens`:** GPT-5.4 requires `max_completion_tokens`. GPT-4.1 accepts `max_tokens`. DeepSeek uses `max_tokens`. The benchmark handles this via a flag.

3. **Kaggle SDK change:** New SDK uses `_public_score` not `publicScore`. Also `kaggle.json` auth is broken — must use `KAGGLE_API_TOKEN` env var.

4. **Anthropic model ID:** Use `claude-sonnet-4-6`, NOT `claude-sonnet-4-6-20250514`. The dated version returns 404.

5. **Court corpus is German, queries are English.** When embedding queries for court dense search, use `bm25_queries_court` from query expansions (German keywords), NOT the raw English query.

6. **Court citation format variance:** GPT predicts `BGE 137 IV 122 E. 4.2` but the same case has E. 1, E. 2, E. 3.1, E. 4.1, E. 4.2, E. 5.1 etc. in the corpus. The `case_prefix_map` maps base → list of all Erwägungen. Use it for sibling expansion, but be careful not to flood predictions.

7. **The "always 40" problem:** If you see all test predictions at exactly 40, your cutoff is broken. V7b averages 28. V6 (broken) was 40.

8. **Fuzzy matching edge case:** Multiple corpus citations may share `(statute, article_number)` — the code picks by `SequenceMatcher.ratio() >= 0.75`. For ambiguous cases, you may need stricter matching or include all candidates.

9. **Live GPT rerank:** Exists in V7/V8 as `live_rerank_candidates()` but is DISABLED. Enabling it HURTS performance by 1-2%. Topical relevance ≠ citation relevance. If you want LLM-as-judge (Path B), write a NEW prompt that explicitly asks about citation likelihood.

10. **Submission file format:** CSV with exactly two columns: `query_id,predicted_citations`. Citations separated by semicolons. No trailing semicolons, no empty predictions (every query must have at least one citation to score).

---

## 13. What Did NOT Work (avoid redoing these)

1. **Cross-encoder reranker (`bge-reranker-v2-m3`):** Doesn't know Swiss legal domain. Scored wrong things. V5 worse than V4.

2. **Court BM25 sharded:** Built 4 shards (2.2GB), found 0 gold citations. Deleted. The shards had BM25 issues — maybe fixable but not worth the time.

3. **CPU court embedding with multilingual-e5-large:** Estimated 65 hours for 2.5M rows. Killed after 8 hours. text-embedding-3-small via OpenAI API is 45 min and $4.

4. **DeepSeek V3.2 for this task:** Hallucinates 568 citations per query on average. 3.59% F1. Don't use.

5. **GPT-5.4 Mini for citation prediction:** 38% corpus hit rate vs 82% for GPT-5.4. Citation format accuracy is critical here — Mini gets it wrong.

6. **Live GPT-5.4 rerank of retrieval candidates:** Approves too many "topically related" laws, hurts precision. See gotcha #9.

7. **Ensemble of 3 GPT runs:** V1+V2 helped a lot (+big). V1+V2+V3 added only +0.07%. Diminishing returns. Don't bother with a 4th run.

8. **V8 court dense with full English query:** 0/102 gold court recall. Use German keywords instead.

9. **V8 budget approach with `law_high_count` floor:** Allowed law budget to explode to 40+ (because many GPT predictions scored ≥0.85). Predictions went to 50-100 per query. Use hard caps.

10. **Court sibling scores at 0.78-0.82:** Displaces good laws from the cutoff. Keep court siblings ≤ 0.50 OR use a separate budget (see Path A).

---

## 14. Cost Tracking

| Item | Cost |
|------|------|
| GPT-5.4 precompute (3 runs × val + test) | ~$2.50 |
| Court corpus embedding (text-embedding-3-small) | $4.19 |
| Model benchmarks (4 models × 10 val queries) | ~$0.50 |
| V8 per-query embedding for live court search | ~$0.01 |
| **Total spent** | **~$7-10** |

Budget plenty of room for: LLM-as-judge ($5-10 per full run × 10 iterations = $50-100). Even at $200 total, we're fine for a $10K prize.

---

## 15. Quick Test: Does Your Environment Work?

```bash
# 1. Can you reach OpenAI?
python3 -c "import os; from openai import OpenAI; from dotenv import load_dotenv; load_dotenv(); c=OpenAI(api_key=os.getenv('OPENAI_API_KEY')); r=c.chat.completions.create(model='gpt-5.4', messages=[{'role':'user','content':'hi'}], max_completion_tokens=10); print('OpenAI OK:', r.choices[0].message.content)"

# 2. Can you load the law FAISS?
python3 -c "import faiss, pickle; idx=faiss.read_index('index/faiss_laws.index'); print(f'Law FAISS: {idx.ntotal:,} vectors')"

# 3. Can you load the court FAISS (if built)?
python3 -c "import faiss; idx=faiss.read_index('index/faiss_court_openai.index'); print(f'Court FAISS: {idx.ntotal:,} vectors')"

# 4. Can you run V7 on val?
python3 run_val_eval_v7.py
# Expected output ending with: "=== V7 MACRO F1: 0.2363 (23.63%) ==="

# 5. Can you submit to Kaggle?
KAGGLE_API_TOKEN=$KAGGLE_KEY python3 -c "
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
print('Kaggle OK:', api.competition_view('llm-agentic-legal-information-retrieval').title)
"
```

---

## 16. Competition Deadline Planning

- **Today:** 2026-04-07
- **Deadline:** 2026-05-24
- **Remaining:** 47 days (enough for multiple full iterations)
- **Submission limit:** 10/day (300+ total possible)

**Suggested cadence:**
- Days 1-5: Fix V8 (Path A), aim for 28%
- Days 6-14: Build V9 LLM-as-judge (Path B), aim for 32%
- Days 15-25: Integrate V9 + training data (Path C), aim for 35%+
- Days 26-40: Ensembling, tuning, edge cases
- Days 41-47: Lock best approach, final submissions (the LAST submission you select counts for private LB)

---

## 17. One Final Note

The 11.4-point gap is real but closeable. We have three things the top team probably doesn't:
1. **Unlimited GPT-5.4 access** — most competitors use small local models
2. **Court FAISS already built** — 4GB index is a significant sunk cost we can exploit
3. **Detailed error analysis** — we know exactly where the gap is (court recall + false positives)

The fix is probably not "build something new from scratch" but "make the existing pieces work together properly." The Path A fix (separate court scoring track) is the single most likely path to a +3-5 point jump, and it's a pure code change with no new API calls.

Good hunting. Make us #1.

— Previous collaborator
