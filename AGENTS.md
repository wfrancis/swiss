# Swiss Legal Retrieval — Kaggle Competition

Competition: [LLM Agentic Legal Information Retrieval](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval)

## Current best

- **Public LB: 0.32107** — `submissions/test_submission_targeted_proc_delta_balanced_swap.csv`
- Submitted 2026-04-23
- Produced by: `scripts/targeted_procedural_deltas.py` (`balanced_swap`) — targeted procedural/boilerplate deltas on the 0.30911 base with a small swap/removal component
- Val companion: `submissions/val_pred_targeted_proc_delta_balanced_swap.csv`
- Frozen baseline: `submissions/test_submission_baseline_public_best_32107.csv`
- Prior high: 0.30911 (`test_submission_baseline_public_best_30911.csv`) — still valid as a stable public/solution-repro reference
- Conservative hedge: 0.30257 (`test_submission_baseline_public_best_30257.csv`) — still valid as a lower-churn final hedge candidate

## HARD CONSTRAINT: Public leaderboard overfitting

**The public leaderboard uses only ~50% of test queries. Final standings use the OTHER 50%.** Every submission where we react to the public score increases overfitting to those ~20 public queries. This is the #1 risk in Kaggle competitions — teams at rank 5 public can drop to rank 50 private.

## HARD CONSTRAINT: Prize reproducibility assets

We are in prize contention. Do **not** delete or overwrite prize-repro assets.
Read `PRIZE_REPRO_DO_NOT_DELETE.md` before any cleanup. Preserve the offline
notebook/code path, competition data, core indices, precompute caches, frozen
candidate CSVs, val companions, and documentation needed to reproduce and
explain the selected submission. In particular, keep:

- `notebooks/swiss_submission_v12.py`, `notebooks/swiss_submission.py`, and `notebooks/swiss_submission.ipynb`
- `scripts/package_v12_for_kaggle.py` and `scripts/targeted_procedural_deltas.py`
- `data/train.csv`, `data/val.csv`, `data/test.csv`, `data/laws_de.csv`, `data/court_considerations.csv`
- `index/bm25_laws.pkl`, `index/faiss_laws.index`, `index/faiss_laws_citations.pkl`, `index/court_citations.pkl`
- `precompute/` especially `precompute/llm_procedural_cache.json`
- frozen/top candidate CSVs and val companions listed in `PRIZE_REPRO_DO_NOT_DELETE.md`

Verified after cleanup on 2026-04-28:

```bash
SUBMISSION_MODE=v12_repro_32107 python3 notebooks/swiss_submission_v12.py
SUBMISSION_MODE=v12_repro_30911 python3 notebooks/swiss_submission_v12.py
```

Both reproduced their frozen test payloads byte-for-byte.

**Rules (non-negotiable):**

1. **Do NOT use the public LB score as a gradient.** Do not iterate by submitting → checking score → adjusting. Each such cycle overfits to the public 50%.
2. **Minimize total Kaggle submissions.** Every submission is a data leak from the public test set into our decision-making. Only submit when local evaluation gives strong, diverse signal — not to "check if it works."
3. **Never submit more than 2 candidates per experimental direction.** If the first two don't improve, the direction is wrong — don't keep submitting variations.
4. **Trust local diversity over public score.** A candidate that's strong across diverse val queries with stable bootstrap confidence is more likely to survive the private shakeup than one hill-climbed on public feedback.
5. **Final submission selection must hedge.** This competition allows **2 final submissions** for private scoring. Pick two legs that fail in different ways: (a) highest public score / aggressive; (b) stable conservative anchor (high Jaccard, minimal changes). Do NOT select two near-duplicates of the top public score.
6. **Never overwrite or delete frozen baselines.** Keep `test_submission_baseline_public_best_32107.csv`, `test_submission_baseline_public_best_30911.csv`, `test_submission_baseline_public_best_30681.csv`, and `test_submission_baseline_public_best_30257.csv` intact as hedge/reference candidates.

**Why 0.32107 is exposed:** It was selected after public-LB pressure and adds/removes targeted procedural deltas relative to the 0.30911 base. If those changes helped mostly on the ~20 public queries, private score can drop. The 0.30911 and 0.30257 baselines remain essential final-submission hedges.

## Competition rules and compliance

- **Submission limit:** max 5 submissions/day.
- **Final submissions:** max 2 selected for judging.
- **Account/team:** use one Kaggle account only; max team size 5; no private sharing outside the team.
- **No hand labeling/prediction of val or test records.** Do not manually label hidden/public/private test rows. Local val gold can be used for evaluation, but do not create human-derived labels for val/test queries.
- **Competition data use:** non-commercial competition/research use only. Do not redistribute competition data or derived labels outside permitted Kaggle/team channels.
- **External data/tools:** allowed only if publicly/equally accessible at minimal/reasonable cost, and must be documented for winner reproducibility.
- **Winner obligations:** code and docs must reproduce the selected final submission; winning code must be released under Apache-2.0 or another OSI-approved permissive license unless an allowed third-party dependency exception applies.

## Testing workflow — how to tell better from worse

**Local val is unreliable on its own.** The val set has only 10 queries with ~5 gold cites each. A single citation flip = ~2pp F1 swing.

**Key learning (2026-04-10):** The old promotion gate was too conservative. combo_a had 82% Jaccard (gate said "likely_worse") but scored 0.30681 on Kaggle — our biggest jump ever. The Jaccard > 0.90 rule was wrong. What matters more: val F1 lift AND LB90 lift together, not Jaccard alone. Candidates with both metrics up are worth submitting even at moderate Jaccard.

### Required evaluation steps for any new candidate

1. **Generate val + test CSVs** for your candidate variant.

2. **Run the multi-signal scorecard** against the frozen baseline:
   ```bash
   python3 scripts/multi_signal_scorecard.py \
       --val-gold data/val.csv \
       --reference-test submissions/test_submission_baseline_public_best_32107.csv \
       --reference-val submissions/val_pred_baseline_public_best_32107.csv \
       --variant CANDIDATE=submissions/val_pred_CANDIDATE.csv,submissions/test_submission_CANDIDATE.csv
   ```
   This reports 5 signals: raw val F1, bootstrap LB90, per-query std, test shape vs baseline, Jaccard overlap.

3. **Run the promotion gate** for a better/worse/unclear verdict:
   ```bash
   python3 promotion_gate.py \
       --candidate-val submissions/val_pred_CANDIDATE.csv \
       --candidate-test submissions/test_submission_CANDIDATE.csv
   ```
   Uses real Kaggle history to predict whether the candidate is likely better or worse.

4. **Run the submission scorecard** for uncertainty analysis:
   ```bash
   python3 submission_scorecard.py \
       --val-csv submissions/val_pred_CANDIDATE.csv \
       --test-csv submissions/test_submission_CANDIDATE.csv \
       --ref-test baseline=submissions/test_submission_baseline_public_best_32107.csv
   ```

5. **Run the diverse eval (Rust)** for per-bucket F1, test Jaccard, churn balance:
   ```bash
   cargo build --release --bin diverse_eval --manifest-path rust/v11_selector/Cargo.toml
   ./rust/v11_selector/target/release/diverse_eval \
       submissions/val_pred_CANDIDATE.csv \
       submissions/val_pred_baseline_public_best_32107.csv \
       submissions/test_submission_CANDIDATE.csv \
       submissions/test_submission_baseline_public_best_32107.csv
   ```
   Slices val into buckets (proceeding/domain/size/length) and reports per-bucket
   F1 with bootstrap 90% CI + per-bucket test churn + overall Jaccard + **churn
   balance** (pure-additive candidates flagged — this catches the claude_agree
   failure mode where val looks great but test adds 258 cites + removes 0 →
   precision collapse on Kaggle). Verdict tiers: STRONG PROMOTE / PROMOTE /
   STRONG-BUT-RISKY / HOLD / REJECT.

### Promotion rules — updated 2026-04-10

- Raw val F1 AND LB90 both improved vs current best (not just one), measured against the frozen 0.32107 companion unless deliberately evaluating a hedge/diversifier against 0.30911 or 0.30257
- Per-query F1 std is not elevated vs winner
- Promotion gate verdict is advisory only — it was wrong on combo_a (0.30681). Do NOT blindly trust it.
- Jaccard > 0.90 is NOT required (combo_a won at 0.82). But lower Jaccard = higher private-shakeup variance.
- Before submitting: check if this is worth burning a submission on (see overfitting constraint above)

### What to avoid

- Do NOT use Kaggle submissions as a search loop. Iterate locally, submit sparingly.
- Do NOT trust a single raw val F1 number. A +2pp local lift means nothing on 10 queries.
- Do NOT overwrite frozen baselines (`test_submission_baseline_public_best_32107.csv`, `test_submission_baseline_public_best_30911.csv`, `test_submission_baseline_public_best_30681.csv`, `test_submission_baseline_public_best_30257.csv`).

## Key files

| File | Purpose |
|------|---------|
| `pipeline_v11.py` | Core V11 retrieval pipeline (dense+BM25+judge) |
| `run_val_eval_v11.py` | Run pipeline on val split |
| `gen_test_submission_v11.py` | Generate test submission |
| `scripts/winner_localperturb_search.py` | The winning perturbation script (hash-verified recovery) |
| `scripts/multi_signal_scorecard.py` | Multi-signal evaluation scorecard |
| `rust/v11_selector/src/bin/diverse_eval.rs` | Per-bucket F1 + test churn + pure-additive flag (Rust) |
| `promotion_gate.py` | Better/worse prediction using Kaggle history |
| `submission_scorecard.py` | Uncertainty & reliability scorecard |
| `run_v11_staged.py` | Staged pipeline runner |
| `run_v11_train_ranker_perturb.py` | Train ranker perturbation search |
| `run_v11_train_selector.py` | Train selector |
| `run_v11_meta_selector.py` | Meta selector runner |
| `CODEX_MEMORY.md` | Detailed session history, recipes, scores |
| `HANDOFF.md` | Architecture overview (partially stale — see notes below) |

## LLM usage

- Use **DeepSeek reasoner** (`deepseek-reasoner`) for all LLM judge/generation steps, via `V11_API_KEY` env var.
- HANDOFF.md says "GPT-5.4 is our moat" — **this is stale**. The V11 pipeline switched to DeepSeek.

## Data layout

- `data/` — competition data (download from Kaggle, gitignored)
- `submissions/` — all val predictions and test submissions (committed)
- `precompute/` — caches for citations, glossary, court dense hits, judge caches (large ones gitignored, regeneratable)
- `artifacts/` — run artifacts, judged bundles, meta configs (gitignored, 3GB+, regeneratable from code)
- `rust/v11_selector/` — Rust hybrid selector (build with `cargo build --release`)
