# Swiss Legal Retrieval — Kaggle Competition

Competition: [LLM Agentic Legal Information Retrieval](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval)

## Current best

- **Public LB: 0.30681** — `submissions/test_submission_overnight_combo_a.csv`
- Submitted 2026-04-10
- Produced by: `scripts/overnight_multi_path.sh` combo phase — perturbation search with 3 overnight API-path neighbors around the prior 0.30257 winner
- Val companion: `submissions/val_pred_overnight_combo_a.csv`
- Frozen baseline: `submissions/test_submission_baseline_public_best_30681.csv`
- Prior best: 0.30257 (`test_submission_baseline_public_best_30257.csv`) — still valid as a reference point

## HARD CONSTRAINT: Public leaderboard overfitting

**The public leaderboard uses only ~50% of test queries. Final standings use the OTHER 50%.** Every submission where we react to the public score increases overfitting to those ~20 public queries. This is the #1 risk in Kaggle competitions — teams at rank 5 public can drop to rank 50 private.

**Rules (non-negotiable):**

1. **Do NOT use the public LB score as a gradient.** Do not iterate by submitting → checking score → adjusting. Each such cycle overfits to the public 50%.
2. **Minimize total Kaggle submissions.** Every submission is a data leak from the public test set into our decision-making. Only submit when local evaluation gives strong, diverse signal — not to "check if it works."
3. **Never submit more than 2 candidates per experimental direction.** If the first two don't improve, the direction is wrong — don't keep submitting variations.
4. **Trust local diversity over public score.** A candidate that's strong across diverse val queries with stable bootstrap confidence is more likely to survive the private shakeup than one hill-climbed on public feedback.
5. **Final submission selection must hedge.** Kaggle allows 2 final submissions for private scoring. One should be the highest public score. The other MUST be a stable, conservative pick (high Jaccard, minimal changes, strong LB90) to hedge against public→private shakeup.
6. **Never overwrite or delete frozen baselines.** Keep `test_submission_baseline_public_best_30681.csv` and `test_submission_baseline_public_best_30257.csv` intact as hedge candidates.

**Why combo_a is exposed:** It changed 83 adds + 73 removes across all 40 test queries. If those changes happened to help mostly on the ~20 public queries, the private score could drop significantly. The prior 0.30257 winner (52 adds, 7 removes) has less shakeup variance.

## Testing workflow — how to tell better from worse

**Local val is unreliable on its own.** The val set has only 10 queries with ~5 gold cites each. A single citation flip = ~2pp F1 swing.

**Key learning (2026-04-10):** The old promotion gate was too conservative. combo_a had 82% Jaccard (gate said "likely_worse") but scored 0.30681 on Kaggle — our biggest jump ever. The Jaccard > 0.90 rule was wrong. What matters more: val F1 lift AND LB90 lift together, not Jaccard alone. Candidates with both metrics up are worth submitting even at moderate Jaccard.

### Required evaluation steps for any new candidate

1. **Generate val + test CSVs** for your candidate variant.

2. **Run the multi-signal scorecard** against the frozen baseline:
   ```bash
   python3 scripts/multi_signal_scorecard.py \
       --val-gold data/val.csv \
       --reference-test submissions/test_submission_baseline_public_best_30257.csv \
       --reference-val submissions/val_pred_baseline_public_best_30257.csv \
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
       --ref-test baseline=submissions/test_submission_baseline_public_best_30257.csv
   ```

### Promotion rules — updated 2026-04-10

- Raw val F1 AND LB90 both improved vs current best (not just one)
- Per-query F1 std is not elevated vs winner
- Promotion gate verdict is advisory only — it was wrong on combo_a (0.30681). Do NOT blindly trust it.
- Jaccard > 0.90 is NOT required (combo_a won at 0.82). But lower Jaccard = higher private-shakeup variance.
- Before submitting: check if this is worth burning a submission on (see overfitting constraint above)

### What to avoid

- Do NOT use Kaggle submissions as a search loop. Iterate locally, submit sparingly.
- Do NOT trust a single raw val F1 number. A +2pp local lift means nothing on 10 queries.
- Do NOT overwrite frozen baselines (`test_submission_baseline_public_best_30681.csv`, `test_submission_baseline_public_best_30257.csv`).

## Key files

| File | Purpose |
|------|---------|
| `pipeline_v11.py` | Core V11 retrieval pipeline (dense+BM25+judge) |
| `run_val_eval_v11.py` | Run pipeline on val split |
| `gen_test_submission_v11.py` | Generate test submission |
| `scripts/winner_localperturb_search.py` | The winning perturbation script (hash-verified recovery) |
| `scripts/multi_signal_scorecard.py` | Multi-signal evaluation scorecard |
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
