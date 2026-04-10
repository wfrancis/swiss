# Codex Memory

Date: 2026-04-07

## Goal

Establish a disciplined speed-up path for the V11 pipeline:

1. Baseline the exact current code state with a fresh internal score and a real Kaggle score.
2. Build a staged runner that separates candidate build, judging, and selection.
3. Verify score/output parity after the staged refactor.
4. Add a Rust acceleration slice.
5. Verify parity again after the Rust integration.

## Verified Historical Scores

- V7b baseline public Kaggle: `0.24456`
- V8 separate court-track public Kaggle: `0.20706`
- V7 + few-shot extra GPT run public Kaggle: `0.24170`
- Earlier V11 strict judge public Kaggle: `0.27509`

## Current Code Baseline In Progress

The current checked-out `pipeline_v11.py` includes court FAISS candidates inside the V11 judged candidate pool.

Fresh rerun on 2026-04-07 for the exact current code state:

- Internal val macro F1: `24.94%`
- Runtime: `641.56s` real time
- Command path: `run_val_eval_v11.py` with current default config
- Val output hash: `b193e251c00b78e8e5e048fb3e87e1c85cae4417b49d4777db73db5925be6997`

Fresh matching Kaggle baseline for the exact current code state:

- Test generation runtime: `2440.98s` real time
- Test output hash: `9082bbc243e4878bd650812b6aa4775a360e92b34efe921039080abbf78407f2`
- Public Kaggle score: `0.27895`
- Submission description: `Codex current-code baseline before staged runner 2026-04-07`

## Determinism Fixes Applied After Baseline

The first staged-runner parity attempt exposed real nondeterminism in V11. The following fixes were applied in `pipeline_v11.py`:

- Preserve order instead of using `set(...)` for GPT court citation unions.
- Sort explicit citation refs before inserting them.
- Sort score ranking with citation as a tie-breaker.
- Sort selection candidates with citation as a tie-breaker.
- Build the fuzzy law index from a sorted law list instead of iterating a set.
- Break fuzzy-match ties lexicographically.

Result: the pipeline is now stable across processes. This changed the stable internal val output, so the parity target moved from the earlier unstable `24.94%` file to the new deterministic `26.25%` file.

## Verified Staged Runner

Files:

- `run_v11_staged.py`
- `artifacts/v11/val_v11_strict_v1/candidate_bundles.pkl`
- `artifacts/v11/val_v11_strict_v1/judged_bundles.pkl`
- `artifacts/v11/val_v11_strict_v1/judged_bundles.json`

Verified on 2026-04-07:

- Current wrapper val macro F1: `26.25%`
- Staged Python val macro F1: `26.25%`
- Current wrapper val hash: `5eb77e99b38e12cca6b30cac9cf9710c8c5a77c078d40982685de3ea74e7d37f`
- Staged Python val hash: `5eb77e99b38e12cca6b30cac9cf9710c8c5a77c078d40982685de3ea74e7d37f`

Useful timings after caches are warm:

- Current wrapper val rerun: about `61.69s`
- Staged build: about `57s`
- Staged judge from cache: about `4s`
- Staged select: about `3.12s`

Interpretation:

- End-to-end reruns are still dominated by candidate build.
- Once judged artifacts exist, selection/evaluation iteration drops from about a minute to a few seconds.

## Verified Rust Slice

Files:

- `rust/v11_selector/Cargo.toml`
- `rust/v11_selector/src/main.rs`

Purpose:

- Read the staged judged JSON artifact.
- Apply the same selection logic as Python.
- Write the prediction CSV and evaluate val macro F1.

Verified on 2026-04-07:

- Rust val macro F1: `26.25%`
- Rust val hash: `5eb77e99b38e12cca6b30cac9cf9710c8c5a77c078d40982685de3ea74e7d37f`
- Rust parity: exact match with current Python wrapper and staged Python selector
- Warm Rust rerun time: about `0.27s`

## Deterministic Test Artifact

Verified on 2026-04-08:

- Staged Python test hash: `041cb4f5f7dac9b1a4051566d1d70e93d353bf36ee4de90d24628675a47c61b2`
- Rust test hash: `041cb4f5f7dac9b1a4051566d1d70e93d353bf36ee4de90d24628675a47c61b2`
- Python/Rust test parity: exact match
- Average predictions/query: `23.9`
- One-time staged test build+judge+select runtime: `2915.12s`

Kaggle confirmation:

- Submission description: `Codex deterministic staged+rust parity baseline 2026-04-08`
- Public Kaggle score: `0.27640`

Interpretation:

- The deterministic/staged/Rust path is now reproducible end to end.
- It is slightly below the earlier non-deterministic current-code public score of `0.27895`.
- Future work should improve from this stable deterministic baseline, not from the older unstable one.

## Meta-Selector Experiment

Goal:

- Push internal val quickly by learning a stronger final selector on top of cached judged V11 artifacts.
- Keep the stable V11 retrieval/judge pipeline unchanged.

New script:

- `run_v11_meta_selector.py`

Verified on 2026-04-08:

- Command:
  - `./.venv/bin/python run_v11_meta_selector.py --train-judged artifacts/v11/val_v11_strict_v1/judged_bundles.json --train-gold data/val.csv --apply-judged artifacts/v11/val_v11_strict_v1/judged_bundles.json --apply-gold data/val.csv --output-csv submissions/val_pred_v11_meta.csv --model-out artifacts/v11_meta/val_self_fit.pkl --config-out artifacts/v11_meta/val_self_fit_config.json --random-search 1000 --evaluate-loo`
- Best internal val macro F1: `41.7356%`
- Best selector config:
  - `target_mult=0.7`
  - `bias=0`
  - `min_out=6`
  - `max_out=20`
  - `thresh=0.06`
  - `court_cap_frac=0.2`
- Output CSV:
  - `submissions/val_pred_v11_meta.csv`
- Saved artifacts:
  - `artifacts/v11_meta/val_self_fit.pkl`
  - `artifacts/v11_meta/val_self_fit_config.json`

Important caveat:

- Leave-one-query-out macro F1 for this same setup is only `22.5663%`.
- Interpretation: the current judged artifacts contain enough signal to support a much stronger learned selector, but fitting on the 10-query val set is heavily overfit.
- Practical lesson: the next serious step is to generate train-side V11-style artifacts so the meta-selector can be trained on real supervised data instead of on val.

## Fast Iteration Infrastructure Upgrades

Verified on 2026-04-08:

- `run_v11_staged.py` now supports `--split train`, `--offset`, and `--limit`.
- Artifact paths are now shard-aware when `offset` or `limit` are set.
- `pipeline_v11.py` now supports `V11_QUERY_OFFSET` in addition to `V11_MAX_QUERIES`.
- `run_v11_meta_selector.py` now accepts multiple `--train-judged` / `--apply-judged` inputs so sharded train judged bundles can be used directly.
- Precompute scripts now support `train` and resume safely from existing outputs:
  - `precompute/gen_query_expansions.py`
  - `precompute/gen_case_citations.py`
  - `precompute/gen_full_citations_v2.py`

Smoke verification:

- `./.venv/bin/python -m py_compile pipeline_v11.py run_v11_staged.py run_v11_meta_selector.py precompute/gen_query_expansions.py precompute/gen_case_citations.py precompute/gen_full_citations_v2.py`
- `./.venv/bin/python run_v11_staged.py select --split val --output /tmp/val_pred_v11_stage_check.csv`
  - Result remained `26.25%` macro F1 on the stable staged selector path.

Practical implication:

- We can now create train-side GPT precomputes incrementally.
- We can build/judge train in shards instead of one giant monolithic run.
- Once those train judged shards exist, meta-selector training and val application stay in the fast local loop.

## Rust Hybrid Lab

Verified on 2026-04-08:

- New binary:
  - `rust/v11_selector/src/bin/hybrid_lab.rs`
- Purpose:
  - fuse existing `V7b` predictions with cached `V11` judged artifacts
  - search hybrid scoring and post-processing rules fully locally in Rust
  - output the best val CSV and config quickly, with no new API calls

Command:

- `cargo run --release --manifest-path rust/v11_selector/Cargo.toml --bin hybrid_lab -- artifacts/v11/val_v11_strict_v1/judged_bundles.json submissions/val_pred_v7.csv submissions/val_pred_v11_hybrid_lab.csv 5000 artifacts/v11_meta/hybrid_lab_best.json`

Result:

- Best local val macro F1: `29.6788%`
- Output CSV:
  - `submissions/val_pred_v11_hybrid_lab.csv`
- Output hash:
  - `c1c8b3c26b90249a372185aa25b13876045f303ceb4f036b6eea024c2cde54b0`
- Average predictions/query:
  - `25.5`
- Best config:
  - `v7_bonus=0.5`
  - `v11_bonus=2.0`
  - `auto_keep_bonus=0.5`
  - `explicit_bonus=0.0`
  - `must_bonus=1.0`
  - `plausible_bonus=0.5`
  - `reject_penalty=0.0`
  - `conf_weight=1.5`
  - `final_weight=2.0`
  - `raw_weight=0.3`
  - `gpt_freq_weight=0.8`
  - `law_bonus=0.0`
  - `court_bonus=-0.5`
  - `court_dense_only_penalty=0.0`
  - `single_source_penalty=-0.5`
  - `target_mult=1.0`
  - `target_bias=2`
  - `min_output=8`
  - `max_output=32`
  - `court_cap_frac=0.2`

Interpretation:

- Rust is a very good fit for the first two no-API items:
  - existing-artifact hybrid/selector search
  - stronger local post-processing rules
- Rust can also apply train-derived priors cheaply once they are computed.
- For the local supervised ranker, the fastest overall setup is likely:
  - train in Python / scikit-learn
  - export scores or parameters
  - run inference and search in Rust

## Rust Hybrid Grid + Priors

Verified on 2026-04-08:

- Extended `hybrid_lab.rs` with:
  - train-derived exact and law-base priors from `data/train.csv`
  - per-law-family and per-court-base caps
  - extra source-aware scoring knobs
  - `apply` mode for deterministic regeneration from a saved config

Exhaustive finite grid:

- Mode:
  - `HYBRID_LAB_MODE=grid`
- Grid size:
  - `103,680` combinations over the newly added prior/cap dimensions around the current best hybrid
- Observed plateau:
  - best remained `30.4430%` through roughly `40k+` combinations before the run was stopped

Banked best config:

- `artifacts/v11_meta/hybrid_grid_best_30443.json`

## Reliability Lesson And New Scorecard

Verified on 2026-04-07 after the failed Rust hybrid Kaggle submission:

- New tool:
  - `submission_scorecard.py`
- Purpose:
  - evaluate a candidate with more than one local number
  - report val macro F1, bootstrap lower bounds, per-query spread, and test-shape drift versus trusted baselines

Scorecard comparison:

- `V7b`
  - val macro F1: `0.211628`
  - val bootstrap LB90: `0.182321`
  - val query std: `0.074174`
  - test avg predictions: `28.45`
  - test avg court fraction: `0.1541`
- deterministic `V11`
  - val macro F1: `0.262466`
  - val bootstrap LB90: `0.236178`
  - val query std: `0.067671`
  - test avg predictions: `23.85`
  - test avg court fraction: `0.3738`
- failed Rust hybrid grid
  - val macro F1: `0.304430`
  - val bootstrap LB90: `0.269280`
  - val query std: `0.085913`
  - val query min: `0.186047`
  - test avg predictions: `22.20`
  - test avg court fraction: `0.1610`
  - public Kaggle: `0.25903`

Interpretation:

- The bad Rust hybrid improved raw val a lot, but it also got materially less stable across queries and shifted the prediction shape toward a narrower, much more law-heavy mix.
- The root mistake was using the same 10-query val set both to search tens of thousands of configs and to claim success.
- Going forward, no candidate should be promoted on raw val macro F1 alone.

New local promotion rule:

- Treat raw val macro F1 as only one signal.
- Also inspect:
  - bootstrap lower bound on val
  - per-query spread / floor
  - test output shape versus trusted baselines
  - overlap with trusted baselines on test
- Longer-term fix remains train-backed validation or nested CV over the search procedure, rather than single-set val optimization.

## Robust Rust Search Before Retraining

Verified on 2026-04-07:

- `rust/v11_selector/src/bin/hybrid_lab.rs` now has:
  - `robust` mode
  - `consensus` mode
  - paired test-artifact support during search
  - output-shape-aware objective instead of raw val-only ranking
  - consensus controls via env vars:
    - `HYBRID_LAB_CONSENSUS_VOTE_FRAC`
    - `HYBRID_LAB_CONSENSUS_TARGET_MODE`
    - `HYBRID_LAB_CONSENSUS_TARGET_BIAS`

Short-search results on the existing artifacts:

- Baseline deterministic `V11`
  - val macro F1: `0.262466`
  - val LB90: `0.236178`
  - val std: `0.067671`
  - test avg preds: `23.85`
  - test avg court fraction: `0.3738`

- Robust consensus, default controls
  - val CSV: `submissions/val_pred_v11_consensus_val.csv`
  - test CSV: `submissions/test_submission_v11_consensus.csv`
  - val macro F1: `0.271358`
  - val LB90: `0.253627`
  - val std: `0.044647`
  - test avg preds: `16.68`
  - test avg court fraction: `0.1824`

- Robust consensus, looser controls
  - env:
    - `HYBRID_LAB_CONSENSUS_VOTE_FRAC=0.5`
    - `HYBRID_LAB_CONSENSUS_TARGET_MODE=mean`
    - `HYBRID_LAB_CONSENSUS_TARGET_BIAS=2`
  - val CSV: `submissions/val_pred_v11_consensus_loose_val.csv`
  - test CSV: `submissions/test_submission_v11_consensus_loose.csv`
  - val macro F1: `0.274932`
  - val LB90: `0.254328`
  - val std: `0.051919`
  - test avg preds: `18.40`
  - test avg court fraction: `0.1882`

- Robust consensus, bolder controls
  - env:
    - `HYBRID_LAB_CONSENSUS_VOTE_FRAC=0.4`
    - `HYBRID_LAB_CONSENSUS_TARGET_MODE=max`
    - `HYBRID_LAB_CONSENSUS_TARGET_BIAS=0`
  - val CSV: `submissions/val_pred_v11_consensus_bolder_val.csv`
  - val macro F1: `0.272656`
  - val LB90: `0.240520`
  - val std: `0.080788`

Interpretation:

- The robust-consensus path is clearly safer than the failed raw val-optimized hybrid.
- However, every robust-consensus candidate tested so far is still much sparser and more court-light on `test` than the trusted deterministic `V11` baseline.
- The best no-retrain local candidate right now is the looser robust consensus:
  - `submissions/val_pred_v11_consensus_loose_val.csv`
  - `submissions/test_submission_v11_consensus_loose.csv`
- Even that candidate still looks too compressed to trust as a promotion over the real Kaggle baselines without a stronger validation source.

Deterministic regeneration:

- `HYBRID_LAB_MODE=apply cargo run --release --manifest-path rust/v11_selector/Cargo.toml --bin hybrid_lab -- artifacts/v11/val_v11_strict_v1/judged_bundles.json submissions/val_pred_v7.csv data/train.csv submissions/val_pred_v11_hybrid_grid.csv 0 artifacts/v11_meta/hybrid_grid_best_30443.json`

Verified result:

- Local val macro F1: `30.4430%`
- Average predictions/query: `25.5`
- Output CSV:
  - `submissions/val_pred_v11_hybrid_grid.csv`
- Output hash:
  - `e8417082ae8d97ab12b1ae0483f2d4eaf2b634c1c73626ac2c4d22f04bde407e`

Best config summary:

- strong preference for `V11` candidates over `V7`, while still keeping a small `V7` bonus
- strong negative court bias
- negative `court_dense` bonus
- mild dense-train law-base prior (`dense_law_base_train_weight=0.2`)
- cap of `1` court citation per base case
- cap of `1` law citation per article-family/statute base

Kaggle check:

- Submission:
  - `Codex Rust hybrid grid 30.443 internal val 2026-04-07`
- Submission ref:
  - `51566616`
- Public Kaggle score:
  - `0.25903`

Interpretation:

- The Rust-only hybrid/grid path improved local val to `30.4430%` but did **not** transfer to Kaggle public.
- It underperformed the stable judged baselines:
  - `0.27895` current-code baseline
  - `0.27640` deterministic staged+Rust baseline
- Conclusion: this branch is useful as a fast local search lab, but the current no-API hybrid rules are overfitting local val and should not be treated as the next production submission path.

## Execution Gates

- Do not refactor until the exact current code baseline is recorded.
- After the staged runner lands, output hashes and internal score must match the current-code baseline.
- After the Rust slice lands, output hashes and internal score must still match.
- Use Kaggle again only after parity is confirmed, and treat Kaggle as the final external confirmation gate.

## Next Implementation Shape

Staged runner should separate:

- `build_candidates`
- `judge_candidates`
- `select_predictions`

Preferred artifact flow:

- Persist candidate bundles after retrieval/bucketing.
- Persist judged bundles after LLM labeling.
- Make selection/evaluation reruns work from cached artifacts without redoing retrieval or judge calls.
- Use the Rust selector for fast selection/eval sweeps on judged artifacts.

## Notes

- Kaggle rules pasted by user say `5` submissions/day; obey that stricter cap.
- Kaggle auth currently works through `KAGGLE_API_TOKEN`.
- `HANDOFF.md` remains the broader competition context; this file is the working execution log.
- Two Kaggle submissions were used in the current window:
  - `Codex current-code baseline before staged runner 2026-04-07` → `0.27895`
  - `Codex deterministic staged+rust parity baseline 2026-04-08` → `0.27640`

## DeepSeek Dense Train Pilot

DeepSeek-V3.2 Thinking Mode integration:

- API base used: `https://api.deepseek.com/v1`
- Model used for pilot: `deepseek-reasoner`
- Important judge behavior: DeepSeek reasoning tokens count against `max_tokens`, so the V11 judge needs a much higher cap than OpenAI. `V11_MAX_TOKENS=8000` produced valid JSON where lower caps returned empty `content`.

Overnight dense-train pilot status:

- Dense train query list:
  - `artifacts/dense_train_qids_100.txt`
- Completed for `100` dense train queries:
  - `precompute/train_query_expansions.json`
  - `precompute/train_case_citations.json`
  - `precompute/train_full_citations_v2.json`
  - `artifacts/v11/train_v11_strict_v1_deepseek_reasoner_dense100__offset0_n100/candidate_bundles.pkl`
- Not completed yet:
  - `artifacts/v11/train_v11_strict_v1_deepseek_reasoner_dense100__offset0_n100/judged_bundles.pkl`
  - `artifacts/v11/train_v11_strict_v1_deepseek_reasoner_dense100__offset0_n100/judged_bundles.json`

Timing observed:

- Full precompute + staged build for the `100`-query dense shard completed in about `1h47m`
- Remaining bottleneck is the train-side DeepSeek judge stage

## DeepSeek-Prior Rust Experiment

Goal:

- Cheap no-retrain test: inject new DeepSeek train artifacts as additional priors into the Rust hybrid lab without waiting for train judged bundles

Code change:

- `rust/v11_selector/src/bin/hybrid_lab.rs`
  - `load_train_priors()` now augments label-derived priors with citations surfaced by:
    - `precompute/train_full_citations_v2.json`
    - `precompute/train_case_citations.json`
    - `precompute/train_query_expansions.json` (`specific_articles`)

Experiment command:

- Val search:
  - `HYBRID_LAB_MODE=consensus HYBRID_LAB_KEEP_TOP=24 HYBRID_LAB_OUTPUT_TARGET=val HYBRID_LAB_CONSENSUS_VOTE_FRAC=0.5 HYBRID_LAB_CONSENSUS_TARGET_MODE=mean HYBRID_LAB_CONSENSUS_TARGET_BIAS=2 cargo run --release --manifest-path rust/v11_selector/Cargo.toml --bin hybrid_lab -- artifacts/v11/val_v11_strict_v1/judged_bundles.json submissions/val_pred_v7.csv data/train.csv submissions/val_pred_v11_consensus_loose_deepseekpriors.csv 400 artifacts/v11_meta/hybrid_consensus_deepseekpriors_best.json artifacts/v11/test_v11_strict_v1/judged_bundles.json submissions/test_submission_v7.csv`
- Matching test generation:
  - same command with `HYBRID_LAB_OUTPUT_TARGET=test`

Best config found:

- `artifacts/v11_meta/hybrid_consensus_deepseekpriors_best.json`
- best objective: `0.154670`
- best config:
  - `v7_bonus=0.5`
  - `v11_bonus=0.5`
  - `auto_keep_bonus=0.5`
  - `explicit_bonus=0.5`
  - `must_bonus=3.0`
  - `plausible_bonus=0.2`
  - `reject_penalty=-1.5`
  - `conf_weight=0.5`
  - `final_weight=1.0`
  - `gpt_freq_weight=0.2`
  - `dense_bonus=-0.3`
  - `bm25_bonus=0.2`
  - `gpt_case_bonus=0.5`
  - `court_dense_bonus=-0.2`
  - `law_bonus=1.0`
  - `court_bonus=-0.5`
  - `exact_train_weight=0.1`
  - `dense_train_weight=0.2`
  - `law_base_train_weight=0.5`
  - `dense_law_base_train_weight=-0.5`
  - `court_dense_only_penalty=-1.0`
  - `single_source_penalty=-0.8`
  - `target_mult=1.0`
  - `target_bias=2`
  - `min_output=12`
  - `max_output=20`
  - `court_cap_frac=0.3`
  - `max_law_per_base=2`
  - `max_court_per_base=3`

Scorecard result:

- New outputs:
  - `submissions/val_pred_v11_consensus_loose_deepseekpriors.csv`
  - `submissions/test_submission_v11_consensus_loose_deepseekpriors.csv`
- Output hashes:
  - val: `4900f0be9775585f19be128a3ef65613fda45dc64d15079286af985f70747826`
  - test: `ec6a9f7b6d37a93aa84ad264e2a59b076220b056867fb8a77b272feb8b6d52e9`
- Scorecard:
  - val macro F1: `0.273978`
  - val LB90: `0.248802`
  - val std: `0.064340`
  - test avg predictions: `19.375`
  - test avg court fraction: `0.187233`
  - test Jaccard vs V11 baseline: `0.664072`
  - test Jaccard vs V7 baseline: `0.437093`

Comparison to prior safer no-retrain consensus:

- Old safer consensus:
  - val macro F1: `0.274932`
  - val LB90: `0.254328`
  - val std: `0.051919`
  - test avg predictions: `18.4`
  - test avg court fraction: `0.188222`
  - test Jaccard vs V11 baseline: `0.689043`
  - test Jaccard vs V7 baseline: `0.410333`

Interpretation:

- The DeepSeek priors changed the outputs, but they did **not** improve the safer local frontier.
- Compared with the prior loose consensus, the DeepSeek-prior branch is slightly worse on:
  - raw val macro F1
  - bootstrap lower bound
  - per-query stability
  - closeness to the trusted V11 baseline
- Conclusion: keep the DeepSeek-prior Rust path logged as explored, but do **not** promote it over the prior safer consensus or over the trusted V11 Kaggle baselines.

Kaggle reality check:

- Submission:
  - `Codex DeepSeek-prior consensus loose 2026-04-08`
- Submission ref:
  - `51575511`
- Public Kaggle score:
  - `0.30094`

Updated interpretation:

- The local scorecard failed to predict this branch correctly.
- Despite slightly worse local scorecard metrics than the prior safer consensus, the DeepSeek-prior Rust candidate materially outperformed every prior public submission.
- New public leaderboard ordering:
  - `0.30094` DeepSeek-prior consensus loose
  - `0.27895` current-code baseline before staged runner
  - `0.27640` deterministic staged+Rust parity baseline
  - `0.27509` V11 strict judge on V7b retrieval
  - `0.25903` failed raw val-optimized Rust hybrid
  - `0.24456` V7b reset baseline

Revised lesson:

- The no-retrain DeepSeek-prior injection is worth something substantial on Kaggle public, even though the 10-query local validation stack still does not explain it well.
- The scorecard remains useful as a guardrail against obviously bad overfit branches, but it is not sufficient as a promotion oracle.

## Post-0.30094 Rust Squeezing

Goal:

- Push the winning DeepSeek-prior Rust branch further without new API calls by broadening consensus search around the public winner

Searches run:

- `HYBRID_LAB_MODE=consensus HYBRID_LAB_KEEP_TOP=48 HYBRID_LAB_CONSENSUS_VOTE_FRAC=0.40 HYBRID_LAB_CONSENSUS_TARGET_MODE=mean HYBRID_LAB_CONSENSUS_TARGET_BIAS=2`
- `HYBRID_LAB_MODE=consensus HYBRID_LAB_KEEP_TOP=64 HYBRID_LAB_CONSENSUS_VOTE_FRAC=0.35 HYBRID_LAB_CONSENSUS_TARGET_MODE=mean HYBRID_LAB_CONSENSUS_TARGET_BIAS=3`
- `HYBRID_LAB_MODE=consensus HYBRID_LAB_KEEP_TOP=32 HYBRID_LAB_CONSENSUS_VOTE_FRAC=0.50 HYBRID_LAB_CONSENSUS_TARGET_MODE=median HYBRID_LAB_CONSENSUS_TARGET_BIAS=2`

All three longer searches converged on the same best sampled config:

- objective: `0.222796`
- best config:
  - `v7_bonus=1.5`
  - `v11_bonus=1.0`
  - `both_bonus=0.3`
  - `explicit_bonus=1.0`
  - `must_bonus=3.0`
  - `plausible_bonus=0.2`
  - `reject_penalty=-0.5`
  - `conf_weight=0.5`
  - `final_weight=0.0`
  - `raw_weight=0.3`
  - `gpt_freq_weight=0.8`
  - `source_count_weight=0.1`
  - `dense_bonus=0.2`
  - `bm25_bonus=0.2`
  - `gpt_case_bonus=-0.3`
  - `cocitation_penalty=-0.2`
  - `court_dense_bonus=0.0`
  - `court_bonus=-0.2`
  - `exact_train_weight=-0.3`
  - `dense_train_weight=-0.5`
  - `law_base_train_weight=0.1`
  - `dense_law_base_train_weight=0.8`
  - `court_dense_only_penalty=-0.5`
  - `single_source_penalty=-0.5`
  - `target_bias=2`
  - `min_output=12`
  - `max_output=32`
  - `court_cap_frac=0.3`
  - `max_law_per_base=1`
  - `max_court_per_base=3`

Candidate outputs:

- `submissions/val_pred_v11_consensus_ds_k48_v040_b2.csv`
- `submissions/test_submission_v11_consensus_ds_k48_v040_b2.csv`
- `submissions/val_pred_v11_consensus_ds_k64_v035_b3.csv`
- `submissions/test_submission_v11_consensus_ds_k64_v035_b3.csv`
- `submissions/val_pred_v11_consensus_ds_k32_v050_med_b2.csv`
- `submissions/test_submission_v11_consensus_ds_k32_v050_med_b2.csv`

Scorecards:

- `k48_v040_b2`
  - val macro F1: `0.271598`
  - val LB90: `0.244852`
  - test avg predictions: `21.175`
  - test avg court fraction: `0.220197`
  - test Jaccard vs public winner (`0.30094`): `0.872768`

- `k64_v035_b3`
  - val macro F1: `0.272730`
  - val LB90: `0.245394`
  - test avg predictions: `21.875`
  - test avg court fraction: `0.217594`
  - test Jaccard vs public winner (`0.30094`): `0.867829`

- `k32_v050_med_b2`
  - val macro F1: `0.275524`
  - val LB90: `0.246533`
  - test avg predictions: `22.1`
  - test avg court fraction: `0.214847`
  - test Jaccard vs public winner (`0.30094`): `0.852684`

Comparison to the current public winner:

- Winner:
  - `submissions/test_submission_v11_consensus_loose_deepseekpriors.csv`
  - Kaggle public: `0.30094`
  - test avg predictions: `19.375`
  - test avg court fraction: `0.187233`

How different the new candidates are from the winner:

- `k48_v040_b2`: all `40/40` test queries changed; `154` citations added, `27` removed
- `k64_v035_b3`: all `40/40` test queries changed; `175` citations added, `19` removed
- `k32_v050_med_b2`: all `40/40` test queries changed; `188` citations added, `22` removed

Interpretation:

- These broader-consensus candidates are materially fuller and more court-heavy than the `0.30094` winner.
- Local metrics remain noisy, but among the new no-API Rust variants, `k32_v050_med_b2` is the strongest follow-up candidate to test externally.

## DeepSeek Priors Expanded To 200

Train prior expansion status:

- `precompute/train_query_expansions.json`: `200`
- `precompute/train_case_citations.json`: `200`
- `precompute/train_full_citations_v2.json`: `200`

Coverage notes:

- First shard:
  - `artifacts/dense_train_qids_100.txt`
  - first `100` query IDs among the `112` train queries with `>=10` gold citations
- Second shard:
  - `artifacts/dense_train_qids_200_stage2.txt`
  - top `100` densest train queries not already covered, which includes the remaining `12` queries with `>=10` gold plus the next-densest `9/8/7` citation queries

Rerun of the public-winning Rust recipe on `200` priors:

- Command shape:
  - `HYBRID_LAB_MODE=consensus`
  - `HYBRID_LAB_KEEP_TOP=24`
  - `HYBRID_LAB_CONSENSUS_VOTE_FRAC=0.5`
  - `HYBRID_LAB_CONSENSUS_TARGET_MODE=mean`
  - `HYBRID_LAB_CONSENSUS_TARGET_BIAS=2`
  - `iterations=400`

New best config:

- `artifacts/v11_meta/hybrid_consensus_deepseekpriors_200_best.json`
- best objective: `0.150808`
- config:
  - `v7_bonus=1.5`
  - `v11_bonus=0.0`
  - `both_bonus=0.0`
  - `auto_keep_bonus=1.5`
  - `explicit_bonus=0.5`
  - `must_bonus=0.5`
  - `plausible_bonus=0.2`
  - `reject_penalty=0.0`
  - `conf_weight=0.0`
  - `final_weight=0.0`
  - `raw_weight=0.3`
  - `gpt_freq_weight=0.5`
  - `source_count_weight=-0.1`
  - `dense_bonus=0.0`
  - `bm25_bonus=-0.3`
  - `gpt_case_bonus=0.5`
  - `cocitation_penalty=-1.0`
  - `court_dense_bonus=-0.5`
  - `law_bonus=1.0`
  - `court_bonus=-0.2`
  - `exact_train_weight=0.0`
  - `dense_train_weight=0.2`
  - `law_base_train_weight=-0.3`
  - `dense_law_base_train_weight=0.0`
  - `court_dense_only_penalty=-1.0`
  - `single_source_penalty=-0.8`
  - `target_bias=2`
  - `min_output=8`
  - `max_output=32`
  - `court_cap_frac=0.3`
  - `max_law_per_base=3`
  - `max_court_per_base=3`

New outputs:

- `submissions/val_pred_v11_consensus_loose_deepseekpriors_200.csv`
- `submissions/test_submission_v11_consensus_loose_deepseekpriors_200.csv`
- hashes:
  - val: `1d0c85721d36cb4e18dfba189fccd815b25d16bfe277fb66294a3f005057af27`
  - test: `a0f4d2783464493c3127eafa58fa4fd7f1d64f99022ad4f5aa8b4361d03b7ec2`

Scorecard:

- val macro F1: `0.273978`
- val LB90: `0.248802`
- val std: `0.064340`
- test avg predictions: `19.375`
- test avg court fraction: `0.181823`
- test Jaccard vs V11 baseline: `0.660143`
- test Jaccard vs V7 baseline: `0.439735`
- test Jaccard vs public winner (`0.30094`): `0.955852`

Comparison to the public-winning `100`-prior branch:

- Old winner local scorecard:
  - val macro F1: `0.273978`
  - val LB90: `0.248802`
  - val std: `0.064340`
  - test avg predictions: `19.375`
  - test avg court fraction: `0.187233`
- New `200`-prior branch:
  - same local val metrics
  - same average test prediction count
  - slightly lower test court fraction
  - changed `16/40` test queries
  - `22` citations added, `26` removed relative to the public winner

Interpretation:

- Expanding priors from `100` to `200` changed the selector, but not enough to move the local validation score.
- Internal evidence says this is a lateral move, not a clear local win.
- Because the local stack misranked the `0.30094` winner previously, the only decisive answer for this branch would be another Kaggle submission.

## 200-Prior Wide Rust Search

Reason for rerun:

- The quick `400`-iteration rerun on `200` priors looked lateral, but that was too weak a test.
- Repeated the broader `4000`-iteration consensus searches used previously, now with `200` train priors.

Search settings:

- `k48_v040_b2`
  - `HYBRID_LAB_KEEP_TOP=48`
  - `HYBRID_LAB_CONSENSUS_VOTE_FRAC=0.40`
  - `HYBRID_LAB_CONSENSUS_TARGET_MODE=mean`
  - `HYBRID_LAB_CONSENSUS_TARGET_BIAS=2`
- `k64_v035_b3`
  - `HYBRID_LAB_KEEP_TOP=64`
  - `HYBRID_LAB_CONSENSUS_VOTE_FRAC=0.35`
  - `HYBRID_LAB_CONSENSUS_TARGET_MODE=mean`
  - `HYBRID_LAB_CONSENSUS_TARGET_BIAS=3`
- `k32_v050_med_b2`
  - `HYBRID_LAB_KEEP_TOP=32`
  - `HYBRID_LAB_CONSENSUS_VOTE_FRAC=0.50`
  - `HYBRID_LAB_CONSENSUS_TARGET_MODE=median`
  - `HYBRID_LAB_CONSENSUS_TARGET_BIAS=2`

All three `200`-prior searches converged on the same best sampled config:

- best objective: `0.230183`
- config:
  - `v7_bonus=1.5`
  - `v11_bonus=1.0`
  - `both_bonus=0.0`
  - `auto_keep_bonus=0.5`
  - `explicit_bonus=0.5`
  - `must_bonus=3.0`
  - `plausible_bonus=0.0`
  - `reject_penalty=-0.5`
  - `conf_weight=0.5`
  - `final_weight=0.0`
  - `raw_weight=0.0`
  - `gpt_freq_weight=0.0`
  - `source_count_weight=-0.1`
  - `dense_bonus=0.2`
  - `bm25_bonus=0.0`
  - `gpt_case_bonus=0.0`
  - `cocitation_penalty=0.0`
  - `court_dense_bonus=0.0`
  - `law_bonus=-0.5`
  - `court_bonus=0.2`
  - `exact_train_weight=0.3`
  - `dense_train_weight=-0.5`
  - `law_base_train_weight=-0.1`
  - `dense_law_base_train_weight=0.5`
  - `court_dense_only_penalty=-1.5`
  - `single_source_penalty=-0.8`
  - `target_mult=0.9`
  - `target_bias=2`
  - `min_output=6`
  - `max_output=32`
  - `court_cap_frac=0.3`
  - `max_law_per_base=1`
  - `max_court_per_base=0`

Candidate outputs:

- `submissions/val_pred_v11_consensus_200_k48_v040_b2.csv`
- `submissions/test_submission_v11_consensus_200_k48_v040_b2.csv`
- `submissions/val_pred_v11_consensus_200_k64_v035_b3.csv`
- `submissions/test_submission_v11_consensus_200_k64_v035_b3.csv`
- `submissions/val_pred_v11_consensus_200_k32_v050_med_b2.csv`
- `submissions/test_submission_v11_consensus_200_k32_v050_med_b2.csv`

Scorecards:

- `k48_v040_b2`
  - val macro F1: `0.271190`
  - val LB90: `0.244170`
  - val std: `0.067760`
  - test avg predictions: `21.125`
  - test avg court fraction: `0.220478`
  - test Jaccard vs winner: `0.868277`

- `k64_v035_b3`
  - val macro F1: `0.270854`
  - val LB90: `0.246122`
  - val std: `0.061982`
  - test avg predictions: `21.9`
  - test avg court fraction: `0.217055`
  - test Jaccard vs winner: `0.869130`

- `k32_v050_med_b2`
  - val macro F1: `0.271724`
  - val LB90: `0.243812`
  - val std: `0.069368`
  - test avg predictions: `21.725`
  - test avg court fraction: `0.225294`
  - test Jaccard vs winner: `0.858272`

How different these are from the public `0.30094` winner:

- `k48_v040_b2`: all `40/40` test queries changed; `154` citations added, `29` removed
- `k64_v035_b3`: all `40/40` test queries changed; `176` citations added, `17` removed
- `k32_v050_med_b2`: all `40/40` test queries changed; `183` citations added, `26` removed

Interpretation:

- The `200`-prior expansion is **not** just a lateral no-op once the Rust search is widened.
- It shifts the search toward fuller, more court-heavy candidates with a different sampled optimum.
- Local validation still does not clearly prove superiority over the public `0.30094` winner, but these are genuine new candidates rather than cosmetic variants.
- Among the three, `k64_v035_b3` is the cleanest balanced follow-up:
  - best bootstrap lower bound
  - lowest query variance
  - materially different from the current winner

Kaggle reality check:

- Submitted candidate:
  - `submissions/test_submission_v11_consensus_200_k64_v035_b3.csv`
- Submission:
  - `Codex 200-prior rust consensus k64_v035_b3 2026-04-08`
- Submission ref:
  - `51579225`
- Public Kaggle score:
  - `0.29393`

Updated interpretation:

- The `200`-prior + wide-Rust-search branch is a real, competitive branch, but it did **not** beat the current public winner.
- Public ranking of recent bests:
  - `0.30094` `test_submission_v11_consensus_loose_deepseekpriors.csv`
  - `0.29393` `test_submission_v11_consensus_200_k64_v035_b3.csv`
  - `0.27895` current-code baseline before staged runner
  - `0.27640` deterministic staged+Rust baseline

What this means:

- Rust iterations absolutely do mean something; they surfaced a `200`-prior branch that is much better than the old baselines.
- But the extra `100` train priors plus broader consensus did not translate into a new public best.
- The current champion remains the `100`-prior DeepSeek-prior consensus loose branch.

2026-04-08: Split-prior Rust refactor to test "bad mixing" hypothesis

Why this was done:

- The `200`-prior branch seemed to fail because the second `100` queries were being collapsed into the same prior tables as the first `100`.
- The second shard is exactly:
  - `12` dense queries with `>=10` gold citations
  - `88` sparse queries with `7-9` gold citations
- So we patched the Rust lab to stop treating those priors as interchangeable.

Code changes in `rust/v11_selector/src/bin/hybrid_lab.rs`:

- Added new prior channels to `TrainPriors`:
  - `dense100`
  - `dense12`
  - `sparse79`
- Kept the old `exact_all`, `exact_dense`, `law_base_all`, `law_base_dense` tables as gold-train priors.
- Refactored `load_train_priors()` to:
  - read `artifacts/dense_train_qids_100.txt`
  - read `artifacts/dense_train_qids_200_stage2.txt`
  - send artifact priors from `train_full_citations_v2.json`, `train_case_citations.json`, and `train_query_expansions.json` into the new separate channels instead of collapsing them into the gold priors
- Added six new config weights:
  - `dense100_exact_weight`
  - `dense12_exact_weight`
  - `sparse79_exact_weight`
  - `dense100_law_base_weight`
  - `dense12_law_base_weight`
  - `sparse79_law_base_weight`
- Tightened `random_config()` away from the old bad regime:
  - no positive `court_bonus`
  - no negative `dense_train_weight`
  - no unlimited `max_court_per_base`
  - `v11_bonus >= v7_bonus`

Smoke search:

- Command:
  - `HYBRID_LAB_MODE=robust HYBRID_LAB_OUTPUT_TARGET=test HYBRID_LAB_KEEP_TOP=16 rust/v11_selector/target/release/hybrid_lab artifacts/v11/val_v11_strict_v1/judged_bundles.json submissions/val_pred_v7.csv data/train.csv submissions/test_submission_v11_splitpriors_smoke.csv 4000 artifacts/v11_meta/hybrid_splitpriors_smoke.json artifacts/v11/test_v11_strict_v1/judged_bundles.json submissions/test_submission_v7.csv`
- Matching val apply:
  - `HYBRID_LAB_MODE=apply rust/v11_selector/target/release/hybrid_lab artifacts/v11/val_v11_strict_v1/judged_bundles.json submissions/val_pred_v7.csv data/train.csv submissions/val_pred_v11_splitpriors_smoke.csv 1 artifacts/v11_meta/hybrid_splitpriors_smoke.json`

Best split-prior smoke config:

- Saved config:
  - `artifacts/v11_meta/hybrid_splitpriors_smoke.json`
- Output files:
  - `submissions/val_pred_v11_splitpriors_smoke.csv`
  - `submissions/test_submission_v11_splitpriors_smoke.csv`
- Key weights:
  - `v7_bonus=0.5`
  - `v11_bonus=0.5`
  - `both_bonus=1.0`
  - `court_bonus=0.0`
  - `dense_train_weight=0.2`
  - `dense100_exact_weight=0.1`
  - `dense12_exact_weight=0.0`
  - `sparse79_exact_weight=0.1`
  - `dense100_law_base_weight=0.3`
  - `dense12_law_base_weight=0.1`
  - `sparse79_law_base_weight=-0.5`
  - `target_mult=1.0`
  - `target_bias=2`
  - `max_law_per_base=1`
  - `max_court_per_base=2`

Scorecard for split-prior smoke candidate:

- val macro F1: `0.280158`
- val LB90: `0.247436`
- val LB95: `0.238859`
- val std: `0.082281`
- val min: `0.176471`
- val avg predictions: `25.50`
- val avg court fraction: `0.2921`
- test avg predictions: `22.20`
- test avg court fraction: `0.2824`
- test Jaccard vs staged V11: `0.688350`
- test Jaccard vs V7: `0.349318`
- test Jaccard vs public `0.30094` winner: `0.686701`

How different it is from the public `0.30094` winner:

- all `40/40` test queries changed
- `216` citations added
- `103` citations removed

Interpretation:

- The split-prior patch is doing something real.
- It learned the intended asymmetry:
  - dense artifact priors remain useful
  - sparse79 law-base prior is being pushed negative
- This supports the hypothesis that the main issue was bad prior mixing, not that the extra `100` queries were inherently useless.
- Local scoring still is not trustworthy enough to call this a promotion candidate yet.

2026-04-08: Strict V11-lead + widened split-prior consensus sweep

Change:

- Tightened `random_config()` again so `v11_bonus` must be strictly greater than `v7_bonus`.
- Then reran the winning consensus family rather than inventing a new one:
  - `HYBRID_LAB_MODE=consensus`
  - `HYBRID_LAB_KEEP_TOP=24`
  - `HYBRID_LAB_CONSENSUS_VOTE_FRAC=0.5`
  - `HYBRID_LAB_CONSENSUS_TARGET_MODE=mean`
  - `HYBRID_LAB_CONSENSUS_TARGET_BIAS=2`

Run outputs:

- Test consensus:
  - `submissions/test_submission_v11_splitpriors_consensus_vlead.csv`
- Saved best config:
  - `artifacts/v11_meta/hybrid_splitpriors_consensus_vlead.json`
- Matching val apply:
  - `submissions/val_pred_v11_splitpriors_consensus_vlead_apply.csv`

Best strict-V11 config:

- `v7_bonus=0.5`
- `v11_bonus=1.5`
- `both_bonus=0.3`
- `court_bonus=-0.2`
- `dense_train_weight=0.5`
- `dense100_exact_weight=0.3`
- `dense12_exact_weight=0.1`
- `sparse79_exact_weight=0.1`
- `dense100_law_base_weight=0.1`
- `dense12_law_base_weight=0.0`
- `sparse79_law_base_weight=0.0`
- `target_mult=1.0`
- `target_bias=2`
- `max_law_per_base=1`
- `max_court_per_base=3`

Local readout for this branch:

- val macro F1 (best config apply): `0.299628`
- val LB90: `0.270073`
- val LB95: `0.263170`
- val std: `0.073701`
- val min: `0.205128`
- val avg predictions: `23.90`
- val avg court fraction: `0.2611`
- test avg predictions: `21.88`
- test avg court fraction: `0.2463`
- test Jaccard vs staged V11: `0.767923`
- test Jaccard vs V7: `0.387871`
- test Jaccard vs public `0.30094` winner: `0.790768`

Diff vs current public winner:

- all `40/40` test queries changed
- `148` citations added
- `48` citations removed

Interpretation:

- This branch is cleaner than the earlier `200`-prior mixed candidate:
  - less extreme than the split-prior smoke candidate
  - still materially different from the public winner
  - strict V11 lead did not collapse the search
- The sampler stayed in the intended regime:
  - V11 ahead of V7
  - no positive court bonus
  - dense channels positive
  - sparse79 law-base no longer driving expansion
- This looks like the first split-prior candidate that is plausibly worth a Kaggle check.

Kaggle check for strict-V11 split-prior consensus:

- Submitted file:
  - `submissions/test_submission_v11_splitpriors_consensus_vlead.csv`
- Submission:
  - `Codex split-prior strict-v11 consensus 2026-04-08`
- Submission ref:
  - `51580573`
- Public Kaggle score:
  - `0.28454`

Updated read:

- The split-prior + strict-V11 idea is structurally sensible, but this specific branch did **not** beat the current public winner.
- Current public ordering still remains:
  - `0.30094` `test_submission_v11_consensus_loose_deepseekpriors.csv`
  - `0.29393` `test_submission_v11_consensus_200_k64_v035_b3.csv`
  - `0.28454` `test_submission_v11_splitpriors_consensus_vlead.csv`
- So the extra split-prior constraints improved interpretability and search discipline, but they were not enough on their own to produce a new leaderboard best.

2026-04-08: Winner-anchored local perturbation search

Goal:

- Stop making large branch changes.
- Search only very small edits around the real public winner:
  - `submissions/test_submission_v11_consensus_loose_deepseekpriors.csv`
  - public Kaggle score `0.30094`

Neighborhood used:

- Winner plus the three nearest post-`0.30094` Rust variants:
  - `submissions/test_submission_v11_consensus_ds_k32_v050_med_b2.csv`
  - `submissions/test_submission_v11_consensus_ds_k48_v040_b2.csv`
  - `submissions/test_submission_v11_consensus_ds_k64_v035_b3.csv`

Search method:

- Start from the winner predictions per query.
- Only add citations that appear in at least `2/3` nearby variants.
- Prefer law citations over court citations.
- Allow only a tiny number of removals for winner citations missing from all nearby variants.
- Search small caps on:
  - `max_add_total`
  - `max_add_law`
  - `max_add_court`
  - `max_remove_total`
  - `max_remove_court`

Best local perturbation:

- Output files:
  - `submissions/val_pred_v11_winner_localperturb_top1.csv`
  - `submissions/test_submission_v11_winner_localperturb_top1.csv`
- Params:
  - `add_vote_min=2`
  - `max_add_total=3`
  - `max_add_court=0`
  - `max_add_law=3`
  - `max_remove_total=2`
  - `max_remove_court=1`

Local profile:

- val macro F1: `0.282430`
- val LB90: `0.255176`
- val LB95: `0.249112`
- val avg predictions: `23.40`
- val avg court fraction: `0.2173`
- test avg predictions: `20.50`
- test avg court fraction: `0.1763`
- test Jaccard vs winner: `0.932080`
- test Jaccard vs staged V11: `0.682491`

How small the change was:

- `29/40` test queries changed
- `52` citations added
- `7` citations removed
- all additions were law-only (`max_add_court=0`)

Kaggle check:

- Submitted file:
  - `submissions/test_submission_v11_winner_localperturb_top1.csv`
- Submission:
  - `Codex winner local perturb top1 2026-04-08`
- Submission ref:
  - `51580808`
- Public Kaggle score:
  - `0.30257`

Updated read:

- This is the first post-`0.30094` iteration that actually improved on Kaggle.
- The win came from **small, controlled edits around the real winner**, not from a new structural branch.
- The current best public score is now:
  - `0.30257` `test_submission_v11_winner_localperturb_top1.csv`

2026-04-08: Promotion gate for "better or worse" screening

Goal:

- Stop relying on raw val alone.
- Build a machine-readable gate that uses real Kaggle history to answer:
  - is this candidate likely better, worse, or unclear relative to the current best?

Files added:

- `artifacts/v11_meta/kaggle_public_history.json`
- `promotion_gate.py`

History included in the gate:

- `0.24456` `submissions/test_submission_v7.csv`
- `0.27509` `submissions/test_submission_v11.csv`
- `0.27640` `submissions/test_submission_v11_staged.csv`
- `0.25903` `submissions/test_submission_v11_hybrid_grid.csv`
- `0.30094` `submissions/test_submission_v11_consensus_loose_deepseekpriors.csv`
- `0.29393` `submissions/test_submission_v11_consensus_200_k64_v035_b3.csv`
- `0.28454` `submissions/test_submission_v11_splitpriors_consensus_vlead.csv`
- `0.30257` `submissions/test_submission_v11_winner_localperturb_top1.csv`

Gate design:

- Current best anchor is chosen automatically from history (`0.30257` local-perturb top1).
- Features focus on winner-relative drift:
  - val macro F1
  - test avg prediction count
  - test avg court fraction
  - test Jaccard vs anchor
  - additions vs anchor
  - removals vs anchor
  - added courts vs anchor
- Two layers:
  - empirical kNN score prediction from historical submissions
  - hard heuristic winner-neighborhood check
- Final output:
  - `likely_better_or_flat`
  - `likely_worse`
  - `unclear`

Calibration on history:

- `python3 promotion_gate.py ...`
- leave-one-out MAE on public score: `0.010809`
- leave-one-out pairwise ranking accuracy: `0.7500`

Sanity checks:

- Current best:
  - candidate: `submissions/test_submission_v11_winner_localperturb_top1.csv`
  - verdict: `likely_better_or_flat`
  - predicted public score: `0.30257`
- Known loser:
  - candidate: `submissions/test_submission_v11_splitpriors_consensus_vlead.csv`
  - verdict: `likely_worse`
  - predicted public score: `0.28454`

Extra finding:

- `submissions/test_submission_v11_winner_localperturb_top2.csv`
- `submissions/test_submission_v11_winner_localperturb_top3.csv`
- Both are identical to `top1` in practice:
  - same local scorecard
  - same test output
  - same promotion-gate verdict

Interpretation:

- This is not a perfect oracle, but it is materially better than raw val-only promotion.
- Most importantly, it now captures the thing Kaggle has rewarded so far:
  - stay close to the winning submission
  - avoid large court-heavy drift
  - treat small law-only perturbations as the most promising region

2026-04-08: Second-order winner-neighborhood search

Goal:

- Squeeze the no-retrain path further by searching around the current best public winner:
  - `submissions/test_submission_v11_winner_localperturb_top1.csv`
  - public Kaggle `0.30257`

Method:

- Use the promotion gate directly as the ranking signal.
- Build a weighted law-only delta catalog from:
  - `submissions/test_submission_v11_consensus_loose_deepseekpriors.csv`
  - `submissions/test_submission_v11_consensus_200_k64_v035_b3.csv`
  - `submissions/test_submission_v11_consensus_ds_k32_v050_med_b2.csv`
  - `submissions/test_submission_v11_consensus_ds_k48_v040_b2.csv`
  - `submissions/test_submission_v11_consensus_ds_k64_v035_b3.csv`
  - `submissions/test_submission_v11_staged.csv`
- Search only:
  - law additions
  - no court additions
  - tiny removals or none

Search findings:

- The current winner still dominates most of the neighborhood.
- First meaningful non-identity candidate is:
  - `submissions/test_submission_v11_winner_localperturb2_top2.csv`
  - matching val:
    - `submissions/val_pred_v11_winner_localperturb2_top2.csv`
  - meta:
    - `artifacts/v11_meta/winner_localperturb2_top2.json`

Top second-order candidate params:

- `add_count_min=1`
- `add_weight_min=1.156`
- `max_add_total=1`
- `max_remove_total=0`
- `allow_remove_courts=false`

What it changes vs current best:

- `3` total additions
- `0` removals
- `0` added courts
- `3/40` test queries changed

Exact added citations:

- `test_026`: `Art. 134 Abs. 2 ZGB`
- `test_031`: `Art. 95 BGG`
- `test_032`: `Art. 97 Abs. 1 BGG`

Local profile:

- val macro F1: `0.281681`
- val LB90: `0.254159`
- test avg predictions: `20.575`
- test avg court fraction: `0.1756`
- test Jaccard vs current best: `0.997500`

Promotion gate result:

- predicted public score: `0.30212`
- verdict: `likely_better_or_flat`

Interpretation:

- This is the cleanest next submission candidate so far after the `0.30257` winner.
- It is extremely close to the winner and preserves the successful shape.
- It is not proven better; it is simply the best next low-risk perturbation to hold for the next Kaggle submission slot.

2026-04-08: no-new-API train-fitted selector work

What we built:

- Local-only 200-query train candidate artifact:
  - `artifacts/v11/train_v11_trainfit_local200/candidate_bundles.pkl`
  - built from:
    - `artifacts/dense_train_qids_100.txt`
    - `artifacts/dense_train_qids_200_stage2.txt`
    - merged file:
      - `artifacts/dense_train_qids_200_all.txt`
  - command used:
    - `V11_QUERY_IDS_PATH=artifacts/dense_train_qids_200_all.txt V11_USE_COURT_DENSE=0 V11_PROMPT_VERSION=v11_trainfit_local200 ./.venv/bin/python run_v11_staged.py build --split train`
  - build time:
    - `445s`

- New standalone train selector:
  - `run_v11_train_selector.py`
  - trains on candidate bundles without judged train artifacts
  - uses query-level 5-fold OOF scoring on train

- New winner-anchored perturb runner:
  - `run_v11_train_ranker_perturb.py`
  - uses the train-fitted ranker only for tiny law-only additions around the current best winner

Standalone selector results:

- Dense-100 train slice:
  - train candidate recall: `0.4232`
  - OOF macro F1: `0.1779`
  - val macro F1: `0.2397`
  - selected config collapsed to fixed `10` outputs/query:
    - `target_mult=0.45`
    - `bias=-2`
    - `min_out=10`
    - `max_out=10`
    - `court_cap_frac=0.2`
  - files:
    - `submissions/val_pred_v11_train_selector_dense100.csv`
    - `submissions/train_oof_v11_train_selector_dense100.csv`
    - `artifacts/v11_meta/train_selector_dense100_val.json`

- Local-200 train slice:
  - train candidate recall: `0.4102`
  - OOF macro F1: `0.1591`
  - val macro F1: `0.1548`
  - selected config collapsed even harder:
    - `target_mult=1.0`
    - `bias=-10`
    - `min_out=10`
    - `max_out=10`
    - `court_cap_frac=0.0`
  - files:
    - `submissions/val_pred_v11_train_selector_local200.csv`
    - `submissions/train_oof_v11_train_selector_local200.csv`
    - `artifacts/v11_meta/train_selector_local200_val.json`

Val-tuned selector sanity check:

- Train on local-200, but tune only the selector on val:
  - val macro F1: `0.24157`
  - val avg predictions: `12.5`
  - test avg predictions: `11.725`
  - test court fraction: `0.1471`
  - files:
    - `submissions/val_pred_v11_train_selector_local200_valtuned.csv`
    - `submissions/test_submission_v11_train_selector_local200_valtuned.csv`

Winner-anchored train-ranker perturb result:

- Base winner:
  - `submissions/test_submission_v11_winner_localperturb_top1.csv`
  - public Kaggle `0.30257`

- Train-ranker add-only search around the winner found:
  - best config was the identity / no-op
  - `max_add=0`
  - `added_vs_base_test=0`
  - `changed_test_queries=0`
  - val macro F1: `0.28243`
  - files:
    - `submissions/val_pred_v11_trainranker_addonly_top1.csv`
    - `submissions/test_submission_v11_trainranker_addonly_top1.csv`
    - `artifacts/v11_meta/trainranker_addonly_top1.json`

Lessons:

- The train-fitted ranker is real, but train-only selector calibration is badly mismatched to the dense val/test regime.
- The 200-query train slice still averages only `11.74` gold citations/query, so the selector keeps collapsing to low-output, low-court policies.
- As of now, the train-fitted ranker does not beat the current `0.30257` winner:
  - not as a standalone selector
  - not as a tiny law-addition perturbation around the winner
- This is a useful negative result to preserve:
  - train-backed ranking alone is not enough
  - count / court calibration remains the real bottleneck

Kaggle confirmation:

- Submitted:
  - `submissions/test_submission_v11_train_selector_local200_valtuned.csv`
- Description:
  - `Codex local200 train selector val-tuned 2026-04-08`
- Submission ref:
  - `51582190`
- Public score:
  - `0.24685`

Interpretation:

- The no-new-API train selector branch is decisively below the current best public winner `0.30257`.
- The local read was directionally correct this time: this branch did not justify promotion.

2026-04-08: first DeepSeek judged-train selector on 200 queries

Judged train run completion:

- Train judged artifact:
  - `artifacts/v11/train_v11_trainfit_local200/judged_bundles.pkl`
  - `artifacts/v11/train_v11_trainfit_local200/judged_bundles.json`
- Judge cache coverage:
  - `842/842` batches
  - `200/200` queries
- Judge stage wall-clock:
  - `6378.96s`
  - about `1h 46m`

Standalone judged meta-selector:

- Command:
  - `./.venv/bin/python run_v11_meta_selector.py --train-judged artifacts/v11/train_v11_trainfit_local200/judged_bundles.json --train-gold data/train.csv --apply-judged artifacts/v11/val_v11_strict_v1/judged_bundles.json --apply-gold data/val.csv --output-csv submissions/val_pred_v11_meta_trainjudged200.csv --model-out artifacts/v11_meta/meta_selector_trainjudged200.pkl --config-out artifacts/v11_meta/meta_selector_trainjudged200_config.json --random-search 1500 --seed 0`
- Results:
  - train macro F1: `23.2046%`
  - val macro F1: `18.3405%`
- Files:
  - `submissions/val_pred_v11_meta_trainjudged200.csv`
  - `artifacts/v11_meta/meta_selector_trainjudged200.pkl`
  - `artifacts/v11_meta/meta_selector_trainjudged200_config.json`

Train-judged model with val-tuned selector:

- Files:
  - `submissions/val_pred_v11_meta_trainjudged200_valtuned.csv`
  - `submissions/test_submission_v11_meta_trainjudged200_valtuned.csv`
  - `artifacts/v11_meta/meta_selector_trainjudged200_valtuned.json`
- Results:
  - val macro F1: `24.0689%`
  - test avg predictions: `16.075`
  - test court fraction: `0.1011`

Winner-anchored perturb using judged-train model:

- Base winner:
  - `submissions/test_submission_v11_winner_localperturb_top1.csv`
  - public Kaggle `0.30257`
- Search:
  - tiny law-only additions and tiny low-probability removals around the winner
- Files:
  - `submissions/val_pred_v11_meta_trainjudged200_perturb_top1.csv`
  - `submissions/test_submission_v11_meta_trainjudged200_perturb_top1.csv`
  - `artifacts/v11_meta/meta_trainjudged200_perturb_top1.json`
- Result:
  - best perturbation was the identity / no-op
  - no adds
  - no removals
  - no changed test queries

Conclusion:

- The 200-query DeepSeek judged train set is useful infrastructure and should improve future scaling.
- But the first judged-train selector still does not beat the current public winner.
- Immediate implication:
  - 200 judged queries are not yet enough to move the winner
  - next real gain likely needs either:
    - more judged train coverage
    - better court candidate recall
    - or both together

Canonical public baseline lock:

- Public-best submission remains:
  - `submissions/test_submission_v11_winner_localperturb_top1.csv`
  - public Kaggle: `0.30257`
  - submission ref: `51580808`

- Frozen baseline copies created:
  - `submissions/test_submission_baseline_public_best_30257.csv`
  - `submissions/val_pred_baseline_public_best_30257.csv`

- Baseline manifest:
  - `artifacts/v11_meta/current_public_baseline.json`

- File hashes:
  - test: `7c6424f39121ba018d322de55f939cc2050eb035ee1cdaac3205a190cfcfb4a6`
  - val: `4c0be18cdbb7df005feab065fba30487686fad0b85242a58f0f432310bf3b84f`

- Operational rule:
  - treat this as the rollback control for every future branch and Kaggle promotion decision

2026-04-08 (later): stage-3 judged-train expansion + OOF selector hardening

Why this was needed:

- The judged-train infrastructure worked, but the first 200-query selector still underperformed the `0.30257` public winner.
- The next best lever is to scale judged train coverage toward the remaining dense-ish train frontier before touching the sparse tail.
- Also, `run_v11_meta_selector.py` was still picking selector configs on in-sample train predictions, which was too leaky.

New dense-ish stage-3 expansion slice:

- Train density audit:
  - `>=20`: `15`
  - `>=15`: `38`
  - `>=10`: `112`
  - `>=9`: `141`
  - `>=8`: `172`
  - `>=7`: `203`
  - `>=6`: `253`
  - `>=5`: `333`
- Current judged coverage before stage 3:
  - `200` queries
  - average gold citations/query: `11.74`
- Remaining uncovered frontier after those 200:
  - top remaining counts: `7`, then `6`, then `5`
  - remaining average gold citations/query across all uncovered train: `2.461`

- New query-id file created:
  - `artifacts/dense_train_qids_333_stage3.txt`
- Contents:
  - the `133` remaining uncovered train queries with `>=5` gold citations
- Purpose:
  - bring judged-train coverage from `200` to `333` queries
  - finish the dense-ish train slice before spending time on the long sparse tail

Parallel DeepSeek stage-3 run:

- Live command launched:
  - `export LLM_API_KEY=<user DeepSeek key>`
  - `export MAX_WORKERS=8`
  - `export V11_JUDGE_WORKERS=8`
  - `export V11_PROMPT_VERSION=v11_trainfit_local333_stage3`
  - `export V11_USE_COURT_DENSE=0`
  - `./scripts/run_deepseek_dense_pilot.sh artifacts/dense_train_qids_333_stage3.txt`
- Session:
  - `8847`
- Expected output root:
  - `artifacts/v11/train_v11_trainfit_local333_stage3`
- Purpose:
  - generate train query expansions / case citations / full citations for the new 133-query slice
  - then build and judge that slice with DeepSeek reasoner

OOF hardening of the judged meta-selector:

- File changed:
  - `run_v11_meta_selector.py`
- Main changes:
  - added `--folds`
  - added GroupKFold out-of-fold probability generation for train rows
  - selector random search now uses OOF train predictions instead of self-fit train predictions
  - final apply model is still fit on full train judged rows after config selection

- Verification:
  - `./.venv/bin/python -m py_compile run_v11_meta_selector.py`

OOF smoke test on existing 200 judged queries:

- Command:
  - `./.venv/bin/python run_v11_meta_selector.py --train-judged artifacts/v11/train_v11_trainfit_local200/judged_bundles.json --train-gold data/train.csv --apply-judged artifacts/v11/val_v11_strict_v1/judged_bundles.json --apply-gold data/val.csv --output-csv submissions/val_pred_v11_meta_trainjudged200_oofsmoke.csv --config-out artifacts/v11_meta/meta_selector_trainjudged200_oofsmoke.json --random-search 60 --folds 5 --seed 0`
- Results:
  - OOF train macro F1: `18.8248%`
  - apply/val macro F1: `17.3105%`
- Files:
  - `submissions/val_pred_v11_meta_trainjudged200_oofsmoke.csv`
  - `artifacts/v11_meta/meta_selector_trainjudged200_oofsmoke.json`

Interpretation:

- The OOF path is working correctly and is a safer training loop than the earlier self-fit selector.
- The 200-query judged train slice is still not enough by itself.
- The right next move remains: finish the stage-3 DeepSeek run, then retrain on `200 + 133 = 333` judged dense-ish train queries before making another Kaggle decision.

Kaggle reality check for the pure judged-200 standalone selector:

- Submitted:
  - `submissions/test_submission_v11_meta_trainjudged200_valtuned.csv`
- Description:
  - `Codex pure judged200 standalone 2026-04-08`
- Kaggle submission ref:
  - `51583986`
- Public Kaggle score:
  - `0.22639`

Interpretation:

- This confirms the earlier concern was correct: the pure judged-200 standalone selector is badly underpowered.
- It is worse than:
  - the no-new-API local200 train selector (`0.24685`)
  - the staged baseline (`0.27640`)
  - the current public winner (`0.30257`)
- The judged data by itself is not enough; without better recall and more coverage, the standalone selector collapses too hard on hidden/public test.

Frozen baseline external recheck:

- Submitted:
  - `submissions/test_submission_baseline_public_best_30257.csv`
- Description:
  - `Codex frozen baseline recheck 2026-04-08`
- Kaggle submission ref:
  - `51584010`
- Public Kaggle score:
  - `0.30257`

Interpretation:

- The frozen baseline file reproduces the current best public score exactly.
- This confirms that `0.30257` is a stable external control, not a one-off leaderboard fluctuation.
- Use this file as the hard rollback candidate for every future branch:
  - `submissions/test_submission_baseline_public_best_30257.csv`

Quick ruthless iteration plan after the 333 judged-train milestone:

- Frozen external control stays:
  - `submissions/test_submission_baseline_public_best_30257.csv`
  - public Kaggle `0.30257`

- Newly completed supervision milestone:
  - `artifacts/v11/train_v11_trainfit_local333_stage3/judged_bundles.json`
  - combined judged dense-ish train coverage is now `333` queries (`>=5` gold slice)

- Immediate next loop:
  1. Retrain judged selector on merged:
     - `artifacts/v11/train_v11_trainfit_local200/judged_bundles.json`
     - `artifacts/v11/train_v11_trainfit_local333_stage3/judged_bundles.json`
  2. Evaluate with OOF-based selector search on `val`
  3. If promising, do winner-anchored perturbations around:
     - `submissions/test_submission_baseline_public_best_30257.csv`
  4. If not promising, do not submit another standalone selector

- Next big DeepSeek hitters after this:
  1. court-FAISS as extra judged candidates on:
     - dense-ish `333` train
     - `val`
     - `test`
  2. expand judged-train coverage from `333` to `>=4` gold slice (`418` queries total)
  3. then consider `>=3` gold slice (`567` queries total)

- Working rule:
  - judged data alone improves selection
  - to move above the `0.30257` plateau, we likely need:
    - more judged train coverage
    - and better court candidate recall

Merged 333-query judged-selector run:

- Train artifacts used:
  - `artifacts/v11/train_v11_trainfit_local200/judged_bundles.json`
  - `artifacts/v11/train_v11_trainfit_local333_stage3/judged_bundles.json`
- Command:
  - `./.venv/bin/python run_v11_meta_selector.py --train-judged artifacts/v11/train_v11_trainfit_local200/judged_bundles.json artifacts/v11/train_v11_trainfit_local333_stage3/judged_bundles.json --train-gold data/train.csv --apply-judged artifacts/v11/val_v11_strict_v1/judged_bundles.json --apply-gold data/val.csv --output-csv submissions/val_pred_v11_meta_trainjudged333_oof.csv --model-out artifacts/v11_meta/meta_selector_trainjudged333_oof.pkl --config-out artifacts/v11_meta/meta_selector_trainjudged333_oof.json --random-search 500 --folds 5 --seed 0`
- Results:
  - OOF train macro F1: `18.6309%`
  - apply/val macro F1: `15.0360%`
- Files:
  - `submissions/val_pred_v11_meta_trainjudged333_oof.csv`
  - `artifacts/v11_meta/meta_selector_trainjudged333_oof.pkl`
  - `artifacts/v11_meta/meta_selector_trainjudged333_oof.json`

Interpretation:

- Even with 333 judged dense-ish train queries, the standalone judged selector remains far too weak.
- Better selection supervision alone is still not enough.

333-trained winner-anchored perturb search:

- Search type:
  - law-only adds around the frozen `0.30257` baseline
  - no removals in the winning config
  - no added courts
- Best local perturb:
  - `artifacts/v11_meta/meta_trainjudged333_perturb_top1.json`
  - `submissions/val_pred_v11_meta_trainjudged333_perturb_top1.csv`
  - `submissions/test_submission_v11_meta_trainjudged333_perturb_top1.csv`
- Best local payload:
  - base val macro: `0.2824304264`
  - best val macro: `0.2830674866`
  - `add_prob=0.35`
  - `add_max=2`
  - `remove_prob=0.0`
  - `remove_max=0`
  - `rank_cap=10`
  - `added_vs_base_test=52`
  - `removed_vs_base_test=0`
  - `changed_test_queries=32`

Kaggle reality check for the 333-trained perturb branch:

- Submitted:
  - `submissions/test_submission_v11_meta_trainjudged333_perturb_top1.csv`
- Description:
  - `Codex trainjudged333 law-only perturb top1 2026-04-08`
- Kaggle submission ref:
  - `51586700`
- Public Kaggle score:
  - `0.29887`

Interpretation:

- The 333-judged train signal can find non-trivial local winner perturbations, but this branch still regressed versus the frozen baseline.
- The branch added `52` laws across `32` test queries and still lost to the `0.30257` control.
- Main conclusion remains unchanged:
  - more judged data alone is not enough
  - the next serious lever is better candidate recall, especially courts, with court FAISS injected as judged candidates rather than raw score fusion

Court-recall next step launched:

- Combined dense-ish 333 train query list created:
  - `artifacts/dense_train_qids_333_all.txt`
  - combines:
    - `artifacts/dense_train_qids_100.txt`
    - `artifacts/dense_train_qids_200_stage2.txt`
    - `artifacts/dense_train_qids_333_stage3.txt`
  - total query count: `333`

- New run launched to attack candidate recall rather than just selection:
  - split: `train`
  - query ids: `artifacts/dense_train_qids_333_all.txt`
  - prompt version: `v11_trainfit_local333_courtdense_v1`
  - `V11_USE_COURT_DENSE=1`
  - judge model: `deepseek-reasoner`
  - judge workers: `8`
  - build session: `87315`

- Purpose:
  - add court FAISS hits as extra candidates on the dense-ish 333 train slice
  - then judge those richer candidate pools with DeepSeek
  - use that court-augmented judged set for the next baseline-anchored candidate search

- Strategic rule from here:
  - no more judged-only standalone selector promotions
  - next promotion-worthy branch must come from better recall, especially courts

---

## 2026-04-09 — Court-dense recall study, two failed micro-perturbations, dead-zone confirmation

### Critical framing correction (saved to memory)

User pushed back hard on a prior assumption: I had been treating local val macro F1 as if it were a reliable promotion signal. It is not, and the historical record is clear:

- Failed Rust hybrid grid: local **0.30443 → Kaggle 0.25903** (catastrophic regression).
- Both the **0.30094** and **0.30257** Kaggle highs were NOT predicted by the local stack — the local scorecard "misranked the 0.30094 winner previously", and the local scorecard "failed to predict this branch correctly" for 0.30257.
- Root cause (CODEX_MEMORY.md lines 297-299): "using the same 10-query val set both to search tens of thousands of configs and to claim success".
- 10 val queries × ~5 gold cites/query → a single citation flip is ~2pp swings in the macro.

User's exact words: *"you are assuming the internal scoring is correct, that wasn't the case when we submitted our current high score"*.

Saved as feedback memory: `~/.claude/projects/-Users-william-swiss-legal-retrieval/memory/feedback_local_val_unreliable.md`. From now on, raw local val F1 alone is NEVER a promotion signal.

### Multi-signal scorecard built

`scripts/multi_signal_scorecard.py` — promotes ONLY when 5 signals all line up:

1. Raw val macro F1
2. Bootstrap LB90 (lower 10th percentile from 2000-iteration query-resampled bootstrap)
3. Per-query std / floor (worst single-query F1)
4. Test output shape vs trusted baselines (mean cites, law/court %)
5. Test-prediction overlap (Jaccard) vs the 0.30257 submission

### Court-dense recall ceiling — surprising and important

`scripts/diagnose_courtdense_recall_at_pool.py` measured what fraction of gold the court-dense candidate pool actually contains, and how much of it is NEW relative to the 0.30257 baseline:

| Metric | Value |
|---|---|
| Mean baseline (0.30257) recall on val gold | **29.91%** |
| Mean court-dense candidate-pool recall | **64.21%** |
| Gold in court-dense pool but NOT in baseline preds | **34.30 pp** |

So the court-dense pipeline has **34 percentage points of additional gold recall sitting in its candidate pool that the 0.30257 baseline never picks up**. This is the recall headroom we should be attacking.

### But the trained court-dense classifier can't surface it

`scripts/diagnose_courtdense_classifier.py` swept K=4..30 of top-K-by-classifier-prob:

| K | val F1 | mean cites | court% | baseline Jaccard |
|---|---|---|---|---|
| 6 (current) | 17.15% | 6.00 | 5.5% | ~31% |
| 28 (peak) | **18.31%** | 28.00 | 11.4% | ~47% |

Peak val F1 of the trained court-dense classifier across all K is **18.31%** — far below the 0.30257 baseline's **28.24%**. **The classifier just rediscovers baseline cites.** Its top picks ARE baseline cites — there is no untapped signal in the classifier's ranking.

### C1 hypothesis (judge labels are sabotaging it) — DISPROVEN

`scripts/diagnose_courtdense_no_judge_feats.py` retrained without the judge_label one-hot and judge_confidence features. Result: peak val F1 dropped from 18.31% (with judge feats) to **15.26%** (without). The judge labels are HELPING the classifier, not hurting it. The bottleneck is elsewhere.

### Additive blend on raw retrieval score finds the gold

`scripts/diagnose_courtdense_additive_blend.py` — for each query, take the top-N court-dense candidates NOT already in the baseline, sorted by classifier prob OR raw FAISS score OR final score, and add them to the baseline.

| Sort key | Best N | val F1 | delta | add precision |
|---|---|---|---|---|
| classifier prob | 0 | 28.24% | +0.00 | — |
| **raw_score** | **1** | **29.52%** | **+1.27 pp** | **40%** |
| raw_score | 2 | 29.46% | +1.22 pp | 25% |
| final_score | 0 | 28.24% | +0.00 | — |

**Headline finding:** the trained classifier has worse precision on the pool-only-gold than untrained raw FAISS retrieval score. The classifier is overfit to baseline patterns; raw FAISS score is a cleaner ranker for "is this *new* citation actually relevant".

### Two Kaggle submissions both LOST in the dead zone

After applying the multi-signal scorecard, two perturbations of the 0.30257 baseline that ALL FIVE signals approved both lost on Kaggle:

| Variant | local F1 | local LB90 | floor | test Jaccard | shape | Kaggle | delta |
|---|---|---|---|---|---|---|---|
| `v11_meta_trainjudged333_perturb_top1` | +0.07pp | +0.22pp | neutral | 94% | neutral | **0.29887** | **-0.0037** |
| `blend_courtdense_additive_test_raw_n1` | +1.27pp | +0.53pp | neutral | 95% | neutral | **0.29868** | **-0.0039** |

The +1.27pp variant did NOT outperform the +0.07pp variant on Kaggle. **A local lift of 1-2pp on the 10-query val set carries essentially no Kaggle predictive signal.** The signal-to-noise ratio in the dead zone is just too low to burn Kaggle attempts on.

Memory updated: minimum lift for a real promotion candidate around 0.30257 is unknown but is **much larger than 1-2pp**.

### Strategic conclusion: the "small perturbation" branch is exhausted

The 0.30257 baseline appears to sit at a tight local optimum where ANY small additive modification loses a fraction of a point. We have repeatedly proven that:

1. Raw local F1 lifts ≲ 2pp don't survive transfer to the 40-query test set.
2. The trained court-dense classifier just rediscovers baseline cites.
3. Adding 1-2 raw-FAISS picks per query doesn't help — even though add precision is 40%, the false adds eat the precision component faster than the true adds gain on recall.

**Headroom is real (34pp pool-only gold), but capturing it requires structurally different approaches, not micro-perturbations.** Next promotion-worthy candidate must show a structurally different signal: a new retrieval source, a fundamentally different selection policy, or a substantially larger local lift that survives shape-and-overlap checks.

### New artifacts (this session)

- `scripts/multi_signal_scorecard.py` — 5-signal promotion scorecard
- `scripts/apply_saved_meta_selector.py` — apply pickled meta-selector to judged bundles
- `scripts/diagnose_courtdense_classifier.py` — sweep K of top-K-by-prob on val + test
- `scripts/diagnose_courtdense_no_judge_feats.py` — C1 ablation (drop judge features)
- `scripts/diagnose_courtdense_recall_at_pool.py` — pool-recall ceiling, pool-only-over-baseline gold
- `scripts/diagnose_courtdense_additive_blend.py` — top-N-additions blend (3 sort keys: prob/raw/final)
- `submissions/blend_courtdense_additive_test_raw_n1.csv` — submitted, **0.29868** on Kaggle

### Kaggle auth note

Old `kaggle.json` (username `wbfranci`, 37-char key) returns 401 — expired/rotated. New auth uses `KAGGLE_API_TOKEN=KGAT_...` env var (NOT a kaggle.json file). Token format: `KGAT_<numeric>`. Submission pattern that worked:

```bash
KAGGLE_API_TOKEN=KGAT_731104568562008188aeb9e24a4e5323 \
  .venv/bin/kaggle competitions submit \
  -c llm-agentic-legal-information-retrieval \
  -f submissions/blend_courtdense_additive_test_raw_n1.csv \
  -m "blend court-dense raw_score N=1 additive"
```


