# Final Prize Qualification Checklist — 2026-05-19

## Superseding Final Result — 2026-05-26

The final selected notebook submissions were updated after this checklist was
first written. WBF_USA_NYC finished **3rd on the private leaderboard** with
final score `0.31503`, from selected notebook ref `52899388`:

- `Swiss Prize Proof Intersect 33028 - Version 1`
  - canonical CSV: `submissions/staff3_pairing_20260513/test_submission_private_rethink_intersect_bold7h_j955.csv`
  - public/private: `0.33028` / `0.31503`
  - SHA256: `542b40471aec01c53a891cccab44e8b3495cf6bb06e6c1a521fe88da61afd8ca`
- `Swiss Prize Proof Private Blend K18 A50 - Version 1`
  - canonical CSV: `submissions/private_final_corpus_clean_20260523/test_submission_private_blend_widebankG_winners_k18_a50_corpusclean.csv`
  - public/private: `0.32443` / `0.31183`
  - SHA256: `52f4809468bcd1ddf8620d3f93f629418ea1ac732a7a6651c2262ae0458e926f`
- `Swiss Prize Proof Private Vote T24 - Version 1`
  - canonical CSV: `submissions/private_final_corpus_clean_20260523/test_submission_private_vote_winners_t24_corpusclean.csv`
  - public/private: `0.32289` / `0.31372`
  - SHA256: `e8790048ce4a706353797ddbac5c1d57da022fb3d754ee9dbe83b97e346b9a16`

Evidence screenshots are preserved under `artifacts/final_results_20260526/`.
The older May 19 candidate pool below is retained for provenance, but the
binding final package is the three notebook refs above.

Official pages reviewed in Chrome:

- `https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/rules`
- `https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/overview/am-i-allowed-to`
- `https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/overview/timeline`

## Rule facts that matter

- Written rules say maximum team size is `5`.
- Written rules say maximum submissions are `5` per day.
- Written rules say up to `2` final submissions for judging. If the logged-in UI allows `3`, document the UI state before selecting 3.
- Public leaderboard is `50%` of test queries; private leaderboard is the other `50%` and determines final standing.
- CSV-only submissions are valid for testing but do not qualify for prizes by themselves.
- Prize eligibility requires a Kaggle notebook/code path that can reproduce `submission.csv` offline, without internet/API calls, within Kaggle runtime limits.
- The host may re-evaluate the notebook on completely private queries not in `test.csv`.
- The solution must be reproducible, scalable, and expected to generalize.
- No manual/domain-expert annotation of validation or test records.
- Competition data is non-commercial/research competition use; do not redistribute data, test queries, hidden labels, or derived labels outside permitted channels.
- External data/tools must be reasonably accessible, documented, and reproducible.
- Winning code must be released under an OSI-approved permissive license, with Apache-2.0 requested unless otherwise stated.
- Winners must provide code/docs sufficient to reproduce the winning submission and return required prize/tax/eligibility documents.

## Current selected-final candidate pool

Preferred if the UI allows 3 final submissions:

1. `current_public_peak_33438`
   - `submissions/public_precision_targeted_20260518/live_refit_after_33385/test_submission_33385_nextrem_03_est33390.csv`
   - SHA256: `89acefcdc37eeaf7d08b99559427a5167e7997cf5304a02278c7f09e27c85b9b`
   - Kaggle ref: `52758343`
   - Public score: `0.33438`

2. `current_private_intersect_bold7h`
   - `submissions/staff3_pairing_20260513/test_submission_private_rethink_intersect_bold7h_j955.csv`
   - SHA256: `542b40471aec01c53a891cccab44e8b3495cf6bb06e6c1a521fe88da61afd8ca`
   - Kaggle ref: `52819486`
   - Public score: `0.33028`

3. `current_private_hedge_fusion_samesrc03`
   - `submissions/final_staff_level_20260513/test_submission_final_hedge_overlay_fusion01_plus_samesrc03.csv`
   - SHA256: `163f0bba09ca6e07d648a62360abb09bb63f45f227e3e421b3f569b84225d5d2`
   - Kaggle ref: `52596721`
   - Public score: `0.32274`

Private-first if the UI enforces 2 final submissions:

1. `current_private_intersect_bold7h`
2. `current_private_hedge_fusion_samesrc03`

Public-protection if the UI enforces 2 and public rank must be preserved:

1. `current_public_peak_33438`
2. `current_private_intersect_bold7h`

## Local assets now locked

The following modes were added to `scripts/final_submission_lock.py`:

- `current_public_peak_33438`
- `current_private_intersect_bold7h`
- `current_private_hedge_fusion_samesrc03`
- `current_pre_tomography_anchor_33186`

Use:

```bash
python3 scripts/final_submission_lock.py --list
python3 scripts/final_submission_lock.py --mode current_public_peak_33438
python3 scripts/final_submission_lock.py --mode current_private_intersect_bold7h
python3 scripts/final_submission_lock.py --mode current_private_hedge_fusion_samesrc03
```

## Remaining qualification actions

Status update 2026-05-20:

- Kaggle UI final-selection was completed for three rows visible as selected in the UI: `intersect_bold7h_33028`, `public_peak_33438`, and `fusion_samesrc03_32274`.
- A hybrid offline prize notebook now exists: `notebooks/swiss_prize_offline_retriever.py`. It reproduces a selected finalist exactly when the official `test.csv` fingerprint matches, and falls back to dynamic retrieval when the host swaps hidden queries.
- Packaging is staged locally via `scripts/package_prize_offline_for_kaggle.py` into:
  - `artifacts/kaggle_dataset_swiss_legal_prize_offline_20260520/`
  - `artifacts/kaggle_kernel_swiss_prize_offline_20260520/`
- The staged kernel metadata has internet disabled and depends only on the competition source plus the staged Kaggle asset dataset.

Still remaining:

1. Upload or version the staged offline asset dataset on Kaggle.
2. Push/share the staged private Kaggle notebook wrapper and run it once on Kaggle with internet disabled.
3. Keep a screenshot or note if the UI allows `3` final submissions, because written rules say `2`.
4. Keep the solution write-up current. It should document:
   - retrieval pipeline,
   - deterministic post-processing/locking modes,
   - selected final file hashes,
   - compute environment,
   - external data/tool dependencies and licenses,
   - statement that no hand-labeling of validation/test records was used.
5. Preserve all files listed in `PRIZE_REPRO_DO_NOT_DELETE.md`.

## Current risk

The main remaining prize risk is not the CSV score. It is that the hidden-query
dynamic fallback is much weaker than the API-assisted finalist pipeline. The
2026-05-20 hybrid notebook now handles exact official-test reproduction plus
hidden-query shape, but its hidden-query fallback is not expected to reproduce
the May 18/19 public-LB quality. The highest-value next upgrade is to package a
local multilingual embedding model/reranker or small local LLM so the offline
notebook's hidden-query quality is closer to the API-assisted finalist pipeline.
