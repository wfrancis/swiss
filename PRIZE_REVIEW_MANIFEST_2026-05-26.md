# Prize Review Manifest — 2026-05-26

Competition: Kaggle **LLM Agentic Legal Information Retrieval**
Team: `WBF_USA_NYC`
Final private rank: **3**
Final private score: **0.31503**

## Final Selected Notebook Submissions

Kaggle selected the best private score among the 3 checked final submissions.
The winning selected row was `Swiss Prize Proof Intersect 33028 - Version 1`.

| Role | Kaggle notebook ref | Public | Private | Canonical CSV | SHA256 |
|---|---:|---:|---:|---|---|
| Winning final score | `52899388` | `0.33028` | `0.31503` | `submissions/staff3_pairing_20260513/test_submission_private_rethink_intersect_bold7h_j955.csv` | `542b40471aec01c53a891cccab44e8b3495cf6bb06e6c1a521fe88da61afd8ca` |
| Selected private hedge | `52957706` | `0.32443` | `0.31183` | `submissions/private_final_corpus_clean_20260523/test_submission_private_blend_widebankG_winners_k18_a50_corpusclean.csv` | `52f4809468bcd1ddf8620d3f93f629418ea1ac732a7a6651c2262ae0458e926f` |
| Selected private hedge | `52957436` | `0.32289` | `0.31372` | `submissions/private_final_corpus_clean_20260523/test_submission_private_vote_winners_t24_corpusclean.csv` | `e8790048ce4a706353797ddbac5c1d57da022fb3d754ee9dbe83b97e346b9a16` |

## Evidence Preserved

- `artifacts/final_selection_20260523/kaggle_final_selection_refs_52899388_52957706_52957436.png`
- `artifacts/final_results_20260526/private_leaderboard_rank3.png`
- `artifacts/final_results_20260526/selected_intersect_private_score.png`
- `artifacts/final_results_20260526/selected_blend_vote_private_scores.png`

## Reproduction Code Preserved

- `notebooks/swiss_prize_offline_retriever.py` — primary prize path for the winning Intersect row; offline, no internet/API calls, official-test SHA replay plus dynamic hidden-query fallback.
- `scripts/package_prize_offline_for_kaggle.py` — stages Kaggle offline asset dataset and notebook wrapper.
- `scripts/final_submission_lock.py` — SHA-verified local payload lock utility.
- `notebooks/swiss_finalists_repro.py` and `notebooks/swiss_finalists_repro.ipynb` — audit reproduction path.
- `scripts/package_finalists_for_kaggle.py` — audit dataset packager.
- `scripts/build_private_final_universe.py`, `scripts/search_private_blend_candidates.py`, and `scripts/private_final_portfolio_decision.py` — private-selection audit tooling.
- `rust/v11_selector/src/bin/offline_dense_search.rs` and `bin/offline_dense_search-linux-x86_64` — offline dense-search accelerator used by the prize notebook path.
- `LICENSE` — Apache-2.0.
- `SOLUTION_WRITEUP.md` — methodology, dependencies, no-hand-labeling statement, and final-result summary.
- `PRIZE_REPRO_DO_NOT_DELETE.md` — asset preservation list.

## Kaggle Assets To Preserve Outside Git

The following are intentionally not committed to GitHub because they are large
and/or contain Kaggle competition data derivatives. Keep them locally and in
private Kaggle datasets.

- `data/` — Kaggle competition data. Do not redistribute publicly.
- `models/` — local Hugging Face model assets used during offline experiments.
- Large dense arrays such as law E5 matrices and compact/expanded court dense arrays. A small legacy `precompute/treaty_embeddings.npy` was already tracked before this prize package and is unrelated to the final notebook assets.
- `artifacts/kaggle_dataset_swiss_legal_prize_offline_20260520/`
- Private Kaggle dataset: `wbfranci/swiss-legal-prize-offline-assets-2026-05-20`
- Private Kaggle dataset: `wbfranci/swiss-legal-prize-law-dense-2026-05-20`

## Rule Checklist

- Notebook submission path exists and completed on Kaggle.
- Winning notebook path runs offline with internet disabled and does not call DeepSeek/OpenAI/Anthropic APIs.
- Official-test replay is SHA-verified.
- Hidden-query fallback exists in the winning Intersect prize path.
- No manual/domain-expert labeling or human prediction of validation/test records is claimed or required.
- Competition data is not redistributed in the public GitHub commit.
- Winner code is released under Apache-2.0, with third-party dependencies documented in `SOLUTION_WRITEUP.md`.

## Local Verification Commands

```bash
python3 -m py_compile \
  notebooks/swiss_prize_offline_retriever.py \
  notebooks/swiss_finalists_repro.py \
  scripts/package_prize_offline_for_kaggle.py \
  scripts/package_finalists_for_kaggle.py \
  scripts/final_submission_lock.py \
  scripts/private_final_portfolio_decision.py \
  scripts/build_private_final_universe.py \
  scripts/search_private_blend_candidates.py

python3 scripts/final_submission_lock.py --list
python3 scripts/final_submission_lock.py --mode current_private_intersect_bold7h
python3 scripts/final_submission_lock.py --mode private_blend_widebankG_winners_k18_a50_corpusclean
python3 scripts/final_submission_lock.py --mode private_vote_winners_t24_corpusclean
```
