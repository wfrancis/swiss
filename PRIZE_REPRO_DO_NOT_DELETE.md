# Prize Repro Assets — DO NOT DELETE

Date: 2026-04-28
Final result update: 2026-05-26

This repo is currently in prize contention for the Kaggle **LLM Agentic Legal
Information Retrieval** competition. Do not delete the files/directories listed
here unless the user explicitly says to abandon prize eligibility.

## Final Private Leaderboard Result

WBF_USA_NYC finished **3rd on the private leaderboard**. Kaggle used the best
private score among the selected notebook submissions:

- Winner/final score: `Swiss Prize Proof Intersect 33028 - Version 1`
  - Kaggle notebook ref: `52899388`
  - Public score: `0.33028`
  - Private score / final score: `0.31503`
  - Canonical CSV: `submissions/staff3_pairing_20260513/test_submission_private_rethink_intersect_bold7h_j955.csv`
  - SHA256: `542b40471aec01c53a891cccab44e8b3495cf6bb06e6c1a521fe88da61afd8ca`
- Selected hedge: `Swiss Prize Proof Private Blend K18 A50 - Version 1`
  - Kaggle notebook ref: `52957706`
  - Public score: `0.32443`
  - Private score: `0.31183`
  - Canonical CSV: `submissions/private_final_corpus_clean_20260523/test_submission_private_blend_widebankG_winners_k18_a50_corpusclean.csv`
  - SHA256: `52f4809468bcd1ddf8620d3f93f629418ea1ac732a7a6651c2262ae0458e926f`
- Selected hedge: `Swiss Prize Proof Private Vote T24 - Version 1`
  - Kaggle notebook ref: `52957436`
  - Public score: `0.32289`
  - Private score: `0.31372`
  - Canonical CSV: `submissions/private_final_corpus_clean_20260523/test_submission_private_vote_winners_t24_corpusclean.csv`
  - SHA256: `e8790048ce4a706353797ddbac5c1d57da022fb3d754ee9dbe83b97e346b9a16`

Evidence to preserve:

- `artifacts/final_selection_20260523/kaggle_final_selection_refs_52899388_52957706_52957436.png`
- `artifacts/final_results_20260526/private_leaderboard_rank3.png`
- `artifacts/final_results_20260526/selected_intersect_private_score.png`
- `artifacts/final_results_20260526/selected_blend_vote_private_scores.png`

Kaggle distinction:

- Ordinary submissions may be CSVs.
- Prize eligibility requires an offline Kaggle notebook/code path that can
  reproduce `submission.csv` without internet/API calls and with documented
  assets, dependencies, and methodology.
- The logged-in UI allowed and used 3 selected notebook submissions for this
  competition. Preserve the selected three plus the top candidate pool below.

## Candidate Payloads To Preserve

Current final-selection pool added 2026-05-19:

- `submissions/public_precision_targeted_20260518/live_refit_after_33385/test_submission_33385_nextrem_03_est33390.csv`
  - Public score: `0.33438`
  - SHA256: `89acefcdc37eeaf7d08b99559427a5167e7997cf5304a02278c7f09e27c85b9b`
  - Role: current public peak; use at most one public-LB-tuned leg.
- `submissions/staff3_pairing_20260513/test_submission_private_rethink_intersect_bold7h_j955.csv`
  - Public score: `0.33028`
  - SHA256: `542b40471aec01c53a891cccab44e8b3495cf6bb06e6c1a521fe88da61afd8ca`
  - Role: private precision/intersection hedge; submitted 2026-05-19.
- `submissions/final_staff_level_20260513/test_submission_final_hedge_overlay_fusion01_plus_samesrc03.csv`
  - Public score: `0.32274`
  - SHA256: `163f0bba09ca6e07d648a62360abb09bb63f45f227e3e421b3f569b84225d5d2`
  - Role: private recall/diversity hedge from staff-level adversarial shortlist.
- `submissions/prepared_public_20260513/test_submission_private_rethink_overlay_samesrc_02.csv`
  - Public score: `0.33186`
  - SHA256: `6c7782dccedaf9faf49808ec307521a326424a12c43c3c7ce931978374920e6f`
  - Role: pre-late-tomography public anchor; backup if final UI allows only two and public rank is prioritized.
- `submissions/staff3_pairing_20260513/val_pred_private_rethink_intersect_bold7h_j955.csv`
- `submissions/final_staff_level_20260513/val_pred_final_hedge_overlay_fusion01_plus_samesrc03.csv`
- `submissions/prepared_public_20260513/val_pred_private_rethink_overlay_samesrc_02.csv`
- `artifacts/final_selection_20260519/private_split_final_audit/`

Simulated-private-split challengers added 2026-05-22:

- `submissions/test_submission_bold_7h_widebankG_hailmary.csv`
  - Public score: `0.30702`
  - SHA256: `bd603ad9497e7f1a32b4c18067454ce0b869aa09728ebc7ff8d68af06e54fa6c`
  - Role: strongest already-submitted private-diverse hedge in the expanded audit; safe no-new-submission replacement candidate for `fusion_samesrc03_32274`.
- `submissions/private_final_blend_20260522/test_submission_private_blend_widebankG_winners_k18_a50.csv`
  - Public score: `unsubmitted`
  - SHA256: `1164bb097cda46ffee43324fcbef498f8daf36edf4bb681cbdeaf88548caccea`
  - Role: generated simulated-private-split blend; mean/LOO-favored challenger when paired with `intersect_bold7h_33028`.
- `submissions/private_final_blend_20260522/test_submission_private_vote_winners_t24.csv`
  - Public score: `unsubmitted`
  - SHA256: `26d371cc759f1491b0e14c7d892d5baea7e100af68e720420e17c179caf85b65`
  - Role: generated simulated-private-split vote; stricter lower-tail challenger, near-superset of the blend and not an independent co-final with it.
- `submissions/val_pred_bold_7h_widebankG_hailmary.csv`
- `submissions/private_final_blend_20260522/val_pred_private_blend_widebankG_winners_k18_a50.csv`
- `submissions/private_final_blend_20260522/val_pred_private_vote_winners_t24.csv`
- `scripts/build_private_final_universe.py`
- `scripts/search_private_blend_candidates.py`
- `scripts/private_final_portfolio_decision.py`
- `artifacts/private_final_recheck_20260522/private_blend_search_v1/`
- `artifacts/private_final_recheck_20260522/universe_v5_blends220/`
- `artifacts/private_final_recheck_20260522/combined_private_sweep_v5_blends220_fullpairs/`
- `artifacts/private_final_recheck_20260522/decision_audit_v5_blends220_vs_current_hailmary/`

Corpus-clean challengers added 2026-05-23:

- `submissions/private_final_corpus_clean_20260523/test_submission_private_blend_widebankG_winners_k18_a50_corpusclean.csv`
  - Public score: `unsubmitted`
  - SHA256: `52f4809468bcd1ddf8620d3f93f629418ea1ac732a7a6651c2262ae0458e926f`
  - Role: removes two test citations absent from the official retrieval-corpus vocabulary; v7 private audit top mean/robustness challenger.
- `submissions/private_final_corpus_clean_20260523/test_submission_private_vote_winners_t24_corpusclean.csv`
  - Public score: `unsubmitted`
  - SHA256: `e8790048ce4a706353797ddbac5c1d57da022fb3d754ee9dbe83b97e346b9a16`
  - Role: removes two test citations absent from the official retrieval-corpus vocabulary; v7 strict-tail challenger with `intersect_bold7h_33028`.
- `submissions/private_final_corpus_clean_20260523/test_submission_widebankG_hailmary_30702_corpusclean.csv`
  - Public score: `unsubmitted`
  - SHA256: `059cbdd5a7e25ce4400445bf115aaba216eb1f3d24763d7fa7d5264a210f581e`
  - Role: corpus-cleaned no-new-submission hedge family member.
- `submissions/private_final_corpus_clean_20260523/test_submission_fusion_samesrc03_32274_corpusclean.csv`
  - Public score: `unsubmitted`
  - SHA256: `f3ebe4734eba752f9a77edf304c53e3fdc70e708e8273e823f5d85862c6287ac`
  - Role: corpus-cleaned old fusion hedge.
- `submissions/private_final_corpus_clean_20260523/val_pred_private_blend_widebankG_winners_k18_a50_corpusclean.csv`
- `submissions/private_final_corpus_clean_20260523/val_pred_private_vote_winners_t24_corpusclean.csv`
- `submissions/private_final_corpus_clean_20260523/val_pred_widebankG_hailmary_30702_corpusclean.csv`
- `submissions/private_final_corpus_clean_20260523/val_pred_fusion_samesrc03_32274_corpusclean.csv`
- `submissions/private_final_corpus_clean_20260523/manifest.json`
- `artifacts/private_final_recheck_20260522/universe_v7_corpusclean224/`
- `artifacts/private_final_recheck_20260522/combined_private_sweep_v7_corpusclean224_fullpairs/`
- `artifacts/private_final_recheck_20260522/decision_audit_v7_corpusclean224/`

Final-selection pool added 2026-05-10:

- `submissions/test_submission_micro_swap_art75_012_040_from_32904.csv`
  - Public score: `0.32904`
  - SHA256: `11bdde695792ff7e43eb9fb991ac5290989ab3d06d9b1d51d75d50d3a077e0ff`
- `submissions/test_submission_sixhour_cross_high_val_45649.csv`
  - Public score: `0.32904`
  - SHA256: `0d755d4e5b5c48d10f8d9aa226da3483da64369df31a2da690d33e06c83dbdda`
- `submissions/aggressive_same_source_32904/samesrc32904_fixed12_plus_lfc1_test.csv`
  - Public score: `0.32562`
  - SHA256: `6c33b45a4a58aefbac6ece3203b41033bbdf3edcbd48ef20091d7bc36f700c5a`
- `submissions/rubik_seventeenth_order_bridge_leader_strict_commutator_test.csv`
  - Public score: `0.32097`
  - SHA256: `c23b6b33f3644acae3dd00c599862a1398844400e19af5716a0d128200123563`
- `submissions/val_pred_micro_swap_art75_012_040_from_32904.csv`
- `submissions/val_pred_sixhour_cross_high_val_45649.csv`
- `submissions/aggressive_same_source_32904/samesrc32904_fixed12_plus_lfc1_val.csv`
- `submissions/rubik_seventeenth_order_bridge_leader_strict_commutator_val.csv`
- `scripts/final_submission_lock.py`
- `scripts/rubik_seventeenth_order_search.py`
- `artifacts/aggressive_same_source_32904_summary.json`
- `artifacts/rubik_seventeenth_order/search_report.md`
- `artifacts/rubik_seventeenth_order/search_report.json`
- `artifacts/private_split_portfolio/sober_pair_consensus_top12.tsv`
- `artifacts/private_split_portfolio/private_risk_sober_top_candidates.txt`
- `artifacts/private_split_portfolio/final_adversarial_exact_sober_20260510/`
- `artifacts/private_split_portfolio/hour_sweep_sober_20260510/`
- `artifacts/private_split_portfolio/hour_sweep_wide_20260510/`
- `FINAL_SUBMISSION_DECISION_2026-05-10.md`

Primary aggressive/public-best candidate:

- `submissions/test_submission_baseline_public_best_32107.csv`
- `submissions/val_pred_baseline_public_best_32107.csv`
- `submissions/test_submission_targeted_proc_delta_balanced_swap.csv`
- `submissions/val_pred_targeted_proc_delta_balanced_swap.csv`
- SHA256 test payload: `99abea389f781bdadcdc8b8063942a20cbc95a5d765715b56bf9285b17dee5d3`

Stable prior/high hedge candidate:

- `submissions/test_submission_baseline_public_best_30911.csv`
- `submissions/val_pred_baseline_public_best_30911.csv`
- `submissions/test_submission_llm_proc_nobgg.csv`
- `submissions/val_pred_llm_proc_nobgg.csv`
- SHA256 test payload: `a966ba26d59bbc6bad0187369811e4cd1edbe40864fbba8ad6c21c08e6851bdf`

Third public-score candidate:

- `submissions/test_submission_baseline_public_best_30681.csv`
- `submissions/val_pred_baseline_public_best_30681.csv`
- `submissions/test_submission_overnight_combo_a.csv`
- `submissions/val_pred_overnight_combo_a.csv`
- SHA256 test payload: `88239ed12ebbbe87d1931d41cde7832febded5aea9426354f24c6fb29128f607`

Conservative hedge candidate:

- `submissions/test_submission_baseline_public_best_30257.csv`
- `submissions/val_pred_baseline_public_best_30257.csv`
- `submissions/test_submission_v11_winner_localperturb_top1.csv`
- `submissions/val_pred_v11_winner_localperturb_top1.csv`
- SHA256 test payload: `7c6424f39121ba018d322de55f939cc2050eb035ee1cdaac3205a190cfcfb4a6`

## Reproduction Code To Preserve

### Final-selection reproducibility package (added 2026-05-19)

- `LICENSE` (Apache-2.0)
- `SOLUTION_WRITEUP.md` (methodology + hashes + dependencies + no-hand-labeling statement; includes the 2026-05-20 offline prize path)
- `notebooks/swiss_finalists_repro.py` (AUDIT path — SHA-256-verified byte-identical reproduction of the locked finalist pool; NOT the prize-qualification notebook)
- `notebooks/swiss_finalists_repro.ipynb` (legacy Kaggle notebook wrapper for the audit path; regenerate from the `.py` source before using for newly added finalists)
- `scripts/package_finalists_for_kaggle.py` (builds a Kaggle-dataset-ready directory with renamed CSVs for the audit path)
- `artifacts/kaggle_dataset_swiss_legal_finalists_20260519/` (output of the packaging script — supports the audit notebook)
- `CODEX_OFFLINE_NOTEBOOK_HANDOVER_2026-05-19.md` (handover for the prize-qualification offline notebook, which must generalize to unseen queries per host re-evaluation right)

### Prize-qualification offline retriever (added 2026-05-20)

- `notebooks/swiss_prize_offline_retriever.py` (hybrid offline prize path; SHA-verified official-test finalist reproduction plus hidden-query dynamic fallback; no API calls)
- `scripts/package_prize_offline_for_kaggle.py` (stages the offline asset dataset and private Kaggle notebook wrapper)
- `scripts/prepare_prize_dense_assets.py` (downloads local public E5/reranker assets and builds/resumes dense caches for the offline notebook)
- `scripts/train_offline_selector.py` (trains plain-JSON local selector from public train/val candidate feature dumps; no API calls)
- `rust/v11_selector/src/bin/offline_dense_search.rs` (offline batched Rust dense-search accelerator; no internet/API path)
- `bin/offline_dense_search-linux-x86_64` (static Kaggle/Linux build of the Rust dense-search accelerator)
- `OFFLINE_PRIZE_NOTEBOOK_STATUS_2026-05-20.md` (architecture, validation, packaging status)
- `KAGGLE_GPU_DENSE_BUILDER_2026-05-20.md` (Kaggle GPU builder instructions for law/chunk dense assets)
- `models/intfloat-multilingual-e5-large/` (local offline E5 encoder used by the dense hidden-query path)
- `precompute/compact_court_dense_e5.npz` (25,621 compact public-court E5 vectors for hidden-query court retrieval)
- `precompute/compact_court_dense_e5_embeddings.npy` and `precompute/compact_court_dense_e5_citations.json` (Rust-friendly compact-court dense matrix)
- `precompute/law_dense_e5_embeddings.npy` and `precompute/law_dense_e5_citations.json` once built (Rust/NumPy no-FAISS law dense fallback)
- `precompute/law_chunk_dense_e5_embeddings.npy` and `precompute/law_chunk_dense_e5_citations.json` once built (Rust long-law chunk dense fallback)
- `precompute/offline_selector.json` once trained (plain-JSON local selector for hidden-query dynamic fallback)
- `index/faiss_laws.index` and `index/faiss_laws_citations.pkl` (staged optional law dense channel; already preserved as core indices)
- `artifacts/kaggle_dataset_swiss_legal_prize_offline_20260520/` (staged Kaggle dataset assets for the offline retriever)
- `artifacts/kaggle_kernel_swiss_prize_offline_20260520/` (staged private Kaggle notebook wrapper)

### Existing reproduction code (preserved)

- `notebooks/swiss_submission_v12.py`
- `notebooks/swiss_submission.py`
- `notebooks/swiss_submission.ipynb`
- `scripts/package_v12_for_kaggle.py`
- `scripts/targeted_procedural_deltas.py`
- `pipeline_v11.py`
- `run_v11_staged.py`
- `run_v11_meta_selector.py`
- `run_v11_train_selector.py`
- `promotion_gate.py`
- `submission_scorecard.py`
- `scripts/multi_signal_scorecard.py`
- `scripts/winner_localperturb_search.py` (deterministic perturbation search that produced the public-peak finalist)

Verified local reproduction after cleanup:

```bash
SUBMISSION_MODE=v12_repro_32107 python3 notebooks/swiss_submission_v12.py
shasum -a 256 notebooks/_local_output/submission.csv submissions/test_submission_baseline_public_best_32107.csv

SUBMISSION_MODE=v12_repro_30911 python3 notebooks/swiss_submission_v12.py
shasum -a 256 notebooks/_local_output/submission.csv submissions/test_submission_baseline_public_best_30911.csv
```

Both modes produced byte-identical outputs on 2026-04-28.

## Data And Cached Assets To Preserve

Competition data:

- `data/train.csv`
- `data/val.csv`
- `data/test.csv`
- `data/sample_submission.csv`
- `data/laws_de.csv`
- `data/court_considerations.csv`

Core local indices/corpora:

- `index/bm25_laws.pkl`
- `index/faiss_laws.index`
- `index/faiss_laws_citations.pkl`
- `index/court_citations.pkl`

Precompute/cache directory:

- Preserve `precompute/` as a whole unless a file is proven unrelated to prize
  reproduction.
- Especially preserve `precompute/llm_procedural_cache.json`.

Submissions/evaluation history:

- Preserve `submissions/` candidate CSVs listed above.
- Preserve `submissions/SCORE_TRACKER.md`.

Documentation/memory:

- `AGENTS.md`
- `CLAUDE.md`
- `CODEX_MEMORY.md`
- `HANDOFF.md`
- `HANDOVER_2026-04-27.md`
- `PRIZE_REPRO_DO_NOT_DELETE.md`

## Assets Already Deleted

The cleanup removed experimental/full-v12 assets such as local model checkpoints,
external staged corpora, and BGE full-corpus FAISS indexes. Those are **not**
needed for the currently verified `v12_repro_32107` and `v12_repro_30911`
paths, but would need to be recreated or redownloaded before resuming those
experimental directions.
