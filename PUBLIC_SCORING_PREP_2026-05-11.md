# Public Scoring Prep - 2026-05-11

Prepared while the 3-hour elite private-stress run was still active.

Run root:

- `artifacts/elite_3h_private_stress_20260511T125331Z`

Interim private-stress consensus at prep time:

- Completed stress reports: 580
- Top pair support: 580/580
- Top pair:
  - `fusion_rubik_base_delta_vote_fusion_08`
  - `samesrc_bold_micro_repair_01_v54176_j98111_a6_r10`

Prepared CSVs:

1. Upside public-scoring candidate:
   - `submissions/prepared_public_20260511/test_submission_elite3h_samesrc_bold_01.csv`
   - SHA256: `b5003a7aef38710385a734b9991844e75e3a3eb5e3f9372139427c6890728c43`
   - Source: `artifacts/eight_hour_private_ai_20260511T043652Z/micro/samesrc_bold/micro_repair_01_v54176_j98111_a6_r10_test.csv`
   - Rust diverse eval: STRONG PROMOTE, val +13.2pp vs 0.32107 baseline, test Jaccard 0.895, balanced churn 22 adds / 76 removes.

2. Conservative stress-anchor candidate:
   - `submissions/prepared_public_20260511/test_submission_elite3h_fusion_rubik_v08.csv`
   - SHA256: `64b0581cc149d0a48a94d6300291d7444262134d631659b3aa4e55e54ebc29f8`
   - Source: `artifacts/eight_hour_private_ai_20260511T043652Z/fusion/rubik_base_delta_vote_fusion_08_test.csv`
   - Rust diverse eval: STRONG PROMOTE, val +13.0pp vs 0.32107 baseline, test Jaccard 0.953, balanced churn 10 adds / 34 removes.

3. Alternate hedge:
   - `submissions/prepared_public_20260511/test_submission_elite3h_samesrc_bold_13_alt.csv`
   - SHA256: `61cd4981b9323ee18e52b115a382d5eb32b9a08cf40f192fa11ab32cd48dc8ae`
   - Source: `artifacts/eight_hour_private_ai_20260511T043652Z/micro/samesrc_bold/micro_repair_13_v53997_j98347_a6_r8_test.csv`
   - Rust diverse eval: STRONG PROMOTE, val +13.0pp vs 0.32107 baseline, test Jaccard 0.897, balanced churn 22 adds / 74 removes.

Validation:

- All three CSVs have 40 prediction rows plus header.
- Query IDs match `data/test.csv` exactly.
- Header is `query_id,predicted_citations`.
- Empty prediction rows: 0.

Recommendation for one public-scoring submission:

- Submit `test_submission_elite3h_samesrc_bold_01.csv` if the goal is maximum upside.
- Submit `test_submission_elite3h_fusion_rubik_v08.csv` if the goal is lower-churn private robustness.

No Kaggle submission was made by this prep step.

Submitted after prep:

- Submitted file: `submissions/prepared_public_20260511/test_submission_elite3h_samesrc_bold_01.csv`
- Kaggle description: `elite3h samesrc bold 01 stress consensus`
- Submitted at: `2026-05-11 15:10:21 UTC`
- Public score: `0.32921`

Completed 3-hour stress result:

- Final run status: complete
- Final stress reports: `993`
- Final top pair support: `993/993`
- Final top pair:
  - `fusion_rubik_base_delta_vote_fusion_08`
  - `samesrc_bold_micro_repair_01_v54176_j98111_a6_r10`
- Final top-pair avg diversity-adjusted score: `0.5446656157`
- Final top-pair avg best-private simulator score: `0.5549047774`
- Final top-pair avg private std: `0.0275831544`
- Final top-pair test Jaccard: `0.8975501114`

Second submitted candidate:

- Submitted file: `submissions/prepared_public_20260511/test_submission_elite3h_fusion_rubik_v08.csv`
- Kaggle description: `elite3h fusion rubik v08 conservative stress anchor`
- Submitted at: `2026-05-11 23:14:41 UTC`
- Public score: `0.32551`
- Interpretation: lower than the `0.32921` upside leg on public LB, but still the lower-churn conservative anchor from the completed stress run.
