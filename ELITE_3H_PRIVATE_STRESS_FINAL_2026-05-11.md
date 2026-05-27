# Elite 3h Private Stress Final - 2026-05-11

Run root:

- `artifacts/elite_3h_private_stress_20260511T125331Z`

Run shape:

- Duration target: 3 hours
- Splits per sweep: 12,000,000
- Rayon threads: 10
- Final completed stress reports: 993
- Final status: clean finish at `2026-05-11T15:47:33Z`

Public scoring outcome:

- Submitted file: `submissions/prepared_public_20260511/test_submission_elite3h_samesrc_bold_01.csv`
- Kaggle public score: `0.32921`
- Prior public anchor: `0.32904`
- Public delta: `+0.00017`
- Second submitted file: `submissions/prepared_public_20260511/test_submission_elite3h_fusion_rubik_v08.csv`
- Second public score: `0.32551`
- Second-file role: conservative lower-churn stress anchor, not public-peak leg.

Final pair consensus:

| Rank | Left | Right | Support | Avg rank | Avg diversity-adjusted | Avg best-private | Avg private std | Test Jaccard |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `fusion_rubik_base_delta_vote_fusion_08` | `samesrc_bold_micro_repair_01_v54176_j98111_a6_r10` | 993/993 | 2.1641 | 0.5446656 | 0.5549048 | 0.0275832 | 0.8975501 |
| 2 | `fusion_rubik_base_delta_vote_fusion_08` | `samesrc_bold_micro_repair_07_v54064_j98347_a6_r8` | 993/993 | 4.8912 | 0.5440768 | 0.5543722 | 0.0272987 | 0.8997773 |
| 3 | `fusion_rubik_base_delta_vote_fusion_08` | `samesrc_bold_micro_repair_13_v53997_j98347_a6_r8` | 993/993 | 4.9003 | 0.5442160 | 0.5543529 | 0.0274915 | 0.8976641 |
| 4 | `fusion_rubik_base_delta_vote_fusion_08` | `samesrc_bold_micro_repair_06_v54085_j98229_a6_r9` | 993/993 | 5.0624 | 0.5440409 | 0.5546178 | 0.0275333 | 0.8965517 |
| 5 | `fusion_rubik_base_delta_vote_fusion_08` | `samesrc_bold_micro_repair_12_v54028_j98580_a4_r8` | 927/993 | 6.8091 | 0.5446841 | 0.5551598 | 0.0281905 | 0.9081747 |

Prepared files:

- Upside/submitted: `submissions/prepared_public_20260511/test_submission_elite3h_samesrc_bold_01.csv`
- Conservative anchor: `submissions/prepared_public_20260511/test_submission_elite3h_fusion_rubik_v08.csv`
- Alternate hedge: `submissions/prepared_public_20260511/test_submission_elite3h_samesrc_bold_13_alt.csv`

Validation:

- All prepared CSVs have 40 rows plus header.
- Query IDs match `data/test.csv`.
- Empty prediction rows: 0.

Final read:

- The 3-hour stress run strongly supports the submitted `samesrc_bold_01` as the upside public/private leg.
- For final two-submission selection, the natural pair remains:
  - submitted upside leg: `test_submission_elite3h_samesrc_bold_01.csv`
  - conservative anchor leg: `test_submission_elite3h_fusion_rubik_v08.csv`
- Avoid submitting more near-duplicate same-source repair variants unless there is a specific reason; the second, third, and fourth same-source variants are close but not better than `samesrc_bold_01` on final consensus.
