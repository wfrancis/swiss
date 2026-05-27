# 8h Private AI Search Run - 2026-05-11

Run root:

`artifacts/eight_hour_private_ai_20260511T043652Z`

Status:

- Started with `RUN_ID=20260511T043652Z EIGHT_HOUR_SEARCH_HOURS=8 scripts/run_8h_private_ai_search.sh`
- Finished cleanly at `2026-05-11T12:38:07Z`
- Generated 118 private-split `pair_report.tsv` files
- Final consensus: `artifacts/eight_hour_private_ai_20260511T043652Z/final_pair_consensus.tsv`
- No Kaggle submission was made during this run

## Main Result

The strongest final-submission pair by private-split survival was:

1. `fusion_rubik_base_delta_vote_fusion_08`
   - Val: `artifacts/eight_hour_private_ai_20260511T043652Z/fusion/rubik_base_delta_vote_fusion_08_val.csv`
   - Test: `artifacts/eight_hour_private_ai_20260511T043652Z/fusion/rubik_base_delta_vote_fusion_08_test.csv`

2. `samesrc_bold_micro_repair_01_v54176_j98111_a6_r10`
   - Val: `artifacts/eight_hour_private_ai_20260511T043652Z/micro/samesrc_bold/micro_repair_01_v54176_j98111_a6_r10_val.csv`
   - Test: `artifacts/eight_hour_private_ai_20260511T043652Z/micro/samesrc_bold/micro_repair_01_v54176_j98111_a6_r10_test.csv`

Final consensus metrics for the pair:

- Support: `108/118` reports
- Support fraction: `0.915254`
- Average rank: `2.12037`
- Average diversity-adjusted score: `0.545164`
- Average best-private simulator score: `0.555172`
- Average private std: `0.027022`
- Average test Jaccard between pair legs: `0.897550`

## Hedge Signal

The repeated adversarial leave-domain and leave-procedure splits favored:

`samesrc_bold_micro_repair_13_v53997_j98347_a6_r8`

- Val: `artifacts/eight_hour_private_ai_20260511T043652Z/micro/samesrc_bold/micro_repair_13_v53997_j98347_a6_r8_val.csv`
- Test: `artifacts/eight_hour_private_ai_20260511T043652Z/micro/samesrc_bold/micro_repair_13_v53997_j98347_a6_r8_test.csv`

This is a hedge candidate for domain/procedure private-shift risk, but the full portfolio consensus still ranked the `fusion_08 + bold_01` pair higher.

## Integrity Checks

Checked test CSV shape after the run:

- `fusion_rubik_base_delta_vote_fusion_08_test.csv`: 41 lines, zero empty prediction rows
- `micro_repair_01_v54176_j98111_a6_r10_test.csv`: 41 lines, zero empty prediction rows
- `micro_repair_13_v53997_j98347_a6_r8_test.csv`: 41 lines, zero empty prediction rows
- `data/test.csv`: 41 lines

## Interpretation

This run should be treated as private-split robustness evidence, not as a public leaderboard gradient. The result does not prove a Kaggle score. It does give a much stronger basis for final-submission selection than reacting to a public score, because the pair survived random, domain, procedure, length, gold-size, and leaveout split simulations over many repeated rounds.
