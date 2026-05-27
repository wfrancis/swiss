# Private Final Portfolio Decision Audit

## Decision

At least one challenger passed the strict replacement-review gate.

Current baseline pair remains:

- `intersect_bold7h_33028`
- `widebankG_hailmary_30702`

## Baseline Metrics

| Metric | Value |
|---|---:|
| `avg_p_contains` | `0.108389` |
| `min_p_contains` | `0.000000` |
| `avg_rank` | `8482.350` |
| `avg_mean_best_private` | `0.558446` |
| `delta_mean_best_private` | `0.000000` |
| `avg_std_best_private` | `0.028686` |
| `avg_test_jaccard` | `0.847755` |
| `delta_test_jaccard` | `0.000000` |
| `exact_p10` | `0.517802` |
| `delta_exact_p10` | `0.000000` |
| `exact_worst` | `0.480027` |
| `exact_mean` | `0.558186` |
| `exact_regret` | `0.021738` |
| `delta_exact_regret` | `0.000000` |

## Top Pair Decisions

| Verdict | Pair | Avg p_contains | Min p_contains | Avg rank | Jaccard | Exact p10 | Reasons |
|---|---|---:|---:|---:|---:|---:|---|
| BASELINE_KEEP | `intersect_bold7h_33028` + `widebankG_hailmary_30702` | `0.108389` | `0.000000` | `8482.350` | `0.847755` | `0.517802` | min_p,exact_p10,exact_regret,diversity |
| REPLACE_REVIEW | `private_blend_widebankG_winners_k18_a50_corpusclean` + `widebankG_hailmary_30702_corpusclean` | `0.352495` | `0.159460` | `210.050` | `0.872497` | `0.536037` | mean_best,avg_p,min_p,exact_p10,exact_regret,diversity |
| REPLACE_REVIEW | `private_blend_widebankG_winners_k18_a50_corpusclean` + `widebankG_hailmary_30702` | `0.352308` | `0.159460` | `215.675` | `0.868835` | `0.536037` | mean_best,avg_p,min_p,exact_p10,exact_regret,diversity |
| REPLACE_REVIEW | `intersect_bold7h_33028` + `private_vote_winners_t24_corpusclean` | `0.315822` | `0.153915` | `77.625` | `0.880496` | `0.541159` | mean_best,avg_p,min_p,exact_p10,exact_regret,diversity |
| REPLACE_REVIEW | `private_vote_winners_t24_corpusclean` + `widebankG_hailmary_30702_corpusclean` | `0.207619` | `0.112230` | `485.825` | `0.861602` | `0.537697` | mean_best,avg_p,min_p,exact_p10,exact_regret,diversity |
| REPLACE_REVIEW | `private_vote_winners_t24_corpusclean` + `widebankG_hailmary_30702` | `0.207432` | `0.112230` | `493.675` | `0.858031` | `0.537697` | mean_best,avg_p,min_p,exact_p10,exact_regret,diversity |
| REPLACE_REVIEW | `samesrc_safe_micro_repair_05_v53422_j99527_a4_r0` + `vote_top16_diverse_t50` | `0.169577` | `0.096430` | `1088.975` | `0.877940` | `0.535946` | mean_best,avg_p,min_p,exact_p10,exact_regret,diversity |
| REPLACE_REVIEW | `samesrc_bold_micro_repair_23_v53827_j98817_a4_r6` + `vote_top16_diverse_t50` | `0.167530` | `0.094605` | `1041.125` | `0.886005` | `0.535946` | mean_best,avg_p,min_p,exact_p10,exact_regret,diversity |
| REPLACE_REVIEW | `samesrc_bold_micro_repair_19_v53917_j98698_a4_r7` + `vote_top16_diverse_t50` | `0.165857` | `0.092615` | `979.450` | `0.887006` | `0.535946` | mean_best,avg_p,min_p,exact_p10,exact_regret,diversity |
| REPLACE_REVIEW | `samesrc_safe_micro_repair_01_v53520_j99292_a6_r0` + `vote_top16_diverse_t50` | `0.165857` | `0.092615` | `1118.600` | `0.880179` | `0.535946` | mean_best,avg_p,min_p,exact_p10,exact_regret,diversity |
| REPLACE_REVIEW | `fusion_samesrc_base_delta_vote_fusion_01` + `vote_top16_diverse_t50` | `0.165857` | `0.092615` | `1124.375` | `0.875000` | `0.536337` | mean_best,avg_p,min_p,exact_p10,exact_regret,diversity |
| REPLACE_REVIEW | `fusion_samesrc_base_delta_vote_fusion_02` + `vote_top16_diverse_t50` | `0.165857` | `0.092615` | `1235.800` | `0.877252` | `0.535946` | mean_best,avg_p,min_p,exact_p10,exact_regret,diversity |

## Candidate Warnings

| Candidate | Warning |
|---|---|
| `live_33545` | public clone J=0.9939 |
| `live_33545_plus005` | public clone J=0.9927 |
| `live_33545_plus011art19` | public clone J=0.9927 |
| `live_33545_plus013` | public clone J=0.9927 |
| `live_33545_plus030art285` | public clone J=0.9927 |
| `live_33545_plus030bge` | public clone J=0.9927 |
| `live_33545_plus040` | public clone J=0.9927 |
| `overlay_samesrc_03` | public clone J=0.9843 |
| `public_peak_33438_prepared` | public clone J=0.9916 |
| `samesrc_bold_01` | public clone J=0.9821 |
| `samesrc_bold_micro_repair_06_v54085_j98229_a6_r9` | public clone J=0.9821 |
| `samesrc_bold_micro_repair_07_v54064_j98347_a6_r8` | public clone J=0.9809 |
| `submitted_samesrc_bold_01` | public clone J=0.9833 |
| `submitted_samesrc_bold_13_alt` | public clone J=0.9809 |

## Hard-Rejected Pair Examples

- `private_blend_widebankG_winners_k18_a50_corpusclean` + `private_vote_winners_t24_corpusclean`: weak hedge J=0.9865
- `base_widebankG_hailmary_30702_samesrc_diverse_k10_a70` + `private_blend_widebankG_winners_k18_a50_corpusclean`: weak hedge J=0.9842
- `base_widebankG_hailmary_30702_winners_private_k24_a70` + `private_blend_widebankG_winners_k18_a50_corpusclean`: weak hedge J=0.9543
- `private_blend_widebankG_winners_k18_a50_corpusclean` + `vote_winners_private_t24`: weak hedge J=0.9843
- `base_widebankG_hailmary_30702_winners_uniform_k18_a42` + `private_blend_widebankG_winners_k18_a50_corpusclean`: weak hedge J=0.9876
- `cpu_capvote_balanced_a46_cap1_r00_rc0` + `private_blend_widebankG_winners_k18_a50_corpusclean`: weak hedge J=0.9559
- `cpu_capvote_balanced_a38_cap1_r00_rc0` + `private_blend_widebankG_winners_k18_a50_corpusclean`: weak hedge J=0.9560
- `cpu_capvote_balanced_a30_cap1_r00_rc0` + `private_blend_widebankG_winners_k18_a50_corpusclean`: weak hedge J=0.9528
