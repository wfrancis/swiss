# Private Max-Effort Blend Breakthrough

Date: 2026-05-22 local / 2026-05-23 UTC

This audit extends the private-final referee beyond the original 90-candidate
universe. It targets simulated half-validation/private-split robustness rather
than the public leaderboard. It does not use hidden/private labels.

## 2026-05-22 Swarm Re-Audit Correction

The new candidates are real challengers, but the evidence is not one-sided.

- `private_blend_widebankG_winners_k18_a50` has the best exact mean and the
  best leave-one-query-out recipe stability. It is the practical mean/LOO
  favorite.
- `private_vote_winners_t24` has slightly better exact p10, lower Jaccard when
  paired with `intersect_bold7h_33028`, and is the only generated challenger
  that passes the strict pair-decision audit against the current
  `intersect + widebankG_hailmary` baseline.
- Both generated challengers are additive relative to `intersect_bold7h_33028`
  (`blend`: +96/-0 atoms; `vote`: +108/-0 atoms), so neither should be used
  without the `intersect` anchor and neither should be paired with the other.
- `vote` is almost a superset of `blend` (`J=0.9865`, +12/-0 atoms), so it is
  not an independent hedge with `blend`.

## 2026-05-23 Corpus-Clean Re-Audit

The final cold pass found a small but important corpus-vocabulary issue:

- `private_blend_widebankG_winners_k18_a50`, `private_vote_winners_t24`, and
  `fusion_samesrc03_32274` each contained two test citations absent from the
  official local retrieval-corpus vocabulary:
  `BGE 146 V 51`, `BGE 146 V 51 E. 8.2`.
- `widebankG_hailmary_30702` contained four absent test citations:
  `Art. 268 ZPO` on `test_001`, `test_008`, `test_035`, and `test_037`.

Because the competition states that only exact citation strings from the
retrieval corpus count, these atoms are expected false positives. Corpus-clean
variants remove only those non-corpus atoms and keep all other predictions
unchanged.

New stable corpus-clean candidates:

| Candidate | SHA256 | Removed test atoms |
|---|---|---:|
| `private_blend_widebankG_winners_k18_a50_corpusclean` | `52f4809468bcd1ddf8620d3f93f629418ea1ac732a7a6651c2262ae0458e926f` | `2` |
| `private_vote_winners_t24_corpusclean` | `e8790048ce4a706353797ddbac5c1d57da022fb3d754ee9dbe83b97e346b9a16` | `2` |
| `widebankG_hailmary_30702_corpusclean` | `059cbdd5a7e25ce4400445bf115aaba216eb1f3d24763d7fa7d5264a210f581e` | `4` |
| `fusion_samesrc03_32274_corpusclean` | `f3ebe4734eba752f9a77edf304c53e3fdc70e708e8273e823f5d85862c6287ac` | `2` |

The v7 all-pair Rust sweep added these four candidates to the 220-candidate
universe and reran all 40 split replicas with 200,000 splits each.

Strict v7 audit highlights:

| Pair | Verdict | Avg p_contains | Min p_contains | Avg rank | Exact p10 | Exact mean | Test Jaccard |
|---|---|---:|---:|---:|---:|---:|---:|
| `intersect + widebankG` | `BASELINE_KEEP` | `0.108389` | `0.000000` | `8482.350` | `0.517802` | `0.558186` | `0.847755` |
| `blend_clean + widebank_clean` | `REPLACE_REVIEW` | `0.352495` | `0.159460` | `210.050` | `0.536037` | `0.575450` | `0.872497` |
| `blend_clean + widebank` | `REPLACE_REVIEW` | `0.352308` | `0.159460` | `215.675` | `0.536037` | `0.575450` | `0.868835` |
| `intersect + vote_clean` | `REPLACE_REVIEW` | `0.315822` | `0.153915` | `77.625` | `0.541159` | `0.575445` | `0.880496` |
| `intersect + blend_clean` | `HOLD_BELOW_BASELINE` | `0.460698` | `0.215235` | `30.700` | `0.538085` | `0.577590` | `0.892571` |

Interpretation:

- `intersect + vote_clean` is the best strict-tail pair among the human-sized
  final recommendations: highest exact p10 among the clean replacement pairs
  and full strict-audit pass.
- `blend_clean + widebank_clean` is the formal v7 top replacement-review pair
  by audit order and keeps more distance from `intersect`, but its exact p10 is
  lower than `intersect + vote_clean`.
- `intersect + blend_clean` has the best exact mean in this small comparison,
  but misses the strict diversity gate by `0.004817` Jaccard over tolerance.

## What Changed

1. Expanded the historical universe from 90 capped candidates to 160, then 190
   with forced old low-public private hedges.
2. Added all historical staff fifteenth-order Markdown manifests to the
   universe builder.
3. Added deterministic private-blend search over existing val/test companions.
4. Re-ran full all-pair Rust private sweeps:
   - 10 split schemes
   - 4 seeds
   - 200,000 splits per run
   - all pair reports, not top-pair truncation
5. Added leave-one-query-out stability checks for the blend family.
6. Promoted the most practical blend challenger into a stable submission path
   and exact-replay/prize-replay mode.

## Stable No-New-Submission Pair

If we do not want to introduce an unsubmitted generated blend, the strongest
private pair remains:

- `intersect_bold7h_33028`
- `widebankG_hailmary_30702`

Against the old `intersect + fusion` baseline in the 160-candidate sweep:

| Pair | Avg p_contains | Min p_contains | Avg rank | Exact p10 | Exact mean | Test Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| `intersect + fusion` | `0.301479` | `0.118155` | `96.575` | `0.516823` | `0.556049` | `0.888510` |
| `intersect + widebankG_hailmary` | `0.521580` | `0.253630` | `2.800` | `0.517802` | `0.558186` | `0.847755` |

Verdict: `widebankG_hailmary_30702` still cleanly replaces
`fusion_samesrc03_32274` if we restrict ourselves to already-submitted,
old-family candidates.

## New Higher-Upside Candidate

Best practical generated second leg:

- `private_blend_widebankG_winners_k18_a50`
- Stable test file:
  `submissions/private_final_blend_20260522/test_submission_private_blend_widebankG_winners_k18_a50.csv`
- Stable val companion:
  `submissions/private_final_blend_20260522/val_pred_private_blend_widebankG_winners_k18_a50.csv`
- SHA256:
  `1164bb097cda46ffee43324fcbef498f8daf36edf4bb681cbdeaf88548caccea`

Recipe:

`base=widebankG_hailmary_30702`, pool=`winners`, weights=`uniform`,
keep threshold=`0.18`, add threshold=`0.50`.

This keeps the widebankG private-diverse spine, removes weak cites unless they
have enough winner-pool support, and adds only high-consensus winner-pool cites.

Strict-tail generated challenger:

- `private_vote_winners_t24`
- Stable test file:
  `submissions/private_final_blend_20260522/test_submission_private_vote_winners_t24.csv`
- Stable val companion:
  `submissions/private_final_blend_20260522/val_pred_private_vote_winners_t24.csv`
- SHA256:
  `26d371cc759f1491b0e14c7d892d5baea7e100af68e720420e17c179caf85b65`

When paired with `intersect_bold7h_33028`, this has slightly better exact p10,
lower pair Jaccard, and passes the strict decision audit. It has weaker
mean/regret and weaker LOO stability than the practical blend. Treat it as the
strict-tail challenger, not as a co-final with the blend.

## Why This Is The Practical Breakthrough

When paired with `intersect_bold7h_33028`:

| Pair | Exact p10 | Exact mean | Exact regret | Test Jaccard |
|---|---:|---:|---:|---:|
| `intersect + fusion` | `0.516823` | `0.556049` | `0.022239` | `0.888510` |
| `intersect + widebankG_hailmary` | `0.517802` | `0.558186` | `0.020102` | `0.847755` |
| `intersect + private_blend_widebankG_winners_k18_a50` | `0.536770` | `0.574709` | `0.003464` | `0.890536` |
| `intersect + private_vote_winners_t24` | `0.539120` | `0.572799` | `0.005488` | `0.878515` |

The raw two-blend pairs sometimes score even higher on exact val, but many are
too same-shaped or too selection-optimized. A generated challenger plus
`intersect` is the only acceptable shape. The disagreement is which generated
challenger best fits the second leg:

- one proven anchor leg (`intersect_bold7h_33028`)
- `blend`: strongest exact mean and LOO stability
- `vote`: strongest strict-tail/diversity audit

## LOO Stability

The blend family was rechecked with a leave-one-query-out recipe selector:

- selected `private_blend_widebankG_winners_k18_a50` on 8/10 held-out folds
- selected `vote_top16_diverse_t50` on 1/10
- selected `vote_nonclone_private_t60` on 1/10

Held-out mean pair F1:

| Pair / Selector | Held-out mean |
|---|---:|
| LOO-selected blend family | `0.587388` |
| `intersect + widebankG_hailmary` | `0.585945` |
| `intersect + fusion` | `0.581937` |

This shrinks the apparent exact-val lift, as expected, but it does not erase it.
That is the main reason to prefer the practical single-blend candidate over a
more aggressive two-blend pair.

## Recommendation

For maximum simulated-private-split robustness:

1. Prefer corpus-clean variants over their raw counterparts. The removals are
   limited to citations absent from the official local retrieval-corpus
   vocabulary.
2. If submitting only one new generated private challenger, submit
   `private_vote_winners_t24_corpusclean` first for the strict-tail objective.
3. If there is room for a second exploratory submission, submit
   `private_blend_widebankG_winners_k18_a50_corpusclean` as the mean/robustness
   challenger.
4. Do not use the public score to choose between them after submission; the
   submit is only a format/sanity check. Precommit the objective before looking
   at Kaggle feedback.
5. If prioritizing strict pair-audit/private-tail behavior, final pair should
   be:
   - `intersect_bold7h_33028`
   - `private_vote_winners_t24_corpusclean`
6. If prioritizing the formal v7 replacement-review ordering, final pair should
   be:
   - `private_blend_widebankG_winners_k18_a50_corpusclean`
   - `widebankG_hailmary_30702_corpusclean`
7. If prioritizing exact mean and accepting the slight diversity-gate miss,
   final pair should be:
   - `intersect_bold7h_33028`
   - `private_blend_widebankG_winners_k18_a50_corpusclean`
8. If we decide not to use an unsubmitted generated candidate, final pair should
   be:
   - `intersect_bold7h_33028`
   - `widebankG_hailmary_30702_corpusclean` if submitted, otherwise
     `widebankG_hailmary_30702`
9. If Kaggle truly honors 3 finals despite written rules, add the remaining
   clean leg after the chosen two.

Do not replace the private leg with the recent public 0.336+ chain unless the
objective is public rank optics. Those candidates are public-clone-shaped and do
not solve the private split problem.

## Verification

Passed locally:

```bash
python3 -m py_compile \
  scripts/build_private_final_universe.py \
  scripts/search_private_blend_candidates.py \
  scripts/private_final_portfolio_decision.py \
  scripts/package_finalists_for_kaggle.py \
  scripts/package_prize_offline_for_kaggle.py \
  notebooks/swiss_finalists_repro.py \
  notebooks/swiss_prize_offline_retriever.py

SUBMISSION_MODE=private_blend_widebankG_winners_k18_a50 \
  python3 notebooks/swiss_finalists_repro.py

SUBMISSION_MODE=private_blend_widebankG_winners_k18_a50 \
  python3 notebooks/swiss_prize_offline_retriever.py

SUBMISSION_MODE=private_vote_winners_t24_corpusclean \
  python3 notebooks/swiss_finalists_repro.py

SUBMISSION_MODE=private_blend_widebankG_winners_k18_a50_corpusclean \
  python3 notebooks/swiss_prize_offline_retriever.py

python3 scripts/package_finalists_for_kaggle.py \
  --out artifacts/kaggle_dataset_swiss_legal_finalists_20260519_v6_corpusclean
```

Key artifacts:

- `artifacts/private_final_recheck_20260522/universe_v5_blends220/`
- `artifacts/private_final_recheck_20260522/combined_private_sweep_v5_blends220_fullpairs/`
- `artifacts/private_final_recheck_20260522/decision_audit_v5_blends220_vs_current_hailmary/`
- `artifacts/private_final_recheck_20260522/universe_v7_corpusclean224/`
- `artifacts/private_final_recheck_20260522/combined_private_sweep_v7_corpusclean224_fullpairs/`
- `artifacts/private_final_recheck_20260522/decision_audit_v7_corpusclean224/`
- `artifacts/private_final_recheck_20260522/private_blend_search_v1/`
- `artifacts/kaggle_dataset_swiss_legal_finalists_20260519_v6_corpusclean/`
