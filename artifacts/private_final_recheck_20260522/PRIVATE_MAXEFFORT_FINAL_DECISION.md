# Private Final Max-Effort Decision

Date: 2026-05-22

## Decision

For **private-score maximization**, the new strict recommendation is:

1. `intersect_bold7h_33028`
   - File: `submissions/staff3_pairing_20260513/test_submission_private_rethink_intersect_bold7h_j955.csv`
   - Kaggle ref: `52819486`
   - Public: `0.33028`
   - SHA256: `542b40471aec01c53a891cccab44e8b3495cf6bb06e6c1a521fe88da61afd8ca`

2. `widebankG_hailmary_30702`
   - File: `submissions/test_submission_bold_7h_widebankG_hailmary.csv`
   - Kaggle ref: `52084244`
   - Public: `0.30702`
   - SHA256: `bd603ad9497e7f1a32b4c18067454ce0b869aa09728ebc7ff8d68af06e54fa6c`

This replaces `fusion_samesrc03_32274` as the second private leg if we are optimizing the private leaderboard rather than public leaderboard optics.

## Why This Changed

The earlier May 19 final referee only considered the final-shortlist universe. I expanded the candidate universe to include older staff portfolio Markdown manifests, which surfaced an already-submitted public-low/private-diverse candidate: `test_submission_bold_7h_widebankG_hailmary.csv`.

I also fixed a decision-methodology problem: the first combined sweep wrote only top pairs per run, which over-favored pairs appearing in only a few schemes. The final sweep wrote **all 4,005 pairs per run**, so every pair is evaluated across the same 40 runs.

## Evidence

Fresh universe:

- Full deduped candidates: `239`
- Capped Rust candidates: `90`
- Rust private sweep: `10` schemes x `4` seeds x `200,000` splits
- Pair rows written per run: all pairs, not top-only
- Output: `artifacts/private_final_recheck_20260522/decision_audit_v2_90_fullpairs_strict/decision.md`

Strict comparison versus current baseline `intersect_bold7h_33028 + fusion_samesrc03_32274`:

| Pair | Avg p_contains | Avg rank | Delta mean private | Delta exact p10 | Delta exact regret | Test Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| Current: `intersect + fusion` | `0.301479` | `85.250` | `0.000000` | `0.000000` | `0.000000` | `0.888510` |
| New: `intersect + widebankG_hailmary` | `0.521393` | `2.800` | `+0.001766` | `+0.000979` | `-0.002137` | `0.847755` |

The new pair is better on the metrics that matter for private survival:

- Higher pair private expected score.
- Higher lower-tail exact half-val score.
- Lower exact regret against the split winner.
- Much lower test Jaccard, so the two final slots fail in more different ways.
- The replacement leg is already a Kaggle submission, so no new public submission is required.

## Caveat

`widebankG_hailmary_30702` is public-low (`0.30702`). That is not a disqualifier for a private hedge; it is exactly why it is useful if the public board is a trap. But it means selecting it will look emotionally wrong in the UI.

Prize-notebook caveat: `widebankG_hailmary_30702` was originally a file-upload submission. I added local exact-replay support for it in:

- `notebooks/swiss_finalists_repro.py`
- `notebooks/swiss_prize_offline_retriever.py`
- `scripts/package_finalists_for_kaggle.py`
- `scripts/package_prize_offline_for_kaggle.py`

It still needs a Kaggle Notebook submission / dataset update before we can call that leg prize-notebook-clean.

## If 3 Finals Are Truly Honored

Use:

1. `intersect_bold7h_33028`
2. `widebankG_hailmary_30702`
3. `fusion_samesrc03_32274`

Exact three-leg check beats the old `intersect + fusion + public_peak` private hedge:

| Triple | Exact p10 | Exact mean | Exact regret |
|---|---:|---:|---:|
| `intersect + widebankG_hailmary + fusion` | `0.518817` | `0.559272` | `0.002183` |
| `intersect + fusion + public_peak` | `0.518254` | `0.557252` | `0.004203` |

Written rules still say two final submissions, so this is only a UI-anomaly fallback.
