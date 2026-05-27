# Public Board Trap Double Check - 2026-05-22

## Staff/PhD Conclusion

The public leaderboard is now a dangerous optimization target. The correct objective is not the highest public score; it is the best final-submission portfolio for the unseen private half.

Current private-first recommendation remains unchanged:

1. `intersect_bold7h_33028`
2. `final_hedge_fusion_samesrc03_32274`
3. If the Kaggle UI truly counts three finals: `public_peak_33438`

Do not replace the private pair with the new `sp09_33681` public chain.

## Why The Public Board Is A Trap

The public leaderboard scores only 20 of the 40 official test rows. A public gain is therefore evidence about those public rows only.

If a single-row citation removal changes the public score, that edit is almost certainly on the public half. Chaining those discoveries improves the visible public score but does not establish private improvement.

The top public gap is small in absolute row-level terms:

| Team | Public score | Public gap vs our `0.33681` |
|---|---:|---:|
| Kanak Raj | `0.35940` | `0.02259` |
| HyperX | `0.35324` | `0.01643` |
| thechint | `0.35198` | `0.01517` |
| my_LAW | `0.34347` | `0.00666` |
| TEERAWAT | `0.33704` | `0.00023` |

Because public LB has only 20 rows, a `0.020` public gap is about `0.4` total F1-points spread across those 20 public rows. That can be a handful of citation precision/recall fixes, not proof of a stronger private system.

## New `sp09_33681` Diagnosis

`sp09_33681` is nearly identical to `public_peak_33438`:

| File | Total citations | Jaccard vs `public_peak_33438` |
|---|---:|---:|
| `public_peak_33438` | `823` | `1.0000` |
| `sp09_33681` | `814` | `0.9891` |

`sp09_33681` adds no citations and only removes these 9:

- `test_011`: `Art. 19 Abs. 1 IPRG`
- `test_011`: `BGE 140 III 134 E. 3.1`
- `test_013`: `Art. 106 Abs. 1 BGG`
- `test_013`: `Art. 48b Abs. 1 AVV`
- `test_017`: `Art. 320 ZPO`
- `test_027`: `Art. 285 Abs. 1 ZGB`
- `test_037`: `Art. 100 Abs. 1 BGG`
- `test_037`: `Art. 951 OR`
- `test_040`: `BGE 138 III 289 E. 11.1.1`

This is public precision trimming, not a new independent private solution. Its private-half behavior is likely the same as `public_peak_33438`, except for any accidental private-row damage.

## Six-Scheme Private Audit

Source: `artifacts/final_selection_20260519/private_split_final_audit/`

| Pair | Avg p_contains_private_winner | Min p | Avg rank | Avg Jaccard |
|---|---:|---:|---:|---:|
| `intersect_bold7h_33028` + `fusion_samesrc03_32274` | `0.75517` | `0.70924` | `1.33` | `0.8885` |
| `public_peak_33438` + `fusion_samesrc03_32274` | `0.44975` | `0.36103` | `3.00` | `0.9232` |
| `public_peak_33438` + `intersect_bold7h_33028` | `0.30542` | `0.17889` | `9.50` | `0.9466` |
| `public_peak_33438` + `pre_tomo_33186` | `0.23548` | `0.11704` | `12.83` | `0.9916` |

The private pair wins because it keeps both:

- a high-ceiling private leg: `intersect_bold7h_33028`
- a lower-variance, different-failure-mode hedge: `fusion_samesrc03_32274`

The public legs are too correlated with each other.

## Decision Rule From Here

Public submissions may still be useful as probes, but they should not automatically become final submissions.

Promotion into final selection now requires one of these:

1. A genuinely new private hedge with materially lower Jaccard against the selected pair.
2. A candidate built from non-public-LB-derived reasoning, with a reproducible notebook path.
3. Evidence from private-split simulation that it improves `p_contains_private_winner` against `intersect + fusion`.

Higher public score alone is insufficient.

## Operational Recommendation

Keep the final portfolio private-first:

- If Kaggle enforces 2 finals: `intersect_bold7h_33028` + `fusion_samesrc03_32274`
- If Kaggle honors 3 finals: `intersect_bold7h_33028` + `fusion_samesrc03_32274` + `public_peak_33438`

Treat `sp09_33681` and later public chains as leaderboard probes unless a separate private audit proves otherwise.
