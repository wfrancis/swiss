# Private Final Recheck - 2026-05-22

## Conclusion

Private optimization still favors the May 19 private pair:

1. `intersect_bold7h_33028`
2. `final_hedge_fusion_samesrc03_32274`

If the Kaggle UI truly counts three final selections, keep `public_peak_33438` as the third public-upside leg. Do not replace either private leg with the new `sp09_33681` public-chain submission.

## Why

The new public best, `sp09_33681`, is a public-leaderboard chain, not a new private hedge. Compared with `public_peak_33438`, it:

- adds `0` citations
- removes `9` citations
- has test atom Jaccard `0.9891` versus `public_peak_33438`

So it is essentially the same public leg with a few public-half precision removals. It increases public rank, but it does not add meaningful private portfolio diversity.

## Six-Scheme Private Audit

Source: `artifacts/final_selection_20260519/private_split_final_audit/`

| Pair | Avg p_contains_private_winner | Min | Avg rank | Ranks by scheme |
|---|---:|---:|---:|---|
| `intersect_bold7h_33028` + `fusion_samesrc03_32274` | `0.75517` | `0.70924` | `1.33` | `1,2,2,1,1,1` |
| `public_peak_33438` + `fusion_samesrc03_32274` | `0.44975` | `0.36103` | `3.00` | `3,3,3,3,3,3` |
| `public_peak_33438` + `intersect_bold7h_33028` | `0.30542` | `0.17889` | `9.50` | `9,10,9,9,9,11` |
| `public_peak_33438` + `pre_tomo_33186` | `0.23548` | `0.11704` | `12.83` | `12,14,12,13,12,14` |

The selected private pair is rank 1 or 2 in all six schemes and has the best hedge geometry. The public-peak pairings are much weaker because they are too correlated with the public hillclimb.

## Test Diversity

Pairwise atom Jaccards:

| Pair | Jaccard |
|---|---:|
| `intersect_bold7h_33028` vs `fusion_samesrc03_32274` | `0.8885` |
| `public_peak_33438` vs `fusion_samesrc03_32274` | `0.9232` |
| `public_peak_33438` vs `intersect_bold7h_33028` | `0.9466` |
| `public_peak_33438` vs `sp09_33681` | `0.9891` |

Lower Jaccard is better for final-submission hedging when the private split may differ from public. `sp09_33681` is almost a clone of `public_peak_33438`.

## Operational Rule

For final selection:

- If only two finals count: select `intersect_bold7h_33028` and `fusion_samesrc03_32274`.
- If three finals count: select `intersect_bold7h_33028`, `fusion_samesrc03_32274`, and `public_peak_33438`.
- Treat `sp09_33681` as a public-rank probe unless a separate notebook and private audit justify promotion.

Public-score improvements from late single-citation removals should not displace private-survival legs.
