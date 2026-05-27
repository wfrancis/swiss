# Final Submission Decision — 2026-05-10

## Situation

The public leaderboard uses approximately 50% of the test queries.  The final
ranking uses the other 50%, so public score is an overfit-prone signal.  Do not
use the public leaderboard as a tuning gradient from here.

Current public position: #3, public best `0.32904`.

## Submitted Candidate Pool

| Role | File | Public | SHA256 |
|---|---|---:|---|
| Highest public anchor | `submissions/test_submission_micro_swap_art75_012_040_from_32904.csv` | 0.32904 | `11bdde695792ff7e43eb9fb991ac5290989ab3d06d9b1d51d75d50d3a077e0ff` |
| Near-duplicate public anchor | `submissions/test_submission_sixhour_cross_high_val_45649.csv` | 0.32904 | `0d755d4e5b5c48d10f8d9aa226da3483da64369df31a2da690d33e06c83dbdda` |
| Private-upside leg | `submissions/aggressive_same_source_32904/samesrc32904_fixed12_plus_lfc1_test.csv` | 0.32562 | `6c33b45a4a58aefbac6ece3203b41033bbdf3edcbd48ef20091d7bc36f700c5a` |
| Private-survival leg | `submissions/rubik_seventeenth_order_bridge_leader_strict_commutator_test.csv` | 0.32097 | `c23b6b33f3644acae3dd00c599862a1398844400e19af5716a0d128200123563` |
| Legacy verified repro anchor | `submissions/test_submission_baseline_public_best_32107.csv` | 0.32107 | `99abea389f781bdadcdc8b8063942a20cbc95a5d765715b56bf9285b17dee5d3` |
| Legacy conservative repro hedge | `submissions/test_submission_baseline_public_best_30911.csv` | 0.30911 | `a966ba26d59bbc6bad0187369811e4cd1edbe40864fbba8ad6c21c08e6851bdf` |

## Private-Split Evidence

The sober private-split simulator evaluated 10 schemes:

- combo
- random
- domain
- procedure
- length
- gold-count
- leave-domain
- leave-procedure
- leave-length
- leave-gold-count

Top sober pair consensus:

| Pair | Support | Avg Rank | Test Jaccard |
|---|---:|---:|---:|
| `rubik_strict + samesrc_fixed12_lfc1` | 10/10 | 1.30 | 0.8762 |
| `rubik_highj_fair + samesrc_fixed12_lfc1` | 10/10 | 1.70 | 0.8754 |
| `linguist_bridge + samesrc_fixed12_lfc1` | 10/10 | 3.00 | 0.8783 |

## Hour-Pass Validation

After the quick prep was judged insufficient, a deeper pass was run on
2026-05-10/11.

New Rust exact/adversarial audit:

- Added `rust/v11_selector/src/bin/final_adversarial_audit.rs`.
- Exact enumeration over all `252` possible 5/5 hidden/public validation
  splits ranked `samesrc_fixed12_lfc1 + rubik_strict` first.
- Exact 5/5 result for `samesrc + rubik_strict`: p10 private `0.50193`,
  mean private `0.54004`, worst private `0.47045`, wins/ties `251/252`.
- Exact private-size stress also ranked `samesrc + rubik_strict` first:
  - private size 3: `116/120` wins/ties
  - private size 4: `207/210` wins/ties
  - private size 6: `210/210` wins/ties

Long sober CPU sweep:

- Output root: `artifacts/private_split_portfolio/hour_sweep_sober_20260510/`
- Manifest: `artifacts/private_split_portfolio/final_val_manifest_sober.tsv`
- Workload: 40 Rust runs = 10 split schemes x 4 seeds x 10,000,000 splits
  each = 400,000,000 randomized split simulations.
- Final consensus:
  - `rubik_strict + samesrc_fixed12_lfc1`: support `40/40`, avg rank `1.30`,
    avg best-private `0.54097`, avg test Jaccard `0.8762`.
  - `rubik_highj_fair + samesrc_fixed12_lfc1`: support `40/40`, avg rank
    `1.70`, avg best-private `0.54050`, avg test Jaccard `0.8754`.
  - `linguist_bridge + samesrc_fixed12_lfc1`: support `40/40`, avg rank
    `3.00`.
- First-place count across the 40 completed sober runs:
  - `samesrc + rubik_strict`: `28`
  - `samesrc + rubik_highj_fair`: `12`

Wide-control extension:

- Output root: `artifacts/private_split_portfolio/hour_sweep_wide_20260510/`
- Manifest: `artifacts/private_split_portfolio/final_val_manifest_wide.tsv`
- Workload completed: first 10 wide runs x 5,000,000 splits each.
- Wide controls ranked `samesrc + boldE_j955` and
  `samesrc + bold_highlift_pareto_j955` above rubik locally.
- Interpretation: this confirms that the wide pool contains higher-local-lift
  but riskier bold controls.  They remain excluded from the final recommendation
  because the sober/risk-filtered pool is the correct pool for private survival,
  and these bold controls were previously marked as likely overfit/high-risk.

Risk audit:

| Candidate | Local Signal | Private Risk | Shape vs 0.32107 |
|---|---|---|---|
| `rubik_strict` | robust local lead | LOW | J 0.976, churn 24, changed rows 15 |
| `samesrc_fixed12_lfc1` | robust local lead | HIGH | J 0.899, churn 94, changed rows 34 |
| `micro_32904` | public best | public-overfit exposed | J 0.927, churn 68 |

## Recommended Final Pair

If optimizing for the hidden/private 50% above all, select:

1. `submissions/aggressive_same_source_32904/samesrc32904_fixed12_plus_lfc1_test.csv`
2. `submissions/rubik_seventeenth_order_bridge_leader_strict_commutator_test.csv`

Rationale: this is the strongest sober private-split pair across exact
enumeration, multiple hidden-size stresses, and the 400M-split hour sweep.
`samesrc` is the high-upside/high-variance leg; `rubik_strict` is the low-churn
private-survival leg.  This pair follows the hidden-50% evidence instead of
preserving the highest public score.

Public-rank protection alternative:

1. `submissions/test_submission_micro_swap_art75_012_040_from_32904.csv`
2. `submissions/rubik_seventeenth_order_bridge_leader_strict_commutator_test.csv`

Rationale: preserves the highest public score while adding the low-churn
private-survival leg.  Risk: this trusts a public-LB-selected candidate more
than the sober private-split simulator.

Do not select both `micro_32904` and `sixhour_cross`: they are too similar and
do not hedge the hidden 50% enough.

## Locked Reproduction Helper

Use:

```bash
python3 scripts/final_submission_lock.py --list
python3 scripts/final_submission_lock.py --mode private_upside_samesrc
python3 scripts/final_submission_lock.py --mode private_survivor_rubik_strict
```

The helper writes `notebooks/_local_output/submission.csv` and verifies the
expected SHA256 before and after copying.
