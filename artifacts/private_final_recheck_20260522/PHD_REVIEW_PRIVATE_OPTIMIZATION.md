# PhD Review of Private Optimization Proposal - 2026-05-22

## Executive Verdict

The proposal is directionally correct, but it needed two upgrades:

1. a mathematical decision referee that treats public-LB candidates as biased estimators after adaptive selection
2. a physics-style overlap/replica analysis so we do not select multiple candidates from the same public-overfit basin

Both upgrades are now implemented.

New executable referee:

```text
scripts/private_final_portfolio_decision.py
```

Current output:

```text
artifacts/private_final_recheck_20260522/decision_audit/decision.md
```

Result: no challenger beats the current private baseline.

## Mathematician Analysis

### Formal Objective

The final score for a two-submission portfolio is not the average of two files. It is:

```text
S(pair, private_split) = max(S(candidate_a), S(candidate_b))
```

So the right optimizer is pair-level, not single-candidate-level.

### Bias Correction

After repeated public submissions, public score is no longer an unbiased estimator. It is selected through feedback, so the observed public score has positive selection bias.

Therefore:

- public score cannot be used as a promotion metric
- public-improving single removals mostly identify public-half rows
- public clones must be penalized unless they also improve private proxy metrics

### Robustness Metrics

The proposal originally used private split simulation correctly, but the gate was too verbal. It now has explicit dominance dimensions:

1. average `p_contains_private_winner`
2. minimum `p_contains_private_winner`
3. exact/enumerated `p10_private`
4. candidate diversity/failure-mode separation

A challenger must pass at least three dimensions and avoid hard failures.

### Current Mathematical Result

Baseline pair:

```text
final_hedge_fusion_samesrc03_32274 + intersect_bold7h_33028
```

Metrics:

```text
avg_p_contains = 0.755172
min_p_contains = 0.709240
avg_rank = 1.333
exact_p10 = 0.516823
exact_worst = 0.480027
avg_jaccard = 0.888510
```

No challenger passed the promotion gate.

## Physicist Analysis

### Landscape Model

Think of each submission as a state in a rugged energy landscape.

The public leaderboard gives feedback on one exposed basin: the public half. Repeated public hillclimbing relaxes the candidate into that basin. That can improve visible score while reducing generalization.

### Order Parameter

The useful order parameter is citation-atom overlap:

```text
q = Jaccard(candidate_a, candidate_b)
```

Interpretation:

- low/moderate `q`: different basin, useful hedge
- high `q`: correlated failures
- `q > 0.98` vs public peak: public clone, not a new private leg

Current relevant overlaps:

```text
intersect vs fusion      J = 0.8885
public_peak vs fusion    J = 0.9232
public_peak vs intersect J = 0.9466
public_peak vs sp09      J = 0.9891
```

`sp09_33681` is a local public-basin relaxation, not a new phase.

### Replica Stability

The current private pair is stable across split replicas:

```text
random, domain, combo, proc, length, leave-gold
```

It ranks:

```text
1, 2, 2, 1, 1, 1
```

That is exactly the behavior we want from a private-final pair.

## Changes Made

### Added

```text
scripts/private_final_portfolio_decision.py
```

This script:

- aggregates all `pair_report.tsv` private-split reports
- enumerates exact half-validation private splits
- computes candidate clone warnings
- flags test-only candidates as non-promotable
- writes a decision report and TSVs

### Updated

```text
artifacts/private_final_recheck_20260522/PRIVATE_OPTIMIZATION_PROTOCOL.md
```

Added:

- mathematical bias/robustness corrections
- physics overlap/replica corrections
- explicit decision-referee command
- current referee output
- hard gate against weak-hedge high-Jaccard pairs

### Generated

```text
artifacts/private_final_recheck_20260522/decision_audit/decision.md
artifacts/private_final_recheck_20260522/decision_audit/pair_decisions.tsv
artifacts/private_final_recheck_20260522/decision_audit/candidate_flags.tsv
```

## Current Decision

Keep:

```text
intersect_bold7h_33028
final_hedge_fusion_samesrc03_32274
```

If the UI truly honors three:

```text
public_peak_33438
```

Do not promote:

```text
sp09_33681
```

Reason:

```text
missing val companion; public clone J=0.9891
```

## What Would Change This Decision

Only a new candidate with a full validation companion can challenge the current pair.

It must beat the baseline in the decision referee, not merely score higher on public LB.

Minimum evidence required:

```text
python3 scripts/private_final_portfolio_decision.py ...
promote_challengers >= 1
```

Until that happens, public-leaderboard gains remain probes, not final-selection evidence.
