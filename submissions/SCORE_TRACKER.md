# Score Tracker — Swiss Legal Retrieval

Track every candidate evaluation so we can see what's working and what isn't.
**Never delete rows. Only append.**

## Baseline numbers (the bar to beat)

| Metric | Winner value | Notes |
|--------|-------------|-------|
| **Kaggle public LB** | **0.30257** | Ground truth |
| Val macro F1 | 0.282430 | |
| Val bootstrap LB90 | 0.255176 | |
| Val bootstrap LB95 | 0.249112 | |
| Val per-query std | 0.067534 | |
| Val per-query min (floor) | 0.200000 | Query val_005 |
| Val avg predictions | 23.40 | |
| Val avg court fraction | 0.2173 | |
| Test avg predictions | 20.50 | |
| Test avg court fraction | 0.1763 | |
| Test law% | 81.71% | |
| Promotion gate verdict | likely_better_or_flat | |
| History LOO MAE | 0.010809 | |
| History pairwise ranking accuracy | 75.00% | |

### Per-query val F1 breakdown

| Query | F1 |
|-------|------|
| val_001 | 0.2727 |
| val_002 | 0.3077 |
| val_003 | 0.2432 |
| val_004 | 0.2759 |
| val_005 | 0.2000 |
| val_006 | 0.4324 |
| val_007 | 0.2083 |
| val_008 | 0.3673 |
| val_009 | 0.2500 |
| val_010 | 0.2667 |

## Candidate history

Each row = one candidate evaluated. Columns: name, date, val F1, LB90, per-query std, test Jaccard vs baseline, promotion verdict, Kaggle public LB (if submitted), outcome.

| # | Candidate | Date | Val F1 | LB90 | Std | Test Jaccard | Gate verdict | Kaggle LB | Outcome |
|---|-----------|------|--------|------|-----|-------------|-------------|-----------|---------|
| 1 | **overnight_combo_a** (NEW BEST) | 2026-04-10 | 0.2865 | 0.2601 | 0.0659 | 0.8240 | likely_worse (WRONG) | **0.30681** | **NEW BEST — gate was wrong, Jaccard 0.82 still won** |
| 2 | overnight_perturb_r7 | 2026-04-10 | 0.2840 | 0.2580 | 0.0657 | 0.9798 | unclear (WRONG) | **0.30532** | Also beat prior best |
| 0 | winner_localperturb_top1 (PRIOR BASELINE) | 2026-04-08 | 0.2824 | 0.2552 | 0.0675 | 1.0000 | likely_better_or_flat | 0.30257 | Dethroned by combo_a |
| -1 | consensus_loose_deepseekpriors | 2026-04-08 | — | — | — | — | — | 0.30094 | Prior best, base for winner |
| -2 | meta_trainjudged333_perturb_top1 | 2026-04-09 | — | — | — | — | — | 0.29887 | Lost vs baseline (dead zone) |
| -3 | blend_courtdense_additive_raw_n1 | 2026-04-09 | — | — | — | — | — | 0.29868 | Lost vs baseline (dead zone) |

## How to add a new candidate row

```bash
# 1. Generate val + test CSVs for your candidate
# 2. Run all three evaluation tools:

.venv/bin/python3 scripts/multi_signal_scorecard.py \
    --val-gold data/val.csv \
    --reference-name baseline_30257 \
    --reference-test submissions/test_submission_baseline_public_best_30257.csv \
    --reference-val submissions/val_pred_baseline_public_best_30257.csv \
    --variant CANDIDATE=submissions/val_pred_CANDIDATE.csv,submissions/test_submission_CANDIDATE.csv

.venv/bin/python3 promotion_gate.py \
    --candidate-val submissions/val_pred_CANDIDATE.csv \
    --candidate-test submissions/test_submission_CANDIDATE.csv

.venv/bin/python3 submission_scorecard.py \
    --val-csv submissions/val_pred_CANDIDATE.csv \
    --test-csv submissions/test_submission_CANDIDATE.csv \
    --ref-test baseline=submissions/test_submission_baseline_public_best_30257.csv

# 3. Append a row to the table above with all numbers
# 4. Only submit to Kaggle if ALL promotion rules pass (see CLAUDE.md)
```

## Lessons learned

- **1-2pp local val lift = dead zone.** Two candidates with small local lifts both lost on Kaggle (0.29887, 0.29868). Don't submit unless the signal is clearly different from baseline.
- **The winner was a tiny perturbation** — 52 law cites added, 7 removed, 29/40 queries touched. Big structural changes have underperformed.
- **Court-dense additions hurt.** The winning config used `max_add_court=0` (law-only). Court FAISS recall ceiling is 64% but the trained classifier overfits to baseline patterns.
- **Next improvement needs structurally different signal**, not another perturbation in the same neighborhood.
