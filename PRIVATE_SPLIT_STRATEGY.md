# Private Split Strategy

The public leaderboard is only an approximate 50% test slice. Treat it as a
noisy diagnostic, not the objective. The final score is the other half, so our
decision process must stop using public LB movement as a gradient.

## Current Read

- Active public anchor: `0.32904`
  - `submissions/test_submission_micro_swap_art75_012_040_from_32904.csv`
  - `submissions/val_pred_micro_swap_art75_012_040_from_32904.csv`
- Recent high-local variants failed publicly:
  - `graph_chain_v49046_from_32904`: local val `0.49046`, public `0.32642`
  - `micro_repair_01`: local val `0.48843`, public `0.32634`
- Interpretation: post-`0.32904` graph/micro repairs are selection-biased.
  Their local val lift is not transferring.

## What To Stop Doing

- Do not submit another graph/micro sibling just because local val improves.
- Do not keep reshuffling generic BGG boilerplate:
  `Art. 72`, `75`, `97`, `100`, `105`, `66`, `68 BGG`.
- Do not remove domain merits citations in exchange for procedural boilerplate
  unless the procedural posture gives a clear legal reason.
- Do not treat high Jaccard as sufficient; the failed candidates were still
  high-overlap with the anchor.

## What Can Generalize

Promote a delta only when it has at least one non-LB reason:

- exact issue trigger in the query;
- same-source or near-source court support;
- independent cross-family support;
- precise statute-family fit: BGG route, StPO, ATSG/IVG, SchKG, ZPO, ZGB/OR;
- row-level legal rationale: `trigger -> legal role -> citation -> add/remove`.

Removals need a much higher burden than additions. A removal should be almost
certainly false: duplicate, wrong paragraph form, or clear statute-family
mismatch. Speculative removals are private-score poison.

## Anti-Overfit Gate

Before any future submission:

1. Run `diverse_eval` against the active anchor.
2. Run `submission_scorecard.py`.
3. Run `promotion_gate.py` as a warning signal, not an optimizer.
4. Run the Rust private-risk audit with failed-family overlap.
5. Reject if the candidate is `HIGH` or `VERY_HIGH` risk unless it was
   precommitted before public feedback and has a new independent legal signal.

Example:

```bash
./rust/v11_selector/target/release/private_risk_audit \
  --bootstrap 5000 \
  --seed 11 \
  --anchor anchor_32904=submissions/val_pred_micro_swap_art75_012_040_from_32904.csv,submissions/test_submission_micro_swap_art75_012_040_from_32904.csv \
  --anchor anchor_32107=submissions/val_pred_baseline_public_best_32107.csv,submissions/test_submission_baseline_public_best_32107.csv \
  --anchor anchor_30911=submissions/val_pred_baseline_public_best_30911.csv,submissions/test_submission_baseline_public_best_30911.csv \
  --anchor anchor_30257=submissions/val_pred_baseline_public_best_30257.csv,submissions/test_submission_baseline_public_best_30257.csv \
  --failed graph_chain_failed=submissions/test_submission_micro_swap_art75_012_040_from_32904.csv,submissions/test_submission_graph_chain_v49046_from_32904.csv \
  --failed micro_repair_failed=submissions/test_submission_micro_swap_art75_012_040_from_32904.csv,artifacts/bold_7h_32904_20260430_215722/micro_cycle_1/micro_repair_01_v48843_j96987_a16_r10_test.csv \
  --candidate CANDIDATE=VAL.csv,TEST.csv
```

## Final Portfolio Rule

Final selection is a two-leg portfolio problem:

- Leg 1: highest credible public/upside candidate, currently `0.32904`.
- Leg 2: decorrelated private hedge, not another graph/micro/BGG-repair sibling.

The second leg should be selected for different failure mode, reproducibility,
lower failed-family overlap, and conservative legal shape. Candidates like
`graph_chain` and `micro_repair_01` should not be final hedges because they are
same-family failures.
