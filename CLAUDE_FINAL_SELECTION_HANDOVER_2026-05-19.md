# Claude Handover: Execute Kaggle Final Selection

Date: 2026-05-19
Competition: LLM Agentic Legal Information Retrieval
Kaggle URL: https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/submissions

## Mission

Execution only: use the logged-in Kaggle web UI to select the two final submissions for private judging.

Do not generate new submissions. Do not run new search. Do not change candidate choice. Do not use the public leaderboard as a feedback loop.

## Final Answer: Select Exactly These Two

1. `test_submission_private_rethink_intersect_bold7h_j955.csv`
   - Kaggle ref: `52819486`
   - Public score: `0.33028`
   - Local path: `submissions/staff3_pairing_20260513/test_submission_private_rethink_intersect_bold7h_j955.csv`
   - SHA256: `542b40471aec01c53a891cccab44e8b3495cf6bb06e6c1a521fe88da61afd8ca`
   - Role: precision/intersection leg, higher ceiling in private-split audit.

2. `test_submission_final_hedge_overlay_fusion01_plus_samesrc03.csv`
   - Kaggle ref: `52596721`
   - Public score: `0.32274`
   - Local path: `submissions/final_staff_level_20260513/test_submission_final_hedge_overlay_fusion01_plus_samesrc03.csv`
   - SHA256: `163f0bba09ca6e07d648a62360abb09bb63f45f227e3e421b3f569b84225d5d2`
   - Role: recall/low-variance hedge leg.

These are the two final submissions to lock.

## Hard Guardrails

- Official rules say up to **two (2)** final submissions for judging. Do not select a third final.
- If the UI appears to allow 3 finals, screenshot the state and pause. Do not click the third.
- Do not select `52758343` unless the user explicitly overrides this handover after seeing the warning below.
- Do not use Computer Use if browser clicks are blocked. Use Chrome MCP / Claude-in-Chrome tools with the user’s logged-in Chrome profile.
- Do not automate Gmail/Kaggle login. If not logged in, pause and ask the user to sign in.
- Before accepting any confirmation dialog, verify the dialog names the intended submission/file.

## Public-Peak Backup, Not Selected By Default

`test_submission_33385_nextrem_03_est33390.csv`
- Kaggle ref: `52758343`
- Public score: `0.33438`
- SHA256: `89acefcdc37eeaf7d08b99559427a5167e7997cf5304a02278c7f09e27c85b9b`

This is the public peak. It is **not** part of the recommended two-final private-first pair. If the user explicitly asks for public-score insurance instead of private-first, the better public-inclusive pair is `52758343 + 52596721`, not `52758343 + 52819486`.

## Why This Pair

Codex performed a deep audit of Claude’s prior plan and corrected overstatements.

True claims:
- The chosen pair is rank 1 across all six private-split schemes for:
  - `mean_best_private`
  - `p_contains_private_winner`
- Average across six schemes:
  - `intersect + hedge`: avg rank `1.33`, avg best-private `0.557871`, avg `p_contains` `0.75517`, test Jaccard `0.888510`
  - `hedge + pre_tomo`: avg rank `1.67`, avg best-private `0.557446`, avg `p_contains` `0.68523`, test Jaccard `0.931073`
  - `public_peak + hedge`: avg rank `3.00`, avg best-private `0.557446`, avg `p_contains` `0.44975`, test Jaccard `0.923164`

Corrected false/overstated claims:
- The chosen pair does not Pareto-dominate on variance or raw Jaccard.
- It is not the lowest-Jaccard pair overall.
- It is not rank 1 on every auxiliary metric.
- The pair is nested: `hedge = intersect + 98 test citations`, with zero removals relative to `intersect`. This is acceptable because the finals hedge precision vs recall.

Decision: still select `52819486 + 52596721`.

## Local Verification Commands

Run only if useful; these are read-only except optional output copy if `--out` is supplied.

```bash
cd /Users/william/swiss-legal-retrieval
python3 scripts/final_submission_lock.py --list
shasum -a 256 \
  submissions/staff3_pairing_20260513/test_submission_private_rethink_intersect_bold7h_j955.csv \
  submissions/final_staff_level_20260513/test_submission_final_hedge_overlay_fusion01_plus_samesrc03.csv \
  submissions/public_precision_targeted_20260518/live_refit_after_33385/test_submission_33385_nextrem_03_est33390.csv
```

Expected hashes:

```text
542b40471aec01c53a891cccab44e8b3495cf6bb06e6c1a521fe88da61afd8ca  ...intersect_bold7h_j955.csv
163f0bba09ca6e07d648a62360abb09bb63f45f227e3e421b3f569b84225d5d2  ...final_hedge_overlay_fusion01_plus_samesrc03.csv
89acefcdc37eeaf7d08b99559427a5167e7997cf5304a02278c7f09e27c85b9b  ...33385_nextrem_03_est33390.csv
```

## Browser Execution Procedure

1. Connect to the user’s existing logged-in Chrome profile/tab using Chrome MCP / Claude-in-Chrome.
2. Navigate to:
   `https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/submissions`
3. Verify logged-in state and that the team/account is correct.
4. Locate row by exact filename or ref:
   - `52819486` / `test_submission_private_rethink_intersect_bold7h_j955.csv`
   - `52596721` / `test_submission_final_hedge_overlay_fusion01_plus_samesrc03.csv`
5. Before clicking, inspect current final-selected state. If two submissions are already selected, note which one will be replaced.
6. Click `Use for Final Score` / `Use for Final Submission` for `52819486`.
7. Confirm only if the dialog clearly refers to the intended submission.
8. Click `Use for Final Score` / `Use for Final Submission` for `52596721`.
9. Confirm only if the dialog clearly refers to the intended submission.
10. Re-read the page and verify both target rows visibly show selected-for-final status.
11. Take a screenshot of the final locked state. Suggested durable path:
    `artifacts/final_selection_20260519/kaggle_ui_confirmation.png`

## Completion Criteria

Report back with:

- Whether both final selections are visible in Kaggle UI.
- Exact selected refs and filenames:
  - `52819486` `test_submission_private_rethink_intersect_bold7h_j955.csv`
  - `52596721` `test_submission_final_hedge_overlay_fusion01_plus_samesrc03.csv`
- Screenshot path if saved.
- Any discrepancy, especially if the UI shows a different final-submission limit.

## Manual Fallback

If Chrome MCP fails, tell the user to manually select exactly:

1. Ref `52819486` — `test_submission_private_rethink_intersect_bold7h_j955.csv`
2. Ref `52596721` — `test_submission_final_hedge_overlay_fusion01_plus_samesrc03.csv`

Then ask them to confirm both show selected for final score.

## Execution Note (Appended 2026-05-19 ~12:55 MDT)

This handover was executed by Claude with one user-directed override of the "do not select a third final" guardrail.

**UI vs written rules conflict discovered at execution time:**
- Kaggle UI text on the submissions page: *"Select up to 3 submissions that will count towards your final leaderboard score. If less than 3 are selected, Kaggle will automatically select from your best scoring submissions."*
- Competition-Specific Rule 2.2.b: *"You may select up to two (2) Final Submissions for judging."*
- Conclusion: UI auto-fills to 3 regardless of user choice, making the 2-cap effectively impossible to comply with via the UI. The "do not select a third" guardrail in this handover assumed a strict 2-cap that the UI does not enforce.

**User decision** (via AskUserQuestion at 12:54 MDT): "Lock 3 explicitly" — manually selecting all three (intersect_bold7h + final_hedge_fusion + public_peak) is preferable to letting Kaggle auto-pick the third slot at the deadline with uncontrolled criteria.

**Final locked state** (verified `3/3` in Kaggle UI):
1. Ref `52819486` — `test_submission_private_rethink_intersect_bold7h_j955.csv` (public 0.33028)
2. Ref `52596721` — `test_submission_final_hedge_overlay_fusion01_plus_samesrc03.csv` (public 0.32274)
3. Ref `52758343` — `test_submission_33385_nextrem_03_est33390.csv` (public 0.33438) — added per override

**Prior auto-selected finals replaced**: `test_submission_combo_layer_1.csv` (0.30911) and `test_submission_baseline_public_best_30257.csv` (0.30257), both ~1mo stale from a prior pipeline era.

**Pending action**: A discussion-forum post to the competition host has been queued (separate task) asking whether the host scores 2 or 3 finals — to remove the residual rule-vs-UI ambiguity before the 2026-05-24 21:55 UTC deadline.

**Audit log**: `artifacts/final_selection_20260519/kaggle_ui_confirmation.md` (full timestamped action sequence + page-text extracts).
