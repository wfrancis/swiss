#!/usr/bin/env bash
#
# Overnight multi-path experiment runner (8 hours).
#
# Runs 6+ experiment paths, evaluates all candidates, runs combination
# phase, and logs everything to SCORE_TRACKER.md.
#
# Prerequisites:
#   V11_API_KEY must be set to a DeepSeek API key.
#   .env must have OPENAI_API_KEY (for embeddings).
#   Rust binary: rust/v11_selector/target/release/hybrid_lab
#   Judged artifacts: artifacts/v11/{val,test}_v11_strict_v1/judged_bundles.{json,pkl}
#
# Usage:
#   export V11_API_KEY=sk-...
#   bash scripts/overnight_multi_path.sh

set -uo pipefail

REPO="/Users/william/swiss-legal-retrieval"
cd "$REPO"

PYTHON="$REPO/.venv/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO/logs/overnight_$TS"
mkdir -p "$LOG_DIR"

# --- Validate prerequisites ---
if [[ -z "${V11_API_KEY:-}" ]]; then
  echo "ERROR: V11_API_KEY must be set (DeepSeek key)." >&2
  exit 1
fi

for f in \
  artifacts/v11/val_v11_strict_v1/judged_bundles.pkl \
  artifacts/v11/test_v11_strict_v1/judged_bundles.pkl \
  artifacts/v11/val_v11_strict_v1/judged_bundles.json \
  artifacts/v11/test_v11_strict_v1/judged_bundles.json \
  submissions/test_submission_baseline_public_best_30257.csv \
  submissions/val_pred_baseline_public_best_30257.csv \
  data/val.csv data/test.csv; do
  if [[ ! -f "$REPO/$f" ]]; then
    echo "ERROR: Missing prerequisite: $f" >&2
    exit 1
  fi
done
echo "[$(date)] Prerequisites OK" | tee "$LOG_DIR/main.log"

# --- Common exports ---
export V11_BASE_URL="${V11_BASE_URL:-https://api.deepseek.com/v1}"
export V11_JUDGE_MODEL="${V11_JUDGE_MODEL:-deepseek-reasoner}"
export V11_USE_MAX_TOKENS=1
export V11_MAX_TOKENS=12000
export V11_JUDGE_MAX_ATTEMPTS=3
export V11_USE_COURT_DENSE=1
export V11_JUDGE_WORKERS=8

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/main.log"; }

run_path() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/path_${name}.log"
  log "START path=$name"
  if "$@" >> "$log_file" 2>&1; then
    log "DONE  path=$name (success)"
    return 0
  else
    log "FAIL  path=$name (rc=$?)"
    return 0  # don't kill orchestrator
  fi
}

run_eval() {
  local tag="$1"
  local val_csv="submissions/val_pred_${tag}.csv"
  local test_csv="submissions/test_submission_${tag}.csv"
  if [[ ! -f "$val_csv" ]] || [[ ! -f "$test_csv" ]]; then
    log "SKIP eval $tag (missing CSVs)"
    return 0
  fi
  local eval_log="$LOG_DIR/eval_${tag}.log"
  log "EVAL $tag"
  {
    echo "=== SCORECARD: $tag ==="
    $PYTHON scripts/multi_signal_scorecard.py \
      --val-gold data/val.csv \
      --reference-name baseline_30257 \
      --reference-test submissions/test_submission_baseline_public_best_30257.csv \
      --reference-val submissions/val_pred_baseline_public_best_30257.csv \
      --variant "${tag}=${val_csv},${test_csv}" 2>&1 || true

    echo ""
    echo "=== PROMOTION GATE: $tag ==="
    $PYTHON promotion_gate.py \
      --candidate-val "$val_csv" \
      --candidate-test "$test_csv" 2>&1 || true

    echo ""
    echo "=== SUBMISSION SCORECARD: $tag ==="
    $PYTHON submission_scorecard.py \
      --val-csv "$val_csv" \
      --test-csv "$test_csv" \
      --ref-test "baseline=submissions/test_submission_baseline_public_best_30257.csv" 2>&1 || true
  } >> "$eval_log" 2>&1
  log "EVAL $tag done -> $eval_log"
}

# ============================================================
# PHASE 1: Instant paths (no API calls)
# ============================================================
log "=== PHASE 1: Local-only paths (parallel) ==="

# Path 3: FAISS injection
run_path "faiss_inject" $PYTHON scripts/overnight_faiss_inject.py &
PID_FAISS=$!

# Path 5: Ensemble vote
run_path "ensemble_vote" $PYTHON scripts/overnight_ensemble_vote.py &
PID_ENSEMBLE=$!

# Path 6a: Selector parameter sweep
run_path "selector_sweep" $PYTHON scripts/overnight_selector_sweep.py --top-k 5 &
PID_SELECTOR=$!

# Wait for all local paths
wait $PID_FAISS $PID_ENSEMBLE $PID_SELECTOR 2>/dev/null
log "=== PHASE 1 complete ==="

# ============================================================
# PHASE 2: API-heavy paths (staggered)
# ============================================================
log "=== PHASE 2: API paths ==="

# --- Path A1: Wider law k=100 ---
run_api_path() {
  local name="$1"
  local prompt_version="$2"
  shift 2

  # Export all env overrides
  for kv in "$@"; do
    export "$kv"
  done
  export V11_PROMPT_VERSION="$prompt_version"

  local val_out="submissions/val_pred_overnight_${name}.csv"
  local test_out="submissions/test_submission_overnight_${name}.csv"

  log "  API path $name: build+judge+select val..."
  $PYTHON run_v11_staged.py full --split val --output "$val_out" \
    >> "$LOG_DIR/path_${name}.log" 2>&1 || { log "  $name val FAILED"; return 1; }

  log "  API path $name: build+judge+select test..."
  $PYTHON run_v11_staged.py full --split test --output "$test_out" \
    >> "$LOG_DIR/path_${name}.log" 2>&1 || { log "  $name test FAILED"; return 1; }

  log "  API path $name: done"
}

# Stream A: wider_law -> enriched_prompt -> generous_prompt
(
  log "Stream A starting"
  # A1: Wider law k=100
  run_api_path "wider_law_k100" "overnight_wider_law_k100" \
    "V11_LAW_JUDGE_TOPK=100" "V11_COURT_JUDGE_TOPK=36" \
    "V11_JUDGE_PROMPT_VARIANT=default"

  # A2: Enriched judge prompt
  run_api_path "enriched_prompt" "overnight_enriched_prompt" \
    "V11_LAW_JUDGE_TOPK=60" "V11_COURT_JUDGE_TOPK=36" \
    "V11_JUDGE_PROMPT_VARIANT=enriched"

  # A3: Generous judge prompt
  run_api_path "generous_prompt" "overnight_generous_prompt" \
    "V11_LAW_JUDGE_TOPK=60" "V11_COURT_JUDGE_TOPK=36" \
    "V11_JUDGE_PROMPT_VARIANT=generous"

  log "Stream A complete"
) >> "$LOG_DIR/stream_a.log" 2>&1 &
PID_STREAM_A=$!

# Stream B: court_k72 -> wider_both -> strict_prompt
(
  log "Stream B starting"
  # B1: Aggressive court k=72
  run_api_path "court_k72" "overnight_court_k72" \
    "V11_LAW_JUDGE_TOPK=60" "V11_COURT_JUDGE_TOPK=72" \
    "V11_COURT_DENSE_TOPK=320" "V11_JUDGE_PROMPT_VARIANT=default"

  # B2: Both wider: law k=100 + court k=72
  run_api_path "wider_both" "overnight_wider_both" \
    "V11_LAW_JUDGE_TOPK=100" "V11_COURT_JUDGE_TOPK=72" \
    "V11_COURT_DENSE_TOPK=320" "V11_JUDGE_PROMPT_VARIANT=default"

  # B3: Strict judge prompt
  run_api_path "strict_prompt" "overnight_strict_prompt" \
    "V11_LAW_JUDGE_TOPK=60" "V11_COURT_JUDGE_TOPK=36" \
    "V11_JUDGE_PROMPT_VARIANT=strict"

  log "Stream B complete"
) >> "$LOG_DIR/stream_b.log" 2>&1 &
PID_STREAM_B=$!

log "Both API streams launched. Waiting..."
wait $PID_STREAM_A $PID_STREAM_B 2>/dev/null
log "=== PHASE 2 complete ==="

# ============================================================
# PHASE 3: Combination — perturbation search with new neighbors
# ============================================================
log "=== PHASE 3: Combination phase ==="

# Find which overnight test CSVs actually got created
OVERNIGHT_TESTS=()
for f in submissions/test_submission_overnight_*.csv; do
  [[ -f "$f" ]] && OVERNIGHT_TESTS+=("$f")
done

log "Found ${#OVERNIGHT_TESTS[@]} overnight test CSVs for combination"

# Run winner perturbation with different neighbor sets (if enough exist)
if [[ ${#OVERNIGHT_TESTS[@]} -ge 3 ]]; then
  # Combo A: first 3 overnight CSVs
  COMBO_A_NEIGHBORS=("${OVERNIGHT_TESTS[@]:0:3}")
  # Derive val partners
  COMBO_A_VAL_NEIGHBORS=()
  for t in "${COMBO_A_NEIGHBORS[@]}"; do
    v="${t/test_submission_/val_pred_}"
    COMBO_A_VAL_NEIGHBORS+=("$v")
  done

  log "Combo A neighbors: ${COMBO_A_NEIGHBORS[*]}"
  $PYTHON scripts/winner_localperturb_search.py \
    --base-test-csv submissions/test_submission_baseline_public_best_30257.csv \
    --base-val-csv submissions/val_pred_baseline_public_best_30257.csv \
    --neighbor-test-csv "${COMBO_A_NEIGHBORS[0]}" \
    --neighbor-test-csv "${COMBO_A_NEIGHBORS[1]}" \
    --neighbor-test-csv "${COMBO_A_NEIGHBORS[2]}" \
    --neighbor-val-csv "${COMBO_A_VAL_NEIGHBORS[0]}" \
    --neighbor-val-csv "${COMBO_A_VAL_NEIGHBORS[1]}" \
    --neighbor-val-csv "${COMBO_A_VAL_NEIGHBORS[2]}" \
    --output-dir submissions \
    --val-output-name val_pred_overnight_combo_a.csv \
    --test-output-name test_submission_overnight_combo_a.csv \
    >> "$LOG_DIR/path_combo_a.log" 2>&1 || log "Combo A failed"

  # Combo B: last 3 overnight CSVs (if different)
  if [[ ${#OVERNIGHT_TESTS[@]} -ge 6 ]]; then
    COMBO_B_NEIGHBORS=("${OVERNIGHT_TESTS[@]:3:3}")
    COMBO_B_VAL_NEIGHBORS=()
    for t in "${COMBO_B_NEIGHBORS[@]}"; do
      v="${t/test_submission_/val_pred_}"
      COMBO_B_VAL_NEIGHBORS+=("$v")
    done
    log "Combo B neighbors: ${COMBO_B_NEIGHBORS[*]}"
    $PYTHON scripts/winner_localperturb_search.py \
      --base-test-csv submissions/test_submission_baseline_public_best_30257.csv \
      --base-val-csv submissions/val_pred_baseline_public_best_30257.csv \
      --neighbor-test-csv "${COMBO_B_NEIGHBORS[0]}" \
      --neighbor-test-csv "${COMBO_B_NEIGHBORS[1]}" \
      --neighbor-test-csv "${COMBO_B_NEIGHBORS[2]}" \
      --neighbor-val-csv "${COMBO_B_VAL_NEIGHBORS[0]}" \
      --neighbor-val-csv "${COMBO_B_VAL_NEIGHBORS[1]}" \
      --neighbor-val-csv "${COMBO_B_VAL_NEIGHBORS[2]}" \
      --output-dir submissions \
      --val-output-name val_pred_overnight_combo_b.csv \
      --test-output-name test_submission_overnight_combo_b.csv \
      >> "$LOG_DIR/path_combo_b.log" 2>&1 || log "Combo B failed"
  fi

  # Combo C: mix — pick selector_sweep_top1 + faiss_inject_top1 + first API path
  if [[ -f "submissions/test_submission_overnight_selector_sweep_top1.csv" ]] && \
     [[ -f "submissions/test_submission_overnight_faiss_inject_top1.csv" ]] && \
     [[ ${#OVERNIGHT_TESTS[@]} -ge 1 ]]; then
    log "Combo C: selector_sweep_top1 + faiss_inject_top1 + ${OVERNIGHT_TESTS[0]}"
    C_VAL0="${OVERNIGHT_TESTS[0]/test_submission_/val_pred_}"
    $PYTHON scripts/winner_localperturb_search.py \
      --base-test-csv submissions/test_submission_baseline_public_best_30257.csv \
      --base-val-csv submissions/val_pred_baseline_public_best_30257.csv \
      --neighbor-test-csv "submissions/test_submission_overnight_selector_sweep_top1.csv" \
      --neighbor-test-csv "submissions/test_submission_overnight_faiss_inject_top1.csv" \
      --neighbor-test-csv "${OVERNIGHT_TESTS[0]}" \
      --neighbor-val-csv "submissions/val_pred_overnight_selector_sweep_top1.csv" \
      --neighbor-val-csv "submissions/val_pred_overnight_faiss_inject_top1.csv" \
      --neighbor-val-csv "$C_VAL0" \
      --output-dir submissions \
      --val-output-name val_pred_overnight_combo_c.csv \
      --test-output-name test_submission_overnight_combo_c.csv \
      >> "$LOG_DIR/path_combo_c.log" 2>&1 || log "Combo C failed"
  fi
else
  log "Not enough overnight test CSVs for combination phase (need >=3, have ${#OVERNIGHT_TESTS[@]})"
fi

log "=== PHASE 3 complete ==="

# ============================================================
# PHASE 4: Evaluate ALL candidates
# ============================================================
log "=== PHASE 4: Evaluation ==="

for f in submissions/val_pred_overnight_*.csv; do
  [[ ! -f "$f" ]] && continue
  tag="${f#submissions/val_pred_}"
  tag="${tag%.csv}"
  run_eval "$tag"
done

# ============================================================
# PHASE 5: Summary report
# ============================================================
log "=== PHASE 5: Summary ==="

REPORT="$LOG_DIR/report.txt"
{
  echo "========================================"
  echo "OVERNIGHT MULTI-PATH EXPERIMENT REPORT"
  echo "Started: $TS"
  echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "========================================"
  echo ""

  echo "--- CANDIDATES PRODUCED ---"
  for f in submissions/test_submission_overnight_*.csv; do
    [[ ! -f "$f" ]] && continue
    tag="${f#submissions/test_submission_}"
    tag="${tag%.csv}"
    lines=$(wc -l < "$f" | tr -d ' ')
    echo "  $tag ($lines queries)"
  done

  echo ""
  echo "--- EVALUATION RESULTS ---"
  for elog in "$LOG_DIR"/eval_overnight_*.log; do
    [[ ! -f "$elog" ]] && continue
    echo ""
    echo ">>> $(basename "$elog" .log) <<<"
    grep -E "(val_macro_f1|val_bootstrap_lb90|test_jaccard|combined_verdict|F1|over_ref|mean_J)" "$elog" 2>/dev/null | head -20
  done

  echo ""
  echo "--- PROMOTION CANDIDATES ---"
  for elog in "$LOG_DIR"/eval_overnight_*.log; do
    [[ ! -f "$elog" ]] && continue
    tag="$(basename "$elog" .log)"
    verdict=$(grep "combined_verdict=" "$elog" 2>/dev/null | head -1 | cut -d= -f2)
    f1=$(grep "val_macro_f1=" "$elog" 2>/dev/null | head -1 | cut -d= -f2)
    lb90=$(grep "val_bootstrap_lb90=" "$elog" 2>/dev/null | head -1 | cut -d= -f2)
    jaccard=$(grep "test_jaccard_baseline=" "$elog" 2>/dev/null | head -1 | cut -d= -f2)
    echo "  $tag: verdict=$verdict f1=$f1 lb90=$lb90 jaccard=$jaccard"
  done
} > "$REPORT"

cat "$REPORT" | tee -a "$LOG_DIR/main.log"

log ""
log "========================================="
log "OVERNIGHT RUN COMPLETE"
log "Report: $REPORT"
log "Logs:   $LOG_DIR/"
log "========================================="
