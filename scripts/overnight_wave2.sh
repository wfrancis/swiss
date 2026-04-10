#!/usr/bin/env bash
#
# Wave 2: More experiments to fill the 8-hour window.
#
# Stream C: GPT-5.4 judge (uses OpenAI API key, parallel with DeepSeek streams)
# Stream D: More DeepSeek variants (after A/B free up, or parallel if rate allows)
# Stream E: Meta-selector retrains on courtdense artifacts + selector sweeps on new judged data
# Stream F: Continuous perturbation search as new candidates land

set -uo pipefail

REPO="/Users/william/swiss-legal-retrieval"
cd "$REPO"

source "$REPO/.env"
PYTHON="$REPO/.venv/bin/python"
LOG_DIR="$REPO/logs/overnight_wave2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/main.log"; }

run_api_path() {
  local name="$1"
  local prompt_version="$2"
  shift 2

  local log_file="$LOG_DIR/path_${name}.log"

  # Apply env overrides from remaining args
  for kv in "$@"; do
    export "$kv"
  done
  export V11_PROMPT_VERSION="$prompt_version"
  export V11_USE_MAX_TOKENS=1
  export V11_MAX_TOKENS=12000
  export V11_JUDGE_MAX_ATTEMPTS=3
  export V11_USE_COURT_DENSE=1
  export V11_JUDGE_WORKERS=8

  local val_out="submissions/val_pred_overnight_${name}.csv"
  local test_out="submissions/test_submission_overnight_${name}.csv"

  log "  API path $name: val..."
  $PYTHON run_v11_staged.py full --split val --output "$val_out" \
    >> "$log_file" 2>&1 || { log "  $name val FAILED"; return 1; }

  log "  API path $name: test..."
  $PYTHON run_v11_staged.py full --split test --output "$test_out" \
    >> "$log_file" 2>&1 || { log "  $name test FAILED"; return 1; }

  log "  API path $name: DONE"

  # Immediate eval
  log "  API path $name: evaluating..."
  {
    $PYTHON promotion_gate.py \
      --candidate-val "$val_out" \
      --candidate-test "$test_out" 2>&1 || true
  } >> "$LOG_DIR/eval_${name}.log" 2>&1
}

run_eval() {
  local tag="$1"
  local val_csv="submissions/val_pred_${tag}.csv"
  local test_csv="submissions/test_submission_${tag}.csv"
  [[ ! -f "$val_csv" ]] || [[ ! -f "$test_csv" ]] && return 0
  {
    echo "=== $tag ==="
    $PYTHON promotion_gate.py --candidate-val "$val_csv" --candidate-test "$test_csv" 2>&1 || true
    echo ""
    $PYTHON scripts/multi_signal_scorecard.py \
      --val-gold data/val.csv \
      --reference-name baseline_30257 \
      --reference-test submissions/test_submission_baseline_public_best_30257.csv \
      --reference-val submissions/val_pred_baseline_public_best_30257.csv \
      --variant "${tag}=${val_csv},${test_csv}" 2>&1 || true
  } >> "$LOG_DIR/eval_${tag}.log" 2>&1
}

# ============================================================
# STREAM C: More DeepSeek variants (different retrieval configs)
# ============================================================
log "=== STREAM C: DeepSeek extra retrieval variants ==="
(
  export V11_API_KEY="sk-8d5e67b45fe64f43914cfa82e3aab96c"
  export V11_BASE_URL="https://api.deepseek.com/v1"
  export V11_JUDGE_MODEL="deepseek-reasoner"

  # C1: No court dense at all (pure BM25+law dense)
  run_api_path "ds_no_court_dense" "overnight_no_court_dense" \
    "V11_LAW_JUDGE_TOPK=60" "V11_COURT_JUDGE_TOPK=36" \
    "V11_USE_COURT_DENSE=0" "V11_JUDGE_PROMPT_VARIANT=default"

  # C2: Wider law k=80 + enriched prompt combo
  run_api_path "ds_k80_enriched" "overnight_k80_enriched" \
    "V11_LAW_JUDGE_TOPK=80" "V11_COURT_JUDGE_TOPK=36" \
    "V11_JUDGE_PROMPT_VARIANT=enriched"

  # C3: Max law+court with generous prompt
  run_api_path "ds_max_generous" "overnight_max_generous" \
    "V11_LAW_JUDGE_TOPK=100" "V11_COURT_JUDGE_TOPK=72" \
    "V11_COURT_DENSE_TOPK=320" "V11_JUDGE_PROMPT_VARIANT=generous"

  # C4: Tight court dense limit (4 instead of 8)
  run_api_path "ds_court_dense4" "overnight_court_dense4" \
    "V11_LAW_JUDGE_TOPK=60" "V11_COURT_JUDGE_TOPK=36" \
    "V11_COURT_DENSE_QUERY_LIMIT=4" "V11_JUDGE_PROMPT_VARIANT=default"

  log "Stream C complete"
) >> "$LOG_DIR/stream_c.log" 2>&1 &
PID_C=$!

# ============================================================
# STREAM D: More DeepSeek variants
# ============================================================
log "=== STREAM D: Extra DeepSeek variants ==="
(
  export V11_API_KEY="sk-8d5e67b45fe64f43914cfa82e3aab96c"
  export V11_BASE_URL="https://api.deepseek.com/v1"
  export V11_JUDGE_MODEL="deepseek-reasoner"

  # D1: Court dense query limit 16 (up from 8)
  run_api_path "ds_court_dense16" "overnight_court_dense16" \
    "V11_LAW_JUDGE_TOPK=60" "V11_COURT_JUDGE_TOPK=36" \
    "V11_COURT_DENSE_QUERY_LIMIT=16" "V11_COURT_DENSE_TOPK=320" \
    "V11_JUDGE_PROMPT_VARIANT=default"

  # D2: Max output 50, min output 15 (allow more citations)
  run_api_path "ds_maxout50" "overnight_maxout50" \
    "V11_LAW_JUDGE_TOPK=60" "V11_COURT_JUDGE_TOPK=36" \
    "V11_MAX_OUTPUT=50" "V11_MIN_OUTPUT=15" \
    "V11_JUDGE_PROMPT_VARIANT=default"

  # D3: Court fraction 0.40 with wider court
  run_api_path "ds_court_frac40" "overnight_court_frac40" \
    "V11_LAW_JUDGE_TOPK=60" "V11_COURT_JUDGE_TOPK=72" \
    "V11_COURT_FRACTION=0.40" "V11_MIN_COURTS_IF_ANY=6" \
    "V11_JUDGE_PROMPT_VARIANT=default"

  # D4: Wider law k=80 (moderate, different from k=100)
  run_api_path "ds_law_k80" "overnight_law_k80" \
    "V11_LAW_JUDGE_TOPK=80" "V11_COURT_JUDGE_TOPK=36" \
    "V11_JUDGE_PROMPT_VARIANT=default"

  log "Stream D complete"
) >> "$LOG_DIR/stream_d.log" 2>&1 &
PID_D=$!

# ============================================================
# STREAM E: Meta-selector retrains on courtdense artifacts (no API)
# ============================================================
log "=== STREAM E: Meta-selector experiments ==="
(
  # E1: Retrain meta-selector on courtdense-aligned judged artifacts (different training data)
  TRAIN_CD="$REPO/artifacts/v11/train_v11_trainfit_local333_courtdense_restart1/judged_bundles.json"
  VAL_CD="$REPO/artifacts/v11/val_v11_courtdense_ds_val1/judged_bundles.json"
  TEST_CD="$REPO/artifacts/v11/test_v11_courtdense_ds_test1/judged_bundles.json"

  if [[ -f "$TRAIN_CD" ]] && [[ -f "$VAL_CD" ]]; then
    log "  E1: Meta-selector retrain on courtdense artifacts (3000 search)"
    mkdir -p "$REPO/artifacts/overnight_meta_cd"
    $PYTHON run_v11_meta_selector.py \
      --train-judged "$TRAIN_CD" \
      --train-gold "$REPO/data/train.csv" \
      --apply-judged "$VAL_CD" \
      --apply-gold "$REPO/data/val.csv" \
      --output-csv "$REPO/submissions/val_pred_overnight_meta_cd_3k.csv" \
      --model-out "$REPO/artifacts/overnight_meta_cd/meta_selector.pkl" \
      --config-out "$REPO/artifacts/overnight_meta_cd/meta_selector.json" \
      --random-search 3000 --folds 5 --seed 42 \
      >> "$LOG_DIR/path_meta_cd.log" 2>&1 || log "  E1 meta_cd FAILED"

    # Apply to test if model was saved
    if [[ -f "$REPO/artifacts/overnight_meta_cd/meta_selector.pkl" ]] && [[ -f "$TEST_CD" ]]; then
      log "  E1b: Applying meta-selector to courtdense test"
      $PYTHON scripts/apply_saved_meta_selector.py \
        --model "$REPO/artifacts/overnight_meta_cd/meta_selector.pkl" \
        --config "$REPO/artifacts/overnight_meta_cd/meta_selector.json" \
        --judged "$TEST_CD" \
        --output "$REPO/submissions/test_submission_overnight_meta_cd_3k.csv" \
        >> "$LOG_DIR/path_meta_cd.log" 2>&1 || log "  E1b apply FAILED"
    fi
  fi

  # E2: Retrain on strict_v1 with higher search (5000 iterations, different seed)
  TRAIN_STRICT="$REPO/artifacts/v11/train_v11_trainfit_local200/judged_bundles.json"
  VAL_STRICT="$REPO/artifacts/v11/val_v11_strict_v1/judged_bundles.json"

  if [[ -f "$TRAIN_STRICT" ]] && [[ -f "$VAL_STRICT" ]]; then
    log "  E2: Meta-selector retrain strict_v1 (5000 search, seed=99)"
    mkdir -p "$REPO/artifacts/overnight_meta_strict5k"
    $PYTHON run_v11_meta_selector.py \
      --train-judged "$TRAIN_STRICT" \
      --train-gold "$REPO/data/train.csv" \
      --apply-judged "$VAL_STRICT" \
      --apply-gold "$REPO/data/val.csv" \
      --output-csv "$REPO/submissions/val_pred_overnight_meta_strict5k.csv" \
      --model-out "$REPO/artifacts/overnight_meta_strict5k/meta_selector.pkl" \
      --config-out "$REPO/artifacts/overnight_meta_strict5k/meta_selector.json" \
      --random-search 5000 --folds 5 --seed 99 \
      >> "$LOG_DIR/path_meta_strict5k.log" 2>&1 || log "  E2 meta_strict5k FAILED"
  fi

  # E3: Selector sweep on courtdense judged artifacts
  if [[ -f "$REPO/artifacts/v11/val_v11_courtdense_ds_val1/judged_bundles.pkl" ]]; then
    log "  E3: Selector sweep on courtdense artifacts"
    $PYTHON scripts/overnight_selector_sweep.py \
      --val-judged "$REPO/artifacts/v11/val_v11_courtdense_ds_val1/judged_bundles.pkl" \
      --test-judged "$REPO/artifacts/v11/test_v11_courtdense_ds_test1/judged_bundles.pkl" \
      --top-k 3 \
      --output-dir "$REPO/submissions" \
      >> "$LOG_DIR/path_selector_sweep_cd.log" 2>&1 || log "  E3 FAILED"
    # Rename to avoid clash with wave1 sweep
    for i in 1 2 3; do
      [[ -f "submissions/val_pred_overnight_selector_sweep_top${i}.csv" ]] && \
        cp "submissions/val_pred_overnight_selector_sweep_top${i}.csv" \
           "submissions/val_pred_overnight_selector_sweep_cd_top${i}.csv" 2>/dev/null
      [[ -f "submissions/test_submission_overnight_selector_sweep_top${i}.csv" ]] && \
        cp "submissions/test_submission_overnight_selector_sweep_top${i}.csv" \
           "submissions/test_submission_overnight_selector_sweep_cd_top${i}.csv" 2>/dev/null
    done
  fi

  log "Stream E complete"
) >> "$LOG_DIR/stream_e.log" 2>&1 &
PID_E=$!

# ============================================================
# STREAM F: Continuous perturbation as new candidates land
# ============================================================
log "=== STREAM F: Rolling perturbation search ==="
(
  # Wait for some API paths to finish (check every 2 minutes)
  WINNER_BASE="submissions/test_submission_baseline_public_best_30257.csv"
  WINNER_VAL="submissions/val_pred_baseline_public_best_30257.csv"

  for round in 1 2 3 4 5 6 7 8; do
    sleep 120  # wait 2 min between checks

    # Collect all overnight test CSVs that exist NOW
    TESTS=()
    for f in submissions/test_submission_overnight_*.csv; do
      [[ -f "$f" ]] && TESTS+=("$f")
    done

    if [[ ${#TESTS[@]} -lt 3 ]]; then
      log "  Perturb round $round: only ${#TESTS[@]} candidates, need 3. Waiting..."
      continue
    fi

    log "  Perturb round $round: ${#TESTS[@]} candidates available"

    # Pick 3 random neighbors from the available ones
    SHUFFLED=($(printf '%s\n' "${TESTS[@]}" | sort -R | head -3))
    VALS=()
    for t in "${SHUFFLED[@]}"; do
      v="${t/test_submission_/val_pred_}"
      [[ -f "$v" ]] && VALS+=("$v") || VALS+=("$WINNER_VAL")
    done

    log "  Perturb round $round neighbors: ${SHUFFLED[*]}"
    $PYTHON scripts/winner_localperturb_search.py \
      --base-test-csv "$WINNER_BASE" \
      --base-val-csv "$WINNER_VAL" \
      --neighbor-test-csv "${SHUFFLED[0]}" \
      --neighbor-test-csv "${SHUFFLED[1]}" \
      --neighbor-test-csv "${SHUFFLED[2]}" \
      --neighbor-val-csv "${VALS[0]}" \
      --neighbor-val-csv "${VALS[1]}" \
      --neighbor-val-csv "${VALS[2]}" \
      --output-dir submissions \
      --val-output-name "val_pred_overnight_perturb_r${round}.csv" \
      --test-output-name "test_submission_overnight_perturb_r${round}.csv" \
      >> "$LOG_DIR/path_perturb_r${round}.log" 2>&1 || log "  Perturb round $round FAILED"
  done

  log "Stream F complete"
) >> "$LOG_DIR/stream_f.log" 2>&1 &
PID_F=$!

log "Launched streams C/D/E/F. PIDs: C=$PID_C D=$PID_D E=$PID_E F=$PID_F"
log "Waiting for all streams..."
wait $PID_C $PID_D $PID_E $PID_F 2>/dev/null

# ============================================================
# Final evaluation of all wave2 candidates
# ============================================================
log "=== WAVE 2 EVALUATION ==="

for f in submissions/val_pred_overnight_*.csv; do
  [[ ! -f "$f" ]] && continue
  tag="${f#submissions/val_pred_}"
  tag="${tag%.csv}"
  # Skip if already evaluated in wave1
  [[ -f "logs/overnight_20260409_212745/eval_${tag}.log" ]] && continue
  run_eval "$tag"
done

# Summary
log ""
log "=== WAVE 2 SUMMARY ==="
for elog in "$LOG_DIR"/eval_*.log; do
  [[ ! -f "$elog" ]] && continue
  tag="$(basename "$elog" .log | sed 's/^eval_//')"
  verdict=$(grep "combined_verdict=" "$elog" 2>/dev/null | head -1 | cut -d= -f2)
  f1=$(grep "val_macro_f1=" "$elog" 2>/dev/null | head -1 | cut -d= -f2)
  log "  $tag: verdict=$verdict f1=$f1"
done

log "=== WAVE 2 COMPLETE ==="
