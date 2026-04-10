#!/usr/bin/env bash
#
# Overnight DeepSeek court-dense alignment sweep.
#
# Produces aligned DeepSeek-judged court-dense candidate pools for
# train / val / test so that a meta-selector trained on train can
# be applied fairly to val/test. This is the foundation for every
# downstream court-recall lever.
#
# Stages:
#   A) judge train (existing court-dense build, 333 queries)
#   B) build + judge val (10 queries, new aligned prompt)
#   C) build + judge test (40 queries, new aligned prompt)
#   D) retrain meta-selector on train_restart1 -> val_courtdense_ds
#
# Requirements:
#   V11_API_KEY must be set to a DeepSeek API key before launching.
#
# Safety:
#   - Every judge call is sha1-cached, so if a stage dies mid-way,
#     re-running this script skips already-judged batches.
#   - Uses a NEW prompt_version per split so existing val/test GPT-5.4
#     judged artifacts are never overwritten.

set -euo pipefail

REPO="/Users/william/swiss-legal-retrieval"
cd "$REPO"

# --- Required env ---
if [[ -z "${V11_API_KEY:-}" ]]; then
  echo "ERROR: V11_API_KEY must be set (DeepSeek key)." >&2
  exit 1
fi

# --- Fixed config for the whole sweep ---
export V11_BASE_URL="${V11_BASE_URL:-https://api.deepseek.com/v1}"
export V11_JUDGE_MODEL="${V11_JUDGE_MODEL:-deepseek-reasoner}"
export V11_USE_MAX_TOKENS="${V11_USE_MAX_TOKENS:-1}"
export V11_MAX_TOKENS="${V11_MAX_TOKENS:-12000}"
export V11_JUDGE_MAX_ATTEMPTS="${V11_JUDGE_MAX_ATTEMPTS:-3}"
export V11_USE_COURT_DENSE=1
export V11_JUDGE_WORKERS=8

PYTHON="${PYTHON_BIN:-$REPO/.venv/bin/python}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO/logs"
LOG_FILE="$LOG_DIR/overnight_courtdense_ds_${TS}.log"
STATUS_FILE="$LOG_DIR/overnight_courtdense_ds_status.json"
mkdir -p "$LOG_DIR"

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg" | tee -a "$LOG_FILE"
}

write_status() {
  local stage="$1"
  local state="$2"
  local elapsed="${3:-0}"
  cat > "$STATUS_FILE" <<EOF
{
  "current_stage": "$stage",
  "state": "$state",
  "elapsed_seconds_in_stage": $elapsed,
  "log_file": "$LOG_FILE",
  "updated": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF
}

CURRENT_STAGE="init"
CURRENT_STAGE_START=0

on_exit() {
  local rc=$?
  if [[ "$rc" != "0" ]]; then
    local elapsed=0
    if [[ "$CURRENT_STAGE_START" -gt 0 ]]; then
      elapsed=$(( $(date +%s) - CURRENT_STAGE_START ))
    fi
    write_status "$CURRENT_STAGE" "crashed" "$elapsed"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] TRAP: sweep exited with rc=$rc in stage $CURRENT_STAGE" >> "$LOG_FILE"
  fi
}
trap on_exit EXIT

run_stage() {
  local stage_name="$1"
  local label="$2"
  shift 2
  log ""
  log "=== Stage $stage_name: $label ==="
  local start_ts
  start_ts=$(date +%s)
  CURRENT_STAGE="$stage_name"
  CURRENT_STAGE_START="$start_ts"
  write_status "$stage_name" "running" 0
  "$@" 2>&1 | tee -a "$LOG_FILE"
  local rc="${PIPESTATUS[0]}"
  local end_ts elapsed
  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))
  if [[ "$rc" != "0" ]]; then
    write_status "$stage_name" "failed" "$elapsed"
    log "Stage $stage_name FAILED after ${elapsed}s (rc=$rc)"
    exit "$rc"
  fi
  write_status "$stage_name" "done" "$elapsed"
  log "Stage $stage_name complete in ${elapsed}s"
  CURRENT_STAGE_START=0
}

log "=== Overnight DeepSeek court-dense alignment sweep ==="
log "Log file: $LOG_FILE"
log "Status file: $STATUS_FILE"
log "Python: $PYTHON"
log "Judge model: $V11_JUDGE_MODEL"
log "Judge workers: $V11_JUDGE_WORKERS"
log "Court dense: $V11_USE_COURT_DENSE"
log "Max tokens: $V11_MAX_TOKENS"

# ============================================================
# Stage A: train judge (existing court-dense build, 333 queries)
# ============================================================
unset V11_QUERY_IDS_PATH
export V11_PROMPT_VERSION="v11_trainfit_local333_courtdense_restart1"
run_stage "A" "judge train (333 queries, existing build)" \
  "$PYTHON" run_v11_staged.py judge --split train

# ============================================================
# Stage B: val build + judge (10 queries, new aligned prompt)
# ============================================================
unset V11_QUERY_IDS_PATH
export V11_PROMPT_VERSION="v11_courtdense_ds_val1"
run_stage "B1" "build val (10 queries)" \
  "$PYTHON" run_v11_staged.py build --split val
run_stage "B2" "judge val (10 queries)" \
  "$PYTHON" run_v11_staged.py judge --split val

# ============================================================
# Stage C: test build + judge (40 queries, new aligned prompt)
# ============================================================
unset V11_QUERY_IDS_PATH
export V11_PROMPT_VERSION="v11_courtdense_ds_test1"
run_stage "C1" "build test (40 queries)" \
  "$PYTHON" run_v11_staged.py build --split test
run_stage "C2" "judge test (40 queries)" \
  "$PYTHON" run_v11_staged.py judge --split test

# ============================================================
# Stage D: meta-selector retrain on aligned data
# ============================================================
TRAIN_JUDGED="$REPO/artifacts/v11/train_v11_trainfit_local333_courtdense_restart1/judged_bundles.json"
VAL_JUDGED="$REPO/artifacts/v11/val_v11_courtdense_ds_val1/judged_bundles.json"
TEST_JUDGED="$REPO/artifacts/v11/test_v11_courtdense_ds_test1/judged_bundles.json"
OUT_CSV="$REPO/submissions/val_pred_v11_courtdense_ds_sweep.csv"
OUT_MODEL="$REPO/artifacts/v11_meta/meta_selector_courtdense_ds.pkl"
OUT_CONFIG="$REPO/artifacts/v11_meta/meta_selector_courtdense_ds.json"

run_stage "D" "meta-selector retrain (train_restart1 -> val_courtdense_ds)" \
  "$PYTHON" run_v11_meta_selector.py \
    --train-judged "$TRAIN_JUDGED" \
    --train-gold "$REPO/data/train.csv" \
    --apply-judged "$VAL_JUDGED" \
    --apply-gold "$REPO/data/val.csv" \
    --output-csv "$OUT_CSV" \
    --model-out "$OUT_MODEL" \
    --config-out "$OUT_CONFIG" \
    --random-search 500 \
    --folds 5 \
    --seed 0

log ""
log "=== Overnight sweep DONE ==="
log "Aligned judged artifacts:"
log "  train: $TRAIN_JUDGED"
log "  val:   $VAL_JUDGED"
log "  test:  $TEST_JUDGED"
log "Meta-selector:"
log "  model:  $OUT_MODEL"
log "  config: $OUT_CONFIG"
log "  val predictions: $OUT_CSV"
write_status "DONE" "done" 0
