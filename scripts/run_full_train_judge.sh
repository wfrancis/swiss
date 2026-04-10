#!/usr/bin/env bash
#
# Judge ALL 1,139 train queries. Fully resumable — safe to kill and restart.
#
# Every step uses append-only caches keyed by query_id or SHA1:
#   - precompute/*.json: skip queries already present
#   - precompute/judge_cache_train_v11.jsonl: skip batches already judged
#
# To restart after crash: just run this script again. It picks up where it left off.
#
# Usage:
#   export V11_API_KEY=sk-...
#   bash scripts/run_full_train_judge.sh
#
# Monitor:
#   tail -f logs/full_train_judge_latest.log
#   cat logs/full_train_judge_status.json

set -uo pipefail

REPO="/Users/william/swiss-legal-retrieval"
cd "$REPO"

source "$REPO/.env" 2>/dev/null || true

PYTHON="$REPO/.venv/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO/logs"
LOG_FILE="$LOG_DIR/full_train_judge_${TS}.log"
STATUS_FILE="$LOG_DIR/full_train_judge_status.json"
LATEST_LINK="$LOG_DIR/full_train_judge_latest.log"
mkdir -p "$LOG_DIR"
ln -sf "$LOG_FILE" "$LATEST_LINK"

# --- Validate ---
if [[ -z "${V11_API_KEY:-}" ]]; then
  echo "ERROR: V11_API_KEY must be set." >&2
  exit 1
fi

for f in data/train.csv data/val.csv; do
  [[ ! -f "$REPO/$f" ]] && { echo "ERROR: Missing $f" >&2; exit 1; }
done

# --- Config ---
export V11_BASE_URL="${V11_BASE_URL:-https://api.deepseek.com/v1}"
export V11_JUDGE_MODEL="${V11_JUDGE_MODEL:-deepseek-reasoner}"
export V11_USE_MAX_TOKENS=1
export V11_MAX_TOKENS=12000
export V11_JUDGE_MAX_ATTEMPTS=3
export V11_USE_COURT_DENSE=1
export V11_JUDGE_WORKERS=16
export V11_PROMPT_VERSION="v11_strict_v1"

CHUNK_SIZE=100  # Process train in chunks of 100 queries

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

write_status() {
  cat > "$STATUS_FILE" <<EOF
{
  "stage": "$1",
  "state": "$2",
  "detail": "$3",
  "log_file": "$LOG_FILE",
  "updated": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF
}

# Count existing progress
count_precompute() {
  local file="$1"
  if [[ -f "$file" ]]; then
    $PYTHON -c "import json; d=json.load(open('$file')); print(len(d))" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

count_judge_cache() {
  if [[ -f "precompute/judge_cache_train_v11.jsonl" ]]; then
    wc -l < "precompute/judge_cache_train_v11.jsonl" | tr -d ' '
  else
    echo 0
  fi
}

log "=== FULL TRAIN JUDGE RUN ==="
log "Goal: Judge all 1,139 train queries"
log "All caches are append-only — safe to kill and restart"
log ""

# ============================================================
# STAGE 1: Precompute (query expansions, case citations, full citations)
# ============================================================
TOTAL_TRAIN=1139

QE_DONE=$(count_precompute "precompute/train_query_expansions.json")
CC_DONE=$(count_precompute "precompute/train_case_citations.json")
FC_DONE=$(count_precompute "precompute/train_full_citations_v2.json")
JC_DONE=$(count_judge_cache)

log "Current progress:"
log "  Query expansions:  $QE_DONE / $TOTAL_TRAIN"
log "  Case citations:    $CC_DONE / $TOTAL_TRAIN"
log "  Full citations:    $FC_DONE / $TOTAL_TRAIN"
log "  Judge cache lines: $JC_DONE"
log ""

# 1a: Query expansions
if [[ $QE_DONE -lt $TOTAL_TRAIN ]]; then
  write_status "precompute" "running" "query_expansions ($QE_DONE/$TOTAL_TRAIN)"
  log "STAGE 1a: Generating query expansions ($QE_DONE done, $((TOTAL_TRAIN - QE_DONE)) remaining)..."
  $PYTHON precompute/gen_query_expansions.py train --max-workers 16 >> "$LOG_FILE" 2>&1
  QE_DONE=$(count_precompute "precompute/train_query_expansions.json")
  log "  Query expansions done: $QE_DONE"
else
  log "STAGE 1a: Query expansions complete ($QE_DONE)"
fi

# 1b: Case citations
if [[ $CC_DONE -lt $TOTAL_TRAIN ]]; then
  write_status "precompute" "running" "case_citations ($CC_DONE/$TOTAL_TRAIN)"
  log "STAGE 1b: Generating case citations ($CC_DONE done, $((TOTAL_TRAIN - CC_DONE)) remaining)..."
  $PYTHON precompute/gen_case_citations.py train --max-workers 16 >> "$LOG_FILE" 2>&1
  CC_DONE=$(count_precompute "precompute/train_case_citations.json")
  log "  Case citations done: $CC_DONE"
else
  log "STAGE 1b: Case citations complete ($CC_DONE)"
fi

# 1c: Full citations v2
if [[ $FC_DONE -lt $TOTAL_TRAIN ]]; then
  write_status "precompute" "running" "full_citations ($FC_DONE/$TOTAL_TRAIN)"
  log "STAGE 1c: Generating full citations v2 ($FC_DONE done, $((TOTAL_TRAIN - FC_DONE)) remaining)..."
  $PYTHON precompute/gen_full_citations_v2.py train --max-workers 16 >> "$LOG_FILE" 2>&1
  FC_DONE=$(count_precompute "precompute/train_full_citations_v2.json")
  log "  Full citations done: $FC_DONE"
else
  log "STAGE 1c: Full citations complete ($FC_DONE)"
fi

log ""
log "=== Precompute complete ==="
log ""

# ============================================================
# STAGE 2: Build + Judge in chunks (the main API-heavy work)
# ============================================================
# Process train in chunks of CHUNK_SIZE queries.
# Each chunk does build+judge. The judge cache is shared across all chunks,
# so if a chunk partially completes, the next restart skips cached entries.

TOTAL_CHUNKS=$(( (TOTAL_TRAIN + CHUNK_SIZE - 1) / CHUNK_SIZE ))

log "STAGE 2: Build + Judge in $TOTAL_CHUNKS chunks of $CHUNK_SIZE queries"
log "  Judge workers: $V11_JUDGE_WORKERS"
log "  Judge model: $V11_JUDGE_MODEL"
log ""

for chunk_idx in $(seq 0 $((TOTAL_CHUNKS - 1))); do
  OFFSET=$((chunk_idx * CHUNK_SIZE))
  REMAINING=$((TOTAL_TRAIN - OFFSET))
  LIMIT=$((REMAINING < CHUNK_SIZE ? REMAINING : CHUNK_SIZE))

  CHUNK_LABEL="chunk $((chunk_idx + 1))/$TOTAL_CHUNKS (offset=$OFFSET, n=$LIMIT)"

  # Check if this chunk's judged artifact already exists
  export V11_QUERY_OFFSET=$OFFSET
  export V11_MAX_QUERIES=$LIMIT
  ARTIFACT_DIR="artifacts/v11/train_${V11_PROMPT_VERSION}__offset${OFFSET}_n${LIMIT}"

  if [[ -f "$ARTIFACT_DIR/judged_bundles.pkl" ]]; then
    log "  $CHUNK_LABEL: SKIP (judged artifact exists)"
    continue
  fi

  write_status "build_judge" "running" "$CHUNK_LABEL"
  log "  $CHUNK_LABEL: build+judge starting..."

  CHUNK_START=$(date +%s)
  $PYTHON run_v11_staged.py full --split train \
    --offset $OFFSET --limit $LIMIT \
    --output "submissions/train_pred_full_chunk_${chunk_idx}.csv" \
    >> "$LOG_FILE" 2>&1

  CHUNK_RC=$?
  CHUNK_END=$(date +%s)
  CHUNK_ELAPSED=$((CHUNK_END - CHUNK_START))

  if [[ $CHUNK_RC -ne 0 ]]; then
    log "  $CHUNK_LABEL: FAILED (rc=$CHUNK_RC) after ${CHUNK_ELAPSED}s"
    write_status "build_judge" "failed" "$CHUNK_LABEL failed after ${CHUNK_ELAPSED}s"
    log ""
    log "Safe to restart — all completed work is cached."
    log "Run: bash scripts/run_full_train_judge.sh"
    exit 1
  fi

  JC_NOW=$(count_judge_cache)
  log "  $CHUNK_LABEL: DONE in ${CHUNK_ELAPSED}s (judge cache: $JC_NOW entries)"
done

log ""
log "=== All chunks complete ==="
log ""

# ============================================================
# STAGE 3: Verify and report
# ============================================================
write_status "verify" "running" "checking coverage"

JC_FINAL=$(count_judge_cache)
log "STAGE 3: Verification"
log "  Total judge cache entries: $JC_FINAL"
log "  Total train queries: $TOTAL_TRAIN"
log ""

# Count how many chunk artifacts exist
CHUNK_COUNT=$(ls -d artifacts/v11/train_${V11_PROMPT_VERSION}__offset*/judged_bundles.pkl 2>/dev/null | wc -l | tr -d ' ')
log "  Judged chunk artifacts: $CHUNK_COUNT / $TOTAL_CHUNKS"

if [[ $CHUNK_COUNT -eq $TOTAL_CHUNKS ]]; then
  log ""
  log "========================================="
  log "FULL TRAIN JUDGE COMPLETE"
  log "All $TOTAL_TRAIN train queries judged."
  log "Judge cache: precompute/judge_cache_train_v11.jsonl ($JC_FINAL entries)"
  log "Chunk artifacts: artifacts/v11/train_${V11_PROMPT_VERSION}__offset*/"
  log ""
  log "Next step: merge chunks and run selector sweep"
  log "========================================="
  write_status "DONE" "complete" "all $TOTAL_TRAIN queries judged"
else
  log "WARNING: Only $CHUNK_COUNT / $TOTAL_CHUNKS chunks complete."
  log "Re-run this script to complete remaining chunks."
  write_status "partial" "incomplete" "$CHUNK_COUNT/$TOTAL_CHUNKS chunks done"
fi
