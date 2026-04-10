#!/usr/bin/env bash
#
# Multi-pass judge: run 5 independent judge variants on ALL train queries.
#
# Each pass uses a different prompt_version → separate cache entries.
# Fully resumable — kill and restart anytime.
#
# After all passes complete, merge results into a single enriched
# judged_bundles JSON with consensus features per candidate.
#
# Usage:
#   export V11_API_KEY=sk-...
#   bash scripts/run_multipass_judge.sh
#
# Cost: ~$50 total for all 5 passes on 1,139 queries.

set -uo pipefail

REPO="/Users/william/swiss-legal-retrieval"
cd "$REPO"

source "$REPO/.env" 2>/dev/null || true

PYTHON="$REPO/.venv/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO/logs"
LOG_FILE="$LOG_DIR/multipass_judge_${TS}.log"
STATUS_FILE="$LOG_DIR/multipass_judge_status.json"
LATEST_LINK="$LOG_DIR/multipass_judge_latest.log"
mkdir -p "$LOG_DIR"
ln -sf "$LOG_FILE" "$LATEST_LINK"

# --- Validate ---
if [[ -z "${V11_API_KEY:-}" ]]; then
  echo "ERROR: V11_API_KEY must be set." >&2
  exit 1
fi

# --- Common config ---
export V11_BASE_URL="${V11_BASE_URL:-https://api.deepseek.com/v1}"
export V11_JUDGE_MODEL="${V11_JUDGE_MODEL:-deepseek-chat}"
export V11_USE_MAX_TOKENS=1
export V11_MAX_TOKENS=4000
export V11_JUDGE_MAX_ATTEMPTS=3
export V11_USE_COURT_DENSE=1
export V11_JUDGE_WORKERS=32
export V11_LAW_BATCH_SIZE=10
export V11_COURT_BATCH_SIZE=6

CHUNK_SIZE=100

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

write_status() {
  cat > "$STATUS_FILE" <<EOF
{
  "pass": "$1",
  "stage": "$2",
  "detail": "$3",
  "log_file": "$LOG_FILE",
  "updated": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF
}

TOTAL_TRAIN=1139
TOTAL_CHUNKS=$(( (TOTAL_TRAIN + CHUNK_SIZE - 1) / CHUNK_SIZE ))

run_judge_pass() {
  local pass_name="$1"
  local prompt_version="$2"
  local law_topk="$3"
  local court_topk="$4"
  local prompt_variant="$5"
  local court_dense_topk="${6:-160}"

  export V11_PROMPT_VERSION="$prompt_version"
  export V11_LAW_JUDGE_TOPK="$law_topk"
  export V11_COURT_JUDGE_TOPK="$court_topk"
  export V11_JUDGE_PROMPT_VARIANT="$prompt_variant"
  export V11_COURT_DENSE_TOPK="$court_dense_topk"

  log ""
  log "=== PASS: $pass_name (prompt=$prompt_version, law_k=$law_topk, court_k=$court_topk, variant=$prompt_variant) ==="

  for chunk_idx in $(seq 0 $((TOTAL_CHUNKS - 1))); do
    local offset=$((chunk_idx * CHUNK_SIZE))
    local remaining=$((TOTAL_TRAIN - offset))
    local limit=$((remaining < CHUNK_SIZE ? remaining : CHUNK_SIZE))

    export V11_QUERY_OFFSET=$offset
    export V11_MAX_QUERIES=$limit

    local artifact_dir="artifacts/v11/train_${prompt_version}__offset${offset}_n${limit}"

    if [[ -f "$artifact_dir/judged_bundles.json" ]]; then
      log "  $pass_name chunk $((chunk_idx+1))/$TOTAL_CHUNKS: SKIP (exists)"
      continue
    fi

    write_status "$pass_name" "build_judge" "chunk $((chunk_idx+1))/$TOTAL_CHUNKS (offset=$offset)"
    log "  $pass_name chunk $((chunk_idx+1))/$TOTAL_CHUNKS: build+judge (offset=$offset, n=$limit)..."

    local chunk_start=$(date +%s)
    $PYTHON run_v11_staged.py full --split train \
      --offset "$offset" --limit "$limit" \
      --output "submissions/train_pred_${pass_name}_chunk${chunk_idx}.csv" \
      >> "$LOG_FILE" 2>&1

    local rc=$?
    local chunk_end=$(date +%s)
    local elapsed=$((chunk_end - chunk_start))

    if [[ $rc -ne 0 ]]; then
      log "  $pass_name chunk $((chunk_idx+1)): FAILED (rc=$rc) after ${elapsed}s"
      log "  Safe to restart — cached work preserved."
      return 1
    fi
    log "  $pass_name chunk $((chunk_idx+1)): DONE in ${elapsed}s"
  done

  log "=== PASS $pass_name COMPLETE ==="
  return 0
}

log "=== MULTI-PASS JUDGE: 5 passes on $TOTAL_TRAIN train queries ==="
log "Estimated cost: ~\$50 total"
log ""

# Pass 1: Default (may already be done from run_full_train_judge.sh)
run_judge_pass "default" "v11_strict_v1" 60 36 "default"

# Pass 2: Enriched prompt (snippet-aware — leans on evidence text)
run_judge_pass "enriched" "v11_multipass_enriched" 60 36 "enriched"

# Pass 3: Strict prompt (higher bar — only confident must_include)
run_judge_pass "strict" "v11_multipass_strict" 60 36 "strict"

log ""
log "========================================="
log "ALL 3 PASSES COMPLETE"
log "========================================="

# Merge into consensus features
log "Merging passes into consensus features..."
$PYTHON scripts/merge_multipass.py \
  --pass default=artifacts/v11 \
  --pass enriched=artifacts/v11 \
  --pass strict=artifacts/v11 \
  --gold data/train.csv \
  --output artifacts/v11_multipass_merged \
  >> "$LOG_FILE" 2>&1

log ""
log "Done. Merged artifact: artifacts/v11_multipass_merged/judged_bundles.json"
log "Run Rust CV sweep on merged data for consensus-based selection."
write_status "DONE" "complete" "3 passes + merge"
