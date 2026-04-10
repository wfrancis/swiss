#!/usr/bin/env bash
#
# One-shot status summary for the overnight court-dense sweep.
#
# Prints:
#   - last status snapshot
#   - running process check
#   - disk state of each expected artifact
#   - tail of the log file

set -u

REPO="/Users/william/swiss-legal-retrieval"
LOG_DIR="$REPO/logs"
STATUS_FILE="$LOG_DIR/overnight_courtdense_ds_status.json"

echo "=== Overnight court-dense sweep status ==="
echo "Now: $(date '+%Y-%m-%d %H:%M:%S')"

if [[ -f "$STATUS_FILE" ]]; then
  echo ""
  echo "Status snapshot:"
  cat "$STATUS_FILE"
else
  echo ""
  echo "(no status file yet)"
fi

echo ""
echo "Running processes:"
pgrep -af "run_v11_staged.py|run_v11_meta_selector|run_overnight_courtdense_ds_sweep" || echo "  none"

echo ""
echo "Disk state:"
for path in \
  "$REPO/artifacts/v11/train_v11_trainfit_local333_courtdense_restart1/judged_bundles.json" \
  "$REPO/artifacts/v11/val_v11_courtdense_ds_val1/candidate_bundles.pkl" \
  "$REPO/artifacts/v11/val_v11_courtdense_ds_val1/judged_bundles.json" \
  "$REPO/artifacts/v11/test_v11_courtdense_ds_test1/candidate_bundles.pkl" \
  "$REPO/artifacts/v11/test_v11_courtdense_ds_test1/judged_bundles.json" \
  "$REPO/artifacts/v11_meta/meta_selector_courtdense_ds.json" \
  "$REPO/submissions/val_pred_v11_courtdense_ds_sweep.csv"; do
  rel="${path#$REPO/}"
  if [[ -f "$path" ]]; then
    size=$(du -h "$path" | cut -f1)
    mtime=$(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$path" 2>/dev/null || stat -c '%y' "$path" 2>/dev/null)
    echo "  [OK]      $rel ($size, $mtime)"
  else
    echo "  [MISSING] $rel"
  fi
done

echo ""
echo "Judge cache counts:"
for split in train val test; do
  f="$REPO/precompute/judge_cache_${split}_v11.jsonl"
  if [[ -f "$f" ]]; then
    lines=$(wc -l < "$f" | tr -d ' ')
    mtime=$(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$f" 2>/dev/null || stat -c '%y' "$f" 2>/dev/null)
    echo "  $split: $lines lines (last mod $mtime)"
  else
    echo "  $split: (no cache)"
  fi
done

latest_log=$(ls -t "$LOG_DIR"/overnight_courtdense_ds_*.log 2>/dev/null | head -1)
if [[ -n "$latest_log" ]]; then
  echo ""
  echo "Latest log: $latest_log"
  echo "Last 30 lines:"
  tail -n 30 "$latest_log" | sed 's/^/  /'
fi
