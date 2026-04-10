#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <query_ids_file> [limit]" >&2
  exit 1
fi

QUERY_IDS_FILE="$1"
LIMIT="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ ! -f "$QUERY_IDS_FILE" ]]; then
  echo "missing query id file: $QUERY_IDS_FILE" >&2
  exit 1
fi

if [[ -z "${LLM_API_KEY:-}" ]]; then
  echo "LLM_API_KEY must be set" >&2
  exit 1
fi

export LLM_BASE_URL="${LLM_BASE_URL:-https://api.deepseek.com/v1}"
export MAX_WORKERS="${MAX_WORKERS:-4}"
export V11_JUDGE_WORKERS="${V11_JUDGE_WORKERS:-8}"
export QUERY_EXPANSIONS_MODEL="${QUERY_EXPANSIONS_MODEL:-deepseek-reasoner}"
export DOMAIN_TEMPLATES_MODEL="${DOMAIN_TEMPLATES_MODEL:-deepseek-reasoner}"
export CASE_CITATIONS_MODEL="${CASE_CITATIONS_MODEL:-deepseek-reasoner}"
export FULL_CITATIONS_MODEL="${FULL_CITATIONS_MODEL:-deepseek-reasoner}"
export LLM_USE_MAX_TOKENS="${LLM_USE_MAX_TOKENS:-1}"

export V11_QUERY_IDS_PATH="$(cd "$(dirname "$QUERY_IDS_FILE")" && pwd)/$(basename "$QUERY_IDS_FILE")"
export V11_USE_COURT_DENSE="${V11_USE_COURT_DENSE:-0}"
export V11_JUDGE_MODEL="${V11_JUDGE_MODEL:-deepseek-reasoner}"
export V11_USE_MAX_TOKENS="${V11_USE_MAX_TOKENS:-1}"
export V11_MAX_TOKENS="${V11_MAX_TOKENS:-8000}"
export V11_API_KEY="${V11_API_KEY:-$LLM_API_KEY}"
export V11_BASE_URL="${V11_BASE_URL:-$LLM_BASE_URL}"
export V11_PROMPT_VERSION="${V11_PROMPT_VERSION:-v11_strict_v1_deepseek_reasoner_dense}"

common_args=(train --query-ids "$QUERY_IDS_FILE")
stage_args=(--split train)

if [[ -n "$LIMIT" ]]; then
  common_args+=(--limit "$LIMIT")
  stage_args+=(--limit "$LIMIT")
fi

echo "[$(date)] starting DeepSeek dense pilot"
echo "query_ids=$QUERY_IDS_FILE limit=${LIMIT:-all}"
echo "models: expansions=$QUERY_EXPANSIONS_MODEL case=$CASE_CITATIONS_MODEL full=$FULL_CITATIONS_MODEL judge=$V11_JUDGE_MODEL"
echo "parallelism: precompute_workers=$MAX_WORKERS judge_workers=$V11_JUDGE_WORKERS"

"$PYTHON_BIN" precompute/gen_query_expansions.py "${common_args[@]}" --max-workers "$MAX_WORKERS" --skip-domain-templates
"$PYTHON_BIN" precompute/gen_case_citations.py "${common_args[@]}" --max-workers "$MAX_WORKERS"
"$PYTHON_BIN" precompute/gen_full_citations_v2.py "${common_args[@]}" --max-workers "$MAX_WORKERS"
"$PYTHON_BIN" run_v11_staged.py build "${stage_args[@]}"
"$PYTHON_BIN" run_v11_staged.py judge "${stage_args[@]}"

echo "[$(date)] DeepSeek dense pilot finished"
