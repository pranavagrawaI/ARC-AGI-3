#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/hipra/projects/arc-agi-3-tycho-nooa-refactor"
OUT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
set -a; source .env; set +a
export ARC_LLM_MODEL=deepseek/deepseek-v4-flash-0731 ARC_LLM_CONTEXT_WINDOW=1048576 ARC_LLM_MAX_TOKENS=4096 ARC_LLM_USE_RESPONSES=0
export TYCHO_NOOA_REQUEST_REASONING=1 TYCHO_NOOA_CAPTURE_RESPONSE_TEXT=1 ARC_LLM_PROVIDER_ENDPOINT=sail-research/fp4 TYCHO_SANDBOX_RUNTIME=docker
uv run --frozen --project solvers/tycho_nooa python "$OUT/run.py" "$@"
