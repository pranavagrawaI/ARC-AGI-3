#!/usr/bin/env bash
set -euo pipefail

NOOA_REPO=${NOOA_REPO:-/tmp/labs-OO-Agents}
PROJECT_ROOT=${PROJECT_ROOT:-/home/hipra/projects/arc-agi-3}
RESULTS_ROOT="$PROJECT_ROOT/experiments/probes/nooa-gemma-epistemic-context/results"
ARM=${1:?usage: run-arm.sh A|B|C}

case "$ARM" in
  A) TREATMENT=control ;;
  B) TREATMENT=evidence_only ;;
  C) TREATMENT=contradiction_first ;;
  *) echo "unknown arm: $ARM (expected A, B, or C)" >&2; exit 2 ;;
esac

set -a
source "$PROJECT_ROOT/.env"
set +a

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is required}"
export ARC_DATA_DIR="$PROJECT_ROOT"
export ARC_LLM_MODEL=google/gemma-4-31b-it
export ARC_LLM_BASE_URL=https://openrouter.ai/api/v1
export ARC_LLM_API_KEY="$OPENROUTER_API_KEY"
export OPENROUTER_PROVIDER_ONLY=modelrun/fp4
export OPENROUTER_REASONING=on
export ARC_EPISTEMIC_TREATMENT="$TREATMENT"

cd "$NOOA_REPO"
exec uv run python examples/arc_agi_3/run_solver.py \
  --game ls20 \
  --variant memory \
  --operation-mode offline \
  --sandbox inproc \
  --max-env-steps 44 \
  --max-actions-per-turn 20 \
  --allowed-game-overs 0 \
  --agent-turn-timeout 1200 \
  --nudge-after 900 \
  --fallback-window 86400 \
  --visual additive \
  --png-scale 8 \
  --skill interactive-game-solver \
  --results-root "$RESULTS_ROOT" \
  --group "arm-$ARM" \
  --tag "${TREATMENT}_gemma-modelrun-fp4_cap44"
