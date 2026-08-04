#!/usr/bin/env bash
set -euo pipefail

NOOA_REPO=${NOOA_REPO:-/tmp/labs-OO-Agents}
PROJECT_ROOT=${PROJECT_ROOT:-/home/hipra/projects/arc-agi-3}
GAME=${1:?usage: run-gemma.sh GAME [extra run_solver.py arguments...]}
shift

set -a
source "$PROJECT_ROOT/.env"
set +a

export ARC_DATA_DIR="$PROJECT_ROOT"
export ARC_LLM_MODEL=google/gemma-4-31b-it
export ARC_LLM_BASE_URL=https://openrouter.ai/api/v1
export ARC_LLM_API_KEY="$OPENROUTER_API_KEY"
export OPENROUTER_PROVIDER_ONLY=deepinfra,chutes,together
export OPENROUTER_REASONING=on

cd "$NOOA_REPO"
exec uv run python examples/arc_agi_3/run_solver.py \
  --game "$GAME" \
  --variant memory \
  --operation-mode offline \
  --sandbox inproc \
  --max-actions-per-turn 2 \
  --agent-turn-timeout 700 \
  --nudge-after 600 \
  --fallback-window 30 \
  --visual additive \
  --png-scale 8 \
  --skill interactive-game-solver \
  --results-root "$PROJECT_ROOT/experiments/probes/nooa-gemma-reproduction/results" \
  --group reproduced \
  --tag gemma-pinned \
  "$@"
