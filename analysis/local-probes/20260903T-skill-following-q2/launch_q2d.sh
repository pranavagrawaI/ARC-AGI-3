#!/usr/bin/env bash
# q2d: reasoning-off skill-following gate. See PRECOMMIT.md Addendum 4.
# Do not source this from fish; run with bash.
set -euo pipefail

PROBE="$(cd "$(dirname "$0")" && pwd)"
WORKTREE="$(cd "$PROBE/../../.." && pwd)"
NOOA="/home/pranav/projects/arc-agi-3-reference/nooa"
PY="$NOOA/.venv/bin/python"

set -a
# shellcheck disable=SC1091
source /home/pranav/projects/arc-agi-3/.env
set +a

export ARC_DATA_DIR="$WORKTREE"
export ARC_LLM_EXTRA_BODY='{"reasoning": {"enabled": false}}'
export PYTHONUNBUFFERED=1

cd "$NOOA"
exec "$PY" examples/arc_agi_3/run_solver.py \
  --game ls20 \
  --variant mdfiles \
  --operation-mode offline \
  --model qwen/qwen3.8-27b \
  --skill interactive-game-solver \
  --sandbox inproc \
  --sandbox-cells \
  --sandbox-require \
  --sandbox-cpu-s 60 \
  --agent-turn-timeout 300 \
  --max-turns 40 \
  --max-env-steps 120 \
  --visual off \
  --effort-ladder '' \
  --results-root "$PROBE/results" \
  --tag q2d
