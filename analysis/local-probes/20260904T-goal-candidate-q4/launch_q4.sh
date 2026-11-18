#!/usr/bin/env bash
# q4: host-owned goal-candidate record. See EXPERIMENT-CARD.md.
# Thinking stays OFF — same actor as q3c.
# Do not source this from fish; run with bash (via daemonize_q4.py).
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
  --agent-turn-timeout 600 \
  --max-turns 60 \
  --max-env-steps 200 \
  --visual off \
  --effort-ladder '' \
  --results-root "$PROBE/results" \
  --tag q4
