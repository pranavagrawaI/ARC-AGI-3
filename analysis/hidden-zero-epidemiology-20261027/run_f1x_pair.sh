#!/bin/bash
# F1x stock vs F1x+keepalive on the slow load only, same 1/10 scaled clocks as run_matrix.sh, F1x replica bundle.
cd "$(dirname "$0")"
PY=/tmp/f1x-faults/venv/bin/python
ENV=/home/pranav/projects/arc-agi-3-reference/competition-env/environment_files
OUT=${OUT:-/tmp/hz/runs-f1x}; mkdir -p "$OUT"
COMMON=(--llm process --startup-timeout 72 --idle-close 90 --per-game-s 600 ARC3_HTTP_RETRY_INITIAL_SECONDS=90 ARC3_KEEPALIVE_S=36 ARC3_HOLD_MAX_S=900)
run() { $PY -B run_competition_idle.py "$2" /tmp/f1x-faults/replica-f1x $ENV "$OUT/$1" "${COMMON[@]}" --server-args "--ready-delay $3" --port $4 > "$OUT/$1.log" 2>&1; echo "$1 exit $?" >> "$OUT/done.txt"; }
run f1x-stock-slow ../../solvers/m2_f1/experiment/screen-2/f1x/kernel/m2-f1-screen2-f1x.ipynb 240 18111 &
run f1x-ka-slow f1x-keepalive.ipynb 240 18112 &
wait
