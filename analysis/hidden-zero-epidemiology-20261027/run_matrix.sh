#!/bin/bash
# 2x2 at 1/10 time scale: {stock control, control+keepalive} x {slow load 240 s (=40 min), fast load 20 s}.
# Scaled clocks: startup release 72 s (=12 min), gateway idle close 90 s (=15 min), first-request grace 90 s
# (=900 s), keepalive 36 s (=6 min), hold max 900 s (=150 min); per-game cap shortened to 600 s.
cd "$(dirname "$0")"
PY=/tmp/f1x-faults/venv/bin/python
ENV=/home/pranav/projects/arc-agi-3-reference/competition-env/environment_files
OUT=${OUT:-/tmp/hz/runs}; mkdir -p "$OUT"
COMMON=(--llm process --startup-timeout 72 --idle-close 90 --per-game-s 600 ARC3_HTTP_RETRY_INITIAL_SECONDS=90 ARC3_KEEPALIVE_S=36 ARC3_HOLD_MAX_S=900)
run() { # name notebook delay port
  $PY -B run_competition_idle.py "$2" /tmp/hz/replica-control $ENV "$OUT/$1" "${COMMON[@]}" --server-args "--ready-delay $3" --port $4 > "$OUT/$1.log" 2>&1
  echo "$1 exit $?" >> "$OUT/done.txt"
}
P=../m2-winners-20261001/runs/raw/dfranzen-m2-raw.ipynb; K=control-keepalive.ipynb
run stock-slow $P 240 18101 & run ka-slow $K 240 18102 & run stock-fast $P 20 18103 & run ka-fast $K 20 18104 &
wait
