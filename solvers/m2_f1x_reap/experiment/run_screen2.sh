#!/bin/bash
# One-shot entry point for screen-2 c16: launch once, collect, analyze, queue on PASS. Idempotent; safe to re-run.
set -u
cd "$(dirname "$0")"
KPY=/home/pranav/.local/share/uv/tools/kaggle/bin/python
exec 9>/tmp/arc-gpu-launch.lock
flock 9
if [ ! -e screen-2/c16/launch-attempt.json ]; then
  "$KPY" -B launch.py screen-2 || exit 1
fi
sleep 90
flock -u 9
[ -e screen-2/c16/launch-response.json ] || { echo "not launched (admission refused?)"; exit 1; }
[ -e screen-2/c16/output/benchmark.json ] || "$KPY" -B collect.py screen-2/c16 || exit 1
[ -e screen-2/RESULT.json ] || python3 -B analyze_c16.py || exit 1
[ -e screen-2/QUEUED ] || { python3 -B queue_c16.py && touch screen-2/QUEUED; }
