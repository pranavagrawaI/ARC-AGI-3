#!/bin/bash
# One-shot entry point for screen-1: launch once, collect, analyze, queue on PASS. Idempotent; safe to re-run.
set -u
cd "$(dirname "$0")"
KPY=/home/pranav/.local/share/uv/tools/kaggle/bin/python
# One launcher at a time account-wide, so two launchers can't both pass the 2-session slot check.
exec 9>/tmp/arc-gpu-launch.lock
flock 9
if [ ! -e screen-1/merge/launch-attempt.json ]; then
  "$KPY" -B launch.py screen-1 || exit 1
fi
sleep 90  # let the new session show up in the next launcher's slot check
flock -u 9
[ -e screen-1/merge/launch-response.json ] || { echo "not launched (admission refused?)"; exit 1; }
[ -e screen-1/merge/output/benchmark.json ] || "$KPY" -B collect.py screen-1/merge || exit 1
[ -e screen-1/RESULT.json ] || python3 -B analyze.py screen-1 || exit 1
[ -e screen-1/QUEUED ] || { python3 -B queue_candidate.py && touch screen-1/QUEUED; }
