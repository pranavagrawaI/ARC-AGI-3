#!/bin/bash
# Timer entry point for screen-1: launch once after the weekly quota reset, collect, analyze. Idempotent.
set -u
cd "$(dirname "$0")"
KPY=/home/pranav/.local/share/uv/tools/kaggle/bin/python
# One launcher at a time account-wide: two screens admitted together would exceed the 2-session cap.
exec 9>/tmp/arc-gpu-launch.lock
flock 9
if [ ! -e screen-1/lex/launch-attempt.json ]; then
  "$KPY" -B launch.py screen-1 || exit 1
fi
sleep 90  # let the new sessions show up in the next launcher's slot check
flock -u 9
[ -e screen-1/lex/launch-response.json ] && [ -e screen-1/rawctl/launch-response.json ] || { echo "not launched (admission refused?)"; exit 1; }
for run in screen-1/lex screen-1/rawctl; do
  [ -e "$run/output/benchmark.json" ] || "$KPY" -B collect.py "$run" || exit 1
done
[ -e screen-1/RESULT.json ] || python3 -B analyze.py screen-1
