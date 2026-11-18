#!/usr/bin/env bash
# Print only DONE / FAILED. Diagnostics to watch_q6.log keyed on pid.
set -euo pipefail
PID=${1:?pid}
PROBE=${2:?probe dir}
LOG="$PROBE/watch_q6_${PID}.log"
log() { printf '%s %s\n' "$(date -Iseconds)" "$*" >>"$LOG"; }

alive() { kill -0 "$PID" 2>/dev/null; }
summary() { ls -1 "$PROBE"/results/*/summary.json 2>/dev/null | tail -1; }

log "watch start pid=$PID"
while :; do
  s=$(summary || true)
  if [ -n "${s:-}" ]; then
    log "summary $s"
    echo "DONE: $s"
    exit 0
  fi
  if ! alive; then
    err=$(tail -c 400 "$PROBE/q6.err" 2>/dev/null | tr '\n' ' ')
    log "pid dead err=$err"
    echo "FAILED: pid $PID exited without summary.json: $err"
    exit 1
  fi
  sleep 30
done
