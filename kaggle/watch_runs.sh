#!/usr/bin/env bash
# Poll both reference-arm kernels until each reaches a terminal state, then pull
# its version-addressed output before anything is interpreted.
#
#   ./watch_runs.sh            # foreground
#   setsid ./watch_runs.sh &   # detached; tail runs/watch.log
#
# Status comes from kernel_status.py, not `kaggle kernels status`: the CLI
# reported QUEUED for 2.5 h on kernels that were running (GPU quota was accruing
# and the REST endpoint said `running`), so its string cannot be trusted here.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/runs"
LOG="$OUT/watch.log"
INTERVAL="${INTERVAL:-180}"
ARMS=("nooa-upstream:prototype-nooa-upstream-public3" "tycho-upstream:prototype-tycho-upstream-public3")

mkdir -p "$OUT"
say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

declare -A done_arm
say "watching ${#ARMS[@]} kernels every ${INTERVAL}s (REST status)"

while true; do
  remaining=0
  for entry in "${ARMS[@]}"; do
    arm="${entry%%:*}"; slug="${entry##*:}"
    [[ -n "${done_arm[$arm]:-}" ]] && continue
    line="$(python3 "$HERE/kernel_status.py" "$slug" 2>&1 | head -1)"
    state="$(awk '{print $2}' <<<"$line")"
    gpu="$(kaggle quota 2>/dev/null | awk '/^GPU/{print $2}')"
    say "$arm $state (gpu used ${gpu:-?})${line#*"$state"}"
    case "$state" in
      complete|error|cancelAcknowledged|cancelRequested)
        done_arm[$arm]=1
        say "$arm terminal ($state) — pulling output"
        mkdir -p "$OUT/$arm"
        kaggle kernels output "pranavagrawai/$slug" -p "$OUT/$arm" >>"$LOG" 2>&1
        say "$arm output in $OUT/$arm"
        ;;
      *) remaining=$((remaining + 1)) ;;
    esac
  done
  [[ $remaining -eq 0 ]] && break
  sleep "$INTERVAL"
done

say "both kernels terminal"
for entry in "${ARMS[@]}"; do
  arm="${entry%%:*}"
  result="$OUT/$arm/$arm/result.json"
  [[ -f "$result" ]] && say "$arm result.json present ($(wc -c <"$result") bytes)"
done
