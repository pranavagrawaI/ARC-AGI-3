#!/usr/bin/env bash
# Scheduled K/KF push at weekly-quota reset (see CARD.md). Idempotent per arm:
# an arm with launched/<arm>.ok is never pushed again. Retries quota refusals
# every 15 min until RETRY_UNTIL (UTC epoch), then gives up.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p launched
LOG=launched/launch.log
RETRY_UNTIL=${RETRY_UNTIL:-$(( $(date -u +%s) + 4*3600 ))}  # relative: a late (post-sleep) start still retries
SDK_PY=/home/pranav/.local/share/uv/tools/kaggle/bin/python
exec >>"$LOG" 2>&1

while :; do
  echo "== $(date -u +%FT%TZ) attempt"
  "$SDK_PY" -c 'from kagglesdk import KaggleClient; from kagglesdk.kaggle_creds import KaggleCredentials as C; c=C.load(KaggleClient()); c.refresh_access_token(); print("token refreshed")' \
    || echo "token refresh FAILED"
  pending=0
  for arm in ${ARMS:-k2 kf2}; do
    [[ -f launched/$arm.ok ]] && continue
    out=$(kaggle kernels push -p "$arm/kernel" 2>&1); echo "-- $arm: $out"
    if grep -qi "successfully pushed" <<<"$out"; then
      date -u +%FT%TZ >"launched/$arm.ok"
    else
      pending=1
    fi
  done
  [[ $pending == 0 ]] && { echo "all arms pushed"; break; }
  (( $(date -u +%s) >= RETRY_UNTIL )) && { echo "giving up at RETRY_UNTIL"; break; }
  sleep 900
done

sleep 300
for arm in ${ARMS:-k2 kf2}; do echo "-- status $arm: $(kaggle kernels status pranavagrawai/duck-flash-next-$arm 2>&1)"; done
