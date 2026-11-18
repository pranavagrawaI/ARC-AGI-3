#!/bin/bash
# Poll the smoke until terminal, then download its output once.
cd "$(dirname "$0")"; K=pranavagrawai/m2-f1x-keepalive; KG=~/.local/bin/kaggle; [ -x $KG ] || KG=kaggle
while true; do
  s=$(timeout 60 $KG kernels status $K 2>&1 | grep -v outdated | tail -1)
  echo "$(date -u +%FT%TZ) $s" >> smoke-status.log
  case "$s" in *COMPLETE*|*ERROR*|*CANCEL*|*FAILED*) break;; esac
  sleep 120
done
mkdir -p smoke-output && timeout 1200 $KG kernels output $K -p smoke-output > smoke-output.log 2>&1
echo "TERMINAL $s"
