#!/bin/sh
# Restart candidate2 collection if the watcher died (e.g. DNS) before writing a candidate2 RESULT.
# Safe to repeat: the launch marker exists, so execute.py only collects and evaluates.
cd /home/pranav/projects/arc-agi-3/analysis/pruning-transfer-20261011 || exit 1
grep -q '"candidate_run": "candidate2"' RESULT.json 2>/dev/null && exit 0
systemctl --user is-active -q arc-reap-candidate2-20261012.service && exit 0
systemctl --user is-active -q arc-reap-collect-20261012.service && exit 0
[ -f execution-stop.json ] && mv execution-stop.json "execution-stop-$(date -u +%Y%m%dT%H%M%S).json"
systemctl --user reset-failed arc-reap-collect-20261012.service 2>/dev/null
exec systemd-run --user --unit=arc-reap-collect-20261012 --working-directory="$PWD" \
  --property=StandardOutput=append:"$PWD/execution.log" --property=StandardError=append:"$PWD/execution.log" \
  /home/pranav/.local/share/uv/tools/kaggle/bin/python -B execute.py --rerun candidate2 --run
