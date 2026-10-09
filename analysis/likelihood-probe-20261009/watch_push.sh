#!/usr/bin/env bash
# Push probe kernels A then B, each only after observing a freed GPU slot (no blind retries: a refused push may poison the id).
cd "$(dirname "$0")"
pending=(arc3-lprobe-a1 arc3-lprobe-b1)
active() { local n=0; for k in "$@"; do s=$(kaggle kernels status "$k" 2>&1 | tail -1); case "$s" in *RUNNING*|*QUEUED*) n=$((n+1));; esac; done; echo $n; }
while ((${#pending[@]})); do
  others=$(active pranavagrawai/m2-evidence-pair1-control pranavagrawai/m2-evidence-pair1-treatment)
  mine=0; for k in arc3-lprobe-a1 arc3-lprobe-b1; do [[ " ${pending[*]} " == *" $k "* ]] || mine=$((mine + $(active pranavagrawai/$k))); done
  free=$((2 - others - mine))
  echo "$(date -u +%FT%TZ) others=$others mine=$mine free=$free pending=${pending[*]}"
  if ((free > 0)); then
    k=${pending[0]}
    out=$(kaggle kernels push -p kernel-$k 2>&1 | tail -2); echo "$(date -u +%FT%TZ) PUSH $k: $out"
    pending=("${pending[@]:1}")
    sleep 60; continue
  fi
  sleep 120
done
echo "$(date -u +%FT%TZ) all pushed"
