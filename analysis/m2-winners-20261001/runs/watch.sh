#!/bin/bash
# poll both kernels; exit when raw reaches a terminal state
refresh() { /home/pranav/.local/share/uv/tools/kaggle/bin/python -c "from kagglesdk import KaggleClient; from kagglesdk.kaggle_creds import KaggleCredentials; KaggleCredentials.load(KaggleClient()).refresh_access_token()" >/dev/null 2>&1; }
last=""
for i in $(seq 1 120); do
  [ $((i % 60)) -eq 0 ] && refresh
  s=""
  for k in raw control; do s="$s $k=$(kaggle kernels status pranavagrawai/dfranzen-m2-$k 2>&1 | grep -o 'status "[A-Z_]*"\|KernelWorkerStatus\.[A-Z_]*' | tail -1)"; done
  [ "$s" != "$last" ] && echo "$(date -u +%H:%M:%S)$s" && last="$s"
  echo "$s" | grep -q "raw=.*\(COMPLETE\|ERROR\|CANCEL\)" && exit 0
  sleep 120
done
