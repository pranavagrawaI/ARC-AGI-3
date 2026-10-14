"""Read-only polling of the one accepted rerun; never launches or retries GPU work."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
START=time.monotonic()
while time.monotonic()-START<4500:
    result=subprocess.run([sys.executable,'-B',str(HERE/'tools.py'),'status'],capture_output=True,text=True,timeout=70)
    if result.returncode:
        print(dt.datetime.now(dt.timezone.utc).isoformat(),'read-only status unavailable',flush=True)
        (HERE/'evidence/watch-error.txt').write_text(result.stderr[-2000:])
    else:
        status=json.loads((HERE/'status.json').read_text())
        with (HERE/'evidence/status-history.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
        print(json.dumps(status),flush=True)
        if status['status'] in ['COMPLETE','ERROR','CANCELED','CANCELLED']:break
    time.sleep(45)
else:raise SystemExit('monitor walltime cap reached; no new GPU launch')
