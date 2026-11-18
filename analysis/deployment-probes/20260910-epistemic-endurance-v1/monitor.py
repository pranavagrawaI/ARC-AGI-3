"""Read-only terminal monitor; never launches or retries a kernel."""
from pathlib import Path
import datetime
import json
import subprocess
import time

HERE=Path(__file__).resolve().parent
KERNEL='pranavagrawai/arc3-first-submission-20260910'
started=time.monotonic()
while time.monotonic()-started < 3900:
    record={'at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    for label,args in [('status',['status',KERNEL]),('logs',['logs',KERNEL])]:
        try:
            r=subprocess.run(['kaggle','kernels',*args],capture_output=True,text=True,timeout=30)
            record[label+'_returncode']=r.returncode
            if label=='status': record['status']=r.stdout.strip() or r.stderr.strip()
            else:
                (HERE/'live-log.txt').write_text(r.stdout+r.stderr)
                record['log_lines']=len(r.stdout.splitlines())
                record['tail']=r.stdout.splitlines()[-5:]
        except subprocess.TimeoutExpired:
            record[label+'_error']='read timed out'
    with (HERE/'monitor.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
    if any(token in record.get('status','') for token in ('COMPLETE','ERROR','CANCELLED')):
        break
    time.sleep(45)
else:
    print('Monitoring deadline reached; inspect status without launching again.',flush=True)
