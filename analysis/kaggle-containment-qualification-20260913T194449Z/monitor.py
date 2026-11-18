"""Read-only monitor; never push, cancel, delete, or infer remote shutdown."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time

HERE=Path(__file__).resolve().parent
IDENTITY='pranavagrawai/containment-q-20260913t194449z'
TERMINAL={'COMPLETE','ERROR','CANCELLED','CANCELED'}

def main():
    launched=json.loads((HERE/'LAUNCH-ATTEMPT.json').read_text())
    assert launched['kernel']==IDENTITY
    index=0
    while True:
        process=subprocess.run(['/home/pranav/.local/bin/kaggle','kernels','status',IDENTITY+'/1'],stdin=subprocess.DEVNULL,
                               capture_output=True,text=True,timeout=60)
        receipt={'utc':datetime.now(timezone.utc).isoformat(),'returncode':process.returncode,
                 'stdout':process.stdout,'stderr':process.stderr,'since_launch_s':time.time()-launched['epoch']}
        match=re.search(r'KernelWorkerStatus\.([A-Z_]+)',process.stdout)
        status=match.group(1) if process.returncode==0 and match else 'UNKNOWN'
        receipt['status']=status
        with (HERE/'monitor.jsonl').open('a') as f: f.write(json.dumps(receipt)+'\n')
        (HERE/'status-latest.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
        if status in TERMINAL:
            (HERE/'status-terminal.json').write_text(json.dumps(receipt,indent=2)+'\n')
            return 0
        if receipt['since_launch_s']>=600:
            print('LIFECYCLE_LIMIT: remote termination NOT verified; provider300s timeout requested; no cancel route',flush=True)
            return 2
        index+=1
        time.sleep(10)

if __name__=='__main__': raise SystemExit(main())
