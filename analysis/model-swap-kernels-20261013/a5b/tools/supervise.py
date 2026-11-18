"""Read-only monitoring, then terminal collection/readout; never pushes or submits."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def main():
    deadline=time.monotonic()+10800
    while time.monotonic()<deadline:
        r=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve().parent/'poll.py')],capture_output=True,text=True)
        print(r.stdout.strip() or r.stderr[-1000:],flush=True)
        if r.returncode:
            if 'AssertionError' in r.stderr:raise RuntimeError('identity mismatch: stop monitoring, reconcile read-only')
        else:
            status=json.loads((ROOT/'a5b/status.json').read_text())['status']
            if status in ['COMPLETE','ERROR','CANCELED','CANCELLED']:
                subprocess.run([sys.executable,'-B',str(Path(__file__).resolve().parent/'collect.py')],check=True)
                subprocess.run(['/usr/bin/python3','-B',str(ROOT/'a5b/readout.py')],check=True)
                return
        time.sleep(45)
    raise RuntimeError('monitor window elapsed; no retry or efficacy claim')

if __name__=='__main__':main()
