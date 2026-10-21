"""Bounded read-only collection of the one launched trial. Never launches or retries it."""
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLI = '/home/pranav/.local/bin/kaggle'
REF = 'pranavagrawai/serving-vllm-smoke1-20261021'


def main():
    deadline = time.monotonic() + 6000
    while time.monotonic() < deadline:
        r = subprocess.run([CLI, 'kernels', 'status', REF], capture_output=True, text=True, timeout=40)
        row = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'rc': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
        with (HERE/'status-history.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        (HERE/'latest-status.json').write_text(json.dumps(row,indent=2)+'\n')
        if r.returncode == 0 and any(x in r.stdout for x in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED')):
            partial = HERE/'output.partial'; partial.mkdir(exist_ok=True)
            r = subprocess.run([CLI, 'kernels', 'output', REF, '-p', str(partial)], capture_output=True, text=True, timeout=240)
            (HERE/'collection.json').write_text(json.dumps({'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n')
            if r.returncode: raise RuntimeError('output download incomplete; preserved output.partial; no automatic retry')
            out=HERE/'output'
            assert not out.exists(), 'existing output preserved'
            partial.rename(out)
            hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
            (HERE/'output-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
            result=list(out.rglob('result.json'))
            summary={'terminal_status':row['stdout'].strip(),'output_files':len(hashes),'result_files':[str(p.relative_to(out)) for p in result]}
            if len(result)==1:
                data=json.loads(result[0].read_text())
                summary.update(trial_status=data.get('status'),error=data.get('error'),replay=data.get('replay'))
            (HERE/'collection-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(json.dumps(summary),flush=True)
            return
        time.sleep(60)
    raise TimeoutError('collector deadline; inspect preserved status receipts, no relaunch')


if __name__ == '__main__':
    main()
