"""Future one-owner read-only monitor; never installed or scheduled by preparation."""
import datetime as dt,json,subprocess,sys,time
from pathlib import Path
from identity import HERE,accepted

def main():
    ref,kid,version=accepted();root=HERE/'remote-readback';root.mkdir(exist_ok=True)
    with (root/'monitor-once.json').open('x') as stream:json.dump(dict(kernel_ref=ref,kernel_id=kid,version=version),stream)
    attempt=json.loads((HERE/'launch-attempt.json').read_text());end=dt.datetime.fromisoformat(attempt['utc'])+dt.timedelta(seconds=10800)
    while dt.datetime.now(dt.timezone.utc)<end:
        directory=root/dt.datetime.now(dt.timezone.utc).strftime('poll-%Y%m%dT%H%M%S%fZ')
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('readback.py')),str(directory)],timeout=120,capture_output=True,text=True)
        (directory/'process.stdout').write_text(result.stdout);(directory/'process.stderr').write_text(result.stderr)
        if result.returncode:break
        receipt=json.loads((directory/'READBACK.json').read_text());(root/'latest-poll.json').write_text(json.dumps(dict(directory=str(directory),**receipt)))
        state=str(receipt['status'].get('status','')).upper()
        if any(s in state for s in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED','CANCELLED','CANCELED')):break
        time.sleep(min(60,max(0,(end-dt.datetime.now(dt.timezone.utc)).total_seconds())))
if __name__=='__main__':main()
