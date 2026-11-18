"""One future accepted-run read-only monitor. No scheduling/restart/auth repair."""
import datetime as dt,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'collection_tools')]
from identity import HERE,accepted

def main():
 ref,kid,version=accepted();root=HERE/'remote-readback';root.mkdir(exist_ok=True)
 attempt=json.loads((HERE/'launch-attempt.json').read_bytes());end=dt.datetime.fromisoformat(attempt['utc'])+dt.timedelta(seconds=10800)
 with (root/'monitor-once.json').open('x') as stream:json.dump(dict(kernel_ref=ref,kernel_id=kid,version=version,absolute_deadline=end.isoformat()),stream)
 reason='DEADLINE_UNRESOLVED';last=None
 try:
  while True:
   remaining=(end-dt.datetime.now(dt.timezone.utc)).total_seconds()
   if remaining<=0:break
   directory=root/dt.datetime.now(dt.timezone.utc).strftime('poll-%Y%m%dT%H%M%S%fZ');last=str(directory)
   try:
    process=subprocess.run([sys.executable,str(ROOT/'readback.py'),str(directory)],timeout=min(120,remaining),capture_output=True,text=True)
   except subprocess.TimeoutExpired as error:
    directory.mkdir(exist_ok=True)
    (directory/'process.stdout').write_text(error.stdout.decode(errors='replace') if isinstance(error.stdout,bytes) else error.stdout or '')
    (directory/'process.stderr').write_text(error.stderr.decode(errors='replace') if isinstance(error.stderr,bytes) else error.stderr or '')
    reason='DEADLINE_UNRESOLVED' if dt.datetime.now(dt.timezone.utc)>=end else 'MONITOR_READBACK_TIMEOUT';break
   (directory/'process.stdout').write_text(process.stdout);(directory/'process.stderr').write_text(process.stderr)
   if process.returncode:reason='API_QUOTA_SOURCE_ERROR';break
   if dt.datetime.now(dt.timezone.utc)>end:reason='DEADLINE_UNRESOLVED';break
   receipt=json.loads((directory/'READBACK.json').read_bytes());(root/'latest-poll.json').write_text(json.dumps(dict(directory=str(directory),**receipt)))
   state=str(receipt['status'].get('status','')).upper()
   if state in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED','CANCELLED','CANCELED'):reason='TERMINAL';break
   time.sleep(min(60,max(0,(end-dt.datetime.now(dt.timezone.utc)).total_seconds())))
 except BaseException as error:
  reason='MONITOR_FAILURE'
  with (root/'monitor-failure.json').open('x') as stream:json.dump(dict(type=type(error).__name__,last_poll=last),stream)
 finally:
  with (root/'monitor-stop.json').open('x') as stream:json.dump(dict(reason=reason,kernel_ref=ref,kernel_id=kid,version=version,absolute_deadline=end.isoformat(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),last_poll=last,restart_authorized=False),stream,indent=2)
 return 0 if reason in ('TERMINAL','DEADLINE_UNRESOLVED') else 1
if __name__=='__main__':raise SystemExit(main())
