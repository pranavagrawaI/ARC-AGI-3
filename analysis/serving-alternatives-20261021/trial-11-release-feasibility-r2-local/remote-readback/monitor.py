"""One bounded read-only monitor; no uploads, cancellation, retry of launch or auth repair."""
import datetime as dt,json,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
attempt=json.loads((P.parent/'launch-attempt.json').read_text());deadline=dt.datetime.fromisoformat(attempt['utc'])+dt.timedelta(seconds=4200)
with (P/'monitor-started.json').open('x') as f:json.dump(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),deadline=deadline.isoformat(),pid=__import__('os').getpid(),poll_seconds=60),f)
while dt.datetime.now(dt.timezone.utc)<deadline:
 started=time.monotonic()
 try:
  left=(deadline-dt.datetime.now(dt.timezone.utc)).total_seconds()
  done=subprocess.run([sys.executable,str(P/'poll.py')],capture_output=True,text=True,timeout=min(55,max(.1,left)))
  if done.returncode:
   print(done.stdout,done.stderr,flush=True);raise RuntimeError('read-only poll failed')
  receipt=json.loads((P/'latest-poll.json').read_text())
  print(json.dumps({k:receipt.get(k) for k in ('utc','status','quota','status_error','quota_error','output_error','log_bytes')}),flush=True)
  status=receipt.get('status',{}).get('status')
  if status in ('COMPLETE','ERROR','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED'):
   reason='TERMINAL';break
  if receipt.get('status_error') or receipt.get('quota_error'):
   reason='API_BLOCKER';break
 except Exception as error:
  reason='MONITOR_ERROR';print(repr(error),flush=True);break
 time.sleep(min(max(0,60-(time.monotonic()-started)),max(0,(deadline-dt.datetime.now(dt.timezone.utc)).total_seconds())))
else:reason='DEADLINE_UNRESOLVED'
with (P/'monitor-stopped.json').open('x') as f:json.dump(dict(reason=reason,utc=dt.datetime.now(dt.timezone.utc).isoformat(),deadline=deadline.isoformat()),f,indent=2)
print(reason,flush=True)
