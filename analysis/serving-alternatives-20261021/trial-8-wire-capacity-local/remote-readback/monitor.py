"""Bounded read-only status monitor. No launch, cancellation or auth repair."""
import contextlib,datetime as dt,io,json,runpy,time
from pathlib import Path
P=Path(__file__).resolve().parent
attempt=json.loads((P.parent/'launch-attempt.json').read_text())
deadline=dt.datetime.fromisoformat(attempt['utc'])+dt.timedelta(seconds=10500)
while dt.datetime.now(dt.timezone.utc)<deadline:
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(P/'poll.py'),run_name='__main__')
    receipt=json.loads((P/'latest-poll.json').read_text())
    status=receipt.get('status',{}).get('status')
    print(json.dumps({k:receipt.get(k) for k in ('utc','status','quota','status_error','quota_error','log_bytes')}),flush=True)
    if status in ('COMPLETE','ERROR','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED'):
        break
    if receipt.get('status_error') or receipt.get('quota_error'):
        print('READ_ONLY_MONITOR_STOPPED_ON_API_ERROR',flush=True);break
    time.sleep(min(60,max(0,(deadline-dt.datetime.now(dt.timezone.utc)).total_seconds())))
else:
    print('READ_ONLY_MONITOR_DEADLINE_REACHED; platform cap remains mandatory',flush=True)
