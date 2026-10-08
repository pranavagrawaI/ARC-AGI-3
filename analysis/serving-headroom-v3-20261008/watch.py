"""Read-only paginated collector and audit. Contains no push/retry path."""
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
READOUT=HERE/'readout'
SLUG='pranavagrawai/serving-headroom-v3-20261008'


def now():return dt.datetime.now(dt.timezone.utc).isoformat()


def collect(api,status):
    import audit
    READOUT.mkdir(exist_ok=True)
    raw=READOUT/'raw';files=[];token=None;seen=set()
    while True:
        page,token=api.kernels_output(SLUG,str(raw),quiet=True,page_size=200,page_token=token)
        files.extend(page)
        if not token:break
        if token in seen:raise RuntimeError('repeated output pagination token')
        seen.add(token)
    receipt=dict(utc=now(),slug=SLUG,status=status,output_files=files,
         file_hashes={str(p.relative_to(raw)):hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.rglob('*') if p.is_file()})
    (READOUT/'collection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    result=audit.audit(raw)
    (READOUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    import sys
    sys.path.insert(0,str(HERE.parent/'mtp-depth-20261006'))
    import launch_watch
    send=requests.Session.send
    def bounded(session,request,**kw):
        kw['timeout']=(15,45);return send(session,request,**kw)
    requests.Session.send=bounded
    READOUT.mkdir(exist_ok=True);launch_watch.EVID=READOUT
    if (READOUT/'collection.json').exists() and (READOUT/'audit.json').exists():
        return
    deadline=time.time()+48*3600
    while time.time()<deadline:
        try:
            launch_watch.refresh_token()
            api=KaggleApi();api.authenticate()
            status=json.loads(str(api.kernels_status(SLUG)))
            row=dict(utc=now(),**status)
            with (READOUT/'status-history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            if status['status'] in ('COMPLETE','ERROR','CANCELED','CANCELLED'):
                print(json.dumps(collect(api,status)),flush=True);return
        except Exception as error:
            with (READOUT/'errors.jsonl').open('a') as f:f.write(json.dumps(dict(utc=now(),error=repr(error)))+'\n')
            print('read-only collection error',repr(error),flush=True)
        time.sleep(120)
    raise RuntimeError('48-hour collection deadline; no relaunch')

if __name__=='__main__':main()
