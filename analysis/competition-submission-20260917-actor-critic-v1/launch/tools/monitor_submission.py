"""Read-only poll of competition submission ref until terminal (COMPLETE/ERROR). Never cancels, never resubmits."""
import json,subprocess,sys,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[2];CLI='/home/pranav/.local/bin/kaggle';REF=int(sys.argv[1]);MAX=int(sys.argv[2]);INTERVAL=int(sys.argv[3])
log=RUN/'launch'/f'monitor-submission-{REF}.jsonl';t0=time.time();errors=0;last=None
while True:
    r=subprocess.run([CLI,'competitions','submissions','arc-prize-2026-arc-agi-3','--format','json'],text=True,capture_output=True,timeout=120)
    rec=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),elapsed=round(time.time()-t0),returncode=r.returncode,stderr=r.stderr.strip()[-200:])
    status=None
    try:
        rows=json.loads(r.stdout);rec['row_count']=len(rows)
        row=[x for x in rows if x.get('ref')==REF];rec['row']=row[0] if row else None
        if len(rows)!=1:rec['unexpected_rows']=[x.get('ref') for x in rows]
        status=(row[0]['status'].split('.')[-1] if row else 'MISSING')
    except Exception as e:errors+=1;rec['parse_error']=str(e)
    rec['status']=status
    with log.open('a') as f:f.write(json.dumps(rec)+'\n')
    if status!=last:print(rec['utc'],status,json.dumps(rec.get('row')),flush=True);last=status
    if status in ('COMPLETE','ERROR','MISSING'):
        q=subprocess.run([CLI,'quota','--format','json'],text=True,capture_output=True,timeout=120)
        (RUN/'launch'/f'quota-after-submission-{REF}.json').write_text(q.stdout);print('TERMINAL',status,q.stdout.strip().replace('\n',' '),flush=True);break
    if errors>=20:print('READ-FAILURES',json.dumps(rec),flush=True);sys.exit(3)
    if time.time()-t0>MAX:print('MONITOR-BOUND',json.dumps(rec),flush=True);sys.exit(4)
    time.sleep(INTERVAL)
