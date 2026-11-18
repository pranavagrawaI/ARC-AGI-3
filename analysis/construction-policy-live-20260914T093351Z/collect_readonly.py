#!/usr/bin/env python3
"""Read-only exact-version monitor, source/metadata/output retrieval, and sanitization."""
from datetime import datetime,timezone
import fnmatch,hashlib,json,shutil,subprocess,tempfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; PACKAGE=REPO/'deployment/kaggle'/HERE.name; E=HERE/'evidence'; CLI='/home/pranav/.local/bin/kaggle'; TERMINAL={'COMPLETE','ERROR','CANCELLED'}
def now(): return datetime.now(timezone.utc).isoformat()
def save(p,v): p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def call(args,timeout=300):
    p=subprocess.run([CLI,*args],stdin=subprocess.DEVNULL,close_fds=True,capture_output=True,text=True,timeout=timeout)
    r={'arguments':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'utc':now()}; return r
def status_name(r):
    text=r['stdout'].strip()
    try: value=json.loads(text); text=str(value.get('status') or value.get('kernelWorkerStatus') or text)
    except Exception: pass
    text=text.upper()
    return next((s for s in TERMINAL|{'QUEUED','RUNNING'} if s in text),None)
def allowed(rel,patterns): return any(fnmatch.fnmatch(rel,p) or (p.endswith('/**') and rel.startswith(p[:-3]+'/')) for p in patterns)
def main():
    launch=json.loads((E/'LAUNCH-RESULT.json').read_text()); ref=f"{launch['returned_identity']}/{launch['version']}"
    deadline=time.monotonic()+4200; observations=[]
    while True:
        r=call(['kernels','status',ref]); observations.append(r); save(E/'status-observations.json',observations)
        if r['returncode']!=0: raise RuntimeError('EXACT_STATUS_FAILED')
        status=status_name(r)
        if status in TERMINAL: break
        if status not in {'QUEUED','RUNNING'} or time.monotonic()>=deadline: raise RuntimeError('MONITOR_STOPPED_WITHOUT_TERMINAL_STATUS')
        time.sleep(30)
    save(E/'status-exact-final.json',{'ref':ref,'status':status,'receipt':r})
    readback=E/'server-readback'; readback.mkdir(exist_ok=False)
    r=call(['kernels','pull',ref,'--metadata','-p',str(readback)]); save(E/'server-metadata-receipt.json',r)
    if r['returncode']!=0: raise RuntimeError('SERVER_METADATA_READBACK_FAILED')
    r=call(['kernels','pull',ref,'-p',str(readback)]); save(E/'server-source-receipt.json',r)
    if r['returncode']!=0: raise RuntimeError('SERVER_SOURCE_READBACK_FAILED')
    with tempfile.TemporaryDirectory(prefix='construction-live-output-') as tmp:
        r=call(['kernels','output',ref,'-p',tmp,'--force'],timeout=900); save(E/'output-retrieval-receipt.json',r)
        if r['returncode']!=0: raise RuntimeError('OUTPUT_RETRIEVAL_FAILED')
        policy=json.loads((HERE/'EVIDENCE-ALLOWLIST.json').read_text()); retained=[]; rejected=[]; remote=E/'remote'; remote.mkdir(exist_ok=False)
        for src in sorted(Path(tmp).rglob('*')):
            if not src.is_file(): continue
            rel=src.relative_to(tmp).as_posix(); lower=rel.lower()
            secret=any(fnmatch.fnmatch(lower,p.lower()) for p in policy['always_reject_without_reading'])
            if secret or not allowed(rel,policy['retained_paths']): rejected.append(rel); continue
            dst=remote/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dst)
            retained.append({'path':rel,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
        save(E/'RETRIEVAL.json',{'exact_ref':ref,'terminal_status':status,'retained':retained,'rejected_without_reading':rejected})
    q=call(['quota','--format','json']); save(E/'quota-after.json',q)
    if q['returncode']!=0: raise RuntimeError('FINAL_QUOTA_FAILED')
    print(json.dumps({'exact_ref':ref,'terminal_status':status,'retained':len(retained)},indent=2))
if __name__=='__main__': main()
