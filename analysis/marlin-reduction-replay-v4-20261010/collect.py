"""Read-only collector: version/source guards, streamed pagination, raw audit, no push."""
import fcntl
import json
import os
import time
from pathlib import Path
from launch import AUDIT_PYTHON,HERE,SLUG,now,save,sha,verify_local

TERMINAL={'COMPLETE','ERROR','CANCELED','CANCELLED'}


def remote_guard(backend,receipt,root=HERE):
    root=Path(root);meta=json.loads((root/'kernel/kernel-metadata.json').read_text())
    notebook=json.loads((root/'kernel'/meta['code_file']).read_text())
    result=backend.verify_remote(SLUG,receipt,notebook,meta)
    source=result.pop('source')
    dest=root/'readout';dest.mkdir(exist_ok=True)
    save(dest/'remote-source.json',source);save(dest/'remote-source-verification.json',{'utc':now(),**result})
    return result


def collect_outputs(backend,receipt,status,root=HERE,auditor=None):
    root=Path(root);readout=root/'readout';raw=readout/'raw';raw.mkdir(parents=True,exist_ok=True)
    remote_guard(backend,receipt,root)
    files=[];tokens=set();token=None
    while True:
        page,next_token,log=backend.output_page(SLUG,token)
        for item in page:files.append(str(backend.download(item,raw).relative_to(raw)))
        if log:(raw/'kernel.log').write_text(log)
        if not next_token:break
        if next_token in tokens:raise ValueError('Output pagination repeated a token; collection incomplete')
        tokens.add(next_token);token=next_token
    # SDK status/output endpoints ignore a version suffix; a current-version guard
    # before AND after download prevents accidentally collecting a later run.
    remote_guard(backend,receipt,root)
    hashes={str(p.relative_to(raw)):sha(p) for p in raw.rglob('*') if p.is_file() and not p.name.endswith('.partial')}
    if not hashes:raise ValueError('Terminal run has no raw outputs')
    save(readout/'collection.json',{'utc':now(),'slug':SLUG,'kernel_id':receipt['kernelId'],'version':receipt['versionNumber'],'status':status,'files':files,'file_hashes':hashes})
    if auditor is None:auditor=subprocess_audit
    try:result=auditor(raw/'marlin-replay')
    except Exception as error:
        # Collection is complete and hashed; an audit failure is terminal, never a reason to re-collect.
        result={'qualified':False,'audit_error':type(error).__name__,'message':str(error)[-2000:]}
    save(readout/'audit.json',result)
    return result


def subprocess_audit(raw):
    import subprocess
    out=Path(raw).parent.parent/'audit-subprocess.json'
    done=subprocess.run([str(AUDIT_PYTHON),'-B',str(HERE/'audit_capture.py'),str(raw),'--output',str(out)],cwd=HERE,capture_output=True,text=True)
    if done.returncode:raise RuntimeError('audit_capture.py failed: '+done.stderr[-1500:])
    return json.loads(out.read_text())


def main():
    root=HERE;verify_local(root);readout=root/'readout';readout.mkdir(exist_ok=True)
    with (readout/'collector.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from kaggle_backend import KaggleBackend
        backend=KaggleBackend()
        save(readout/'collector-ready.json',{'utc':now(),'pid':os.getpid(),'slug':SLUG,'read_only':True,'freeze_sha256':sha(root/'launch-freeze.json')})
        deadline=time.time()+48*3600
        while time.time()<deadline:
            attempt=root/'evidence/launch-attempt.json'
            if not attempt.exists():time.sleep(1);continue
            row=json.loads(attempt.read_text())
            if row['state']!='accepted':
                if row['state']=='attempting':time.sleep(1);continue
                save(readout/'collector-stop.json',{'utc':now(),'reason':'push not accepted or unresolved; no relaunch'});return
            receipt=row['response']
            try:
                remote_guard(backend,receipt,root)
                status=backend.status(SLUG)
                with (readout/'status-history.jsonl').open('a') as stream:stream.write(json.dumps({'utc':now(),**status})+'\n')
                if status['status'] in TERMINAL:
                    result=collect_outputs(backend,receipt,status,root)
                    print(json.dumps({'utc':now(),'status':status['status'],'qualified':result.get('qualified',False)}),flush=True);return
            except Exception as error:
                with (readout/'errors.jsonl').open('a') as stream:stream.write(json.dumps({'utc':now(),'error_type':type(error).__name__,'message':str(error)[:300] if isinstance(error,ValueError) else None})+'\n')
            time.sleep(120)
        save(readout/'collector-stop.json',{'utc':now(),'reason':'48-hour read-only collection deadline; no relaunch'})


if __name__=='__main__':main()
