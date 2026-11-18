"""One private GPU launch after qualification and live resource admission. No retry."""
from pathlib import Path
import datetime,hashlib,json,sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'mtp-depth-20261006'))
import launch_watch as lw

def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def main():
    evidence=HERE/'evidence';ident=json.loads((HERE/'identity.json').read_text())
    if (evidence/'launch-attempt.json').exists():raise RuntimeError('attempt already exists; reconcile, never retry')
    metadata=json.loads((HERE/'kernel/kernel-metadata.json').read_text())
    assert metadata==ident['metadata'] and metadata['is_private']
    notebook=HERE/'kernel'/metadata['code_file']
    assert hashlib.sha256(notebook.read_bytes()).hexdigest()==ident['notebook_sha256']
    for name,sha in ident['manifest'].items():
        if not name.startswith('sources/'):
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha,('drift',name)
    assert (evidence/'transport-qualification.json').exists()
    patch_qualification=json.loads((evidence/'patch-qualification.json').read_text())
    assert patch_qualification['git_reverse_then_forward_restores_all_files']
    qualification=json.loads((evidence/'e2e-qualification.json').read_text())
    assert [r['status'] for r in qualification['results']]==['solved','http_error']
    lw.EVID=evidence;lw.NOTEBOOK_DIR=HERE/'kernel';lw.refresh_token()
    api=lw.KaggleApi();api.authenticate()
    active=lw.active_sessions(api);used,quota=lw.quota_used_s(api)
    admission={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'active':active,'quota':quota,
               'used_and_reserved_seconds':used,'hard_cap_seconds':10500}
    save(evidence/'admission.json',admission)
    if len(active)>=2 or any(str(status).startswith('unknown') for _,status in active) or used+10500>30*3600:
        raise RuntimeError('no free GPU slot/quota; no launch attempted')
    if '--dry' in sys.argv:print('READY',json.dumps(admission));return
    attempt={**admission,'state':'attempting','identity_sha256':hashlib.sha256((HERE/'identity.json').read_bytes()).hexdigest()}
    with (evidence/'launch-attempt.json').open('x') as f:json.dump(attempt,f,indent=2)
    try:
        response=json.loads(str(api.kernels_push(str(HERE/'kernel'),timeout='10500')))
        save(evidence/'launch-response.json',response)
        if response.get('error') or not response.get('versionNumber'):raise RuntimeError(str(response))
        attempt.update(state='accepted',response=response)
    except Exception as exc:
        attempt.update(state='acceptance_unresolved',error=repr(exc))
        response=getattr(exc,'response',None)
        if response is not None:attempt.update(http_status=response.status_code,response_body=response.text)
    save(evidence/'launch-attempt.json',attempt);print(json.dumps(attempt,indent=2))

if __name__=='__main__':main()
