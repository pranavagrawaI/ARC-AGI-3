"""One-attempt launch and read-only version/source-bound monitoring."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiGetKernelSessionLogsStreamRequest

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'analysis/mtp-depth-20261006'))
import launch_watch as lw
lw.EVID=HERE;lw.NOTEBOOK_DIR=HERE/'kernel'


def save(name,value):
    lw.save(HERE/name,value)


def verify():
    q=json.loads((HERE/'LOCAL-QUALIFICATION.json').read_text())
    meta=json.loads((HERE/'kernel/kernel-metadata.json').read_text())
    nb=HERE/'kernel'/meta['code_file']
    if hashlib.sha256(nb.read_bytes()).hexdigest()!=q['notebook_sha256']:raise ValueError('notebook changed')
    if hashlib.sha256((HERE/'kernel/kernel-metadata.json').read_bytes()).hexdigest()!=q['metadata_sha256']:raise ValueError('metadata changed')
    for name in ['routing.py','patch_runtime.py','test_routing.py','screen.py','isolated_phase.py']:
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=q['asset_sha256'][name]:raise ValueError('asset changed: '+name)
    cells=json.loads(nb.read_text())['cells']
    cell=''.join(cells[4]['source'])
    start=cell.index("KIT = Path('/kaggle/reference-mass-kit')")
    end=cell.index("exec(compile((KIT/'screen.py')",start)
    embedded={}
    for node in ast.parse(cell[start:end]).body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
            call=node.value
            if isinstance(call.func,ast.Attribute) and call.func.attr=='write_text':
                embedded[ast.literal_eval(call.func.value.right)]=ast.literal_eval(call.args[0])
    if set(embedded)!=set(q['asset_sha256']):raise ValueError('embedded asset coverage mismatch')
    for name,content in embedded.items():
        if hashlib.sha256(content.encode()).hexdigest()!=q['asset_sha256'][name]:raise ValueError('embedded asset drift: '+name)
    if not q['model_sources_unchanged'] or q['hard_cap_s']!=3600:raise ValueError('invalid cap/model identity')
    return q,meta


def api():
    original=requests.Session.send
    def send(session,request,**kwargs):
        kwargs['timeout']=(15,30)
        return original(session,request,**kwargs)
    requests.Session.send=send
    lw.refresh_token()
    a=KaggleApi();a.authenticate();return a


def launch():
    if (HERE/'launch-attempt.json').exists():raise RuntimeError('already attempted; never retry')
    q,meta=verify();a=api();used,quota=lw.quota_used_s(a);active=lw.active_sessions(a)
    save('admission.json',{'utc':lw.now(),'used_and_reserved_s':used,'quota':quota,'active':active,'cap_s':3600})
    if len(active)>=2 or any(s.startswith('unknown') for _,s in active) or used+3600>30*3600:raise RuntimeError('quota/slots not available')
    kernels=a.kernels_list(mine=True,user=meta['id'].split('/')[0],page_size=100,sort_by='dateRun')
    if meta['id'] in {getattr(k,'ref',None) for k in kernels}:raise RuntimeError('remote slug already exists; reconcile')
    identity={**q,'metadata':meta,'card_sha256':hashlib.sha256((HERE/'EXPERIMENT-CARD.md').read_bytes()).hexdigest()}
    save('launch-identity.json',identity)
    body=(f"GPU slot claim: codex-reference-mass {meta['id']} v1. Pranav requested the repaired rerun on 2026-10-15: one short screen, 3600 GPU-seconds TOTAL including startup, no retry. "
          f"One RTX PRO6000, sequential fresh control top10 then target active5/reference10, 900s play each, unchanged top10 draft/weights/c14. "
          f"Exact wheel/source-bound callback locally qualified; CUDA graph test before model loading. Used/reserved {used/3600:.3f}h, active {active}. "
          f"Notebook SHA {q['notebook_sha256']}; card analysis/expert-reference-mass-20261014/repaired/EXPERIMENT-CARD.md. No competition submit or promotion.")
    subprocess.run([sys.executable,'-B',str(Path.home()/'.agent-board/board.py'),'reply','682','--as','codex-reference-mass','--body',body],check=True)
    with (HERE/'launch-attempt.json').open('x') as f:json.dump({'utc':lw.now(),'state':'attempting','identity':identity},f,indent=2)
    try:
        result=json.loads(str(a.kernels_push(str(HERE/'kernel'),timeout='3600')))
        save('launch-response.json',result)
        if result.get('error') or not result.get('versionNumber'):raise RuntimeError(result)
        save('launch-attempt.json',{'utc':lw.now(),'state':'accepted','response':result,'identity':identity})
        print(json.dumps(result),flush=True)
    except Exception as exc:
        save('launch-error.json',{'utc':lw.now(),'exception':repr(exc),'acceptance':'unresolved; do not retry'})
        raise


def status(stream=False):
    q,meta=verify();a=api();receipt=json.loads((HERE/'launch-response.json').read_text())
    request=ApiGetKernelRequest();request.user_name,request.kernel_slug=meta['id'].split('/')
    with a.build_kaggle_client() as c:remote=c.kernels.kernels_api_client.get_kernel(request)
    if remote.metadata.id!=receipt['kernelId'] or remote.metadata.current_version_number!=receipt['versionNumber']:raise ValueError('remote identity mismatch')
    cells=lambda n:[(c['cell_type'],''.join(c['source'])) for c in n['cells']]
    if cells(json.loads(remote.blob.source))!=cells(json.loads((HERE/'kernel'/meta['code_file']).read_text())):raise ValueError('remote source mismatch')
    state={'utc':lw.now(),**json.loads(str(a.kernels_status(meta['id']))),'source_verified':True}
    save('stream-status.json' if stream else 'status.json',state);print(json.dumps(state),flush=True)
    if stream:
        request=ApiGetKernelSessionLogsStreamRequest();request.user_name,request.kernel_slug=meta['id'].split('/');request.wait_for_logs_url_seconds=1
        chunks=[];deadline=time.monotonic()+25
        try:
            with a.build_kaggle_client() as c:
                response=c.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
                with response:
                    response.raise_for_status()
                    for line in response.iter_lines(chunk_size=1024,decode_unicode=True):
                        if line:chunks.append(line if isinstance(line,str) else line.decode(errors='replace'))
                        if len(chunks)>15000 or time.monotonic()>deadline:break
        except requests.RequestException as exc:print('log stream boundary',type(exc).__name__,flush=True)
        raw='\n'.join(chunks);(HERE/'evidence/latest-kernel-log.txt').write_text(raw)
        decoded=[]
        for line in chunks:
            if line.startswith('data: '):
                try:decoded.append(json.loads(line[6:]).get('data',''))
                except json.JSONDecodeError:pass
        text=''.join(decoded)
        (HERE/'evidence/latest-kernel-log-decoded.txt').write_text(text)
        print(text[-1500:] if text else raw[-500:],flush=True)
    return state


if __name__=='__main__':
    if sys.argv[1]=='launch':launch()
    elif sys.argv[1]=='status':status('--stream' in sys.argv)
    else:raise ValueError('launch or status')
