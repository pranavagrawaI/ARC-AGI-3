"""Read-only poll of the pushed kernel until terminal. Each tick records the exact-version_label call (may 404 while queued,
as in R9) and the current-identity call; identity is unambiguous because exactly one version exists and no repush is allowed."""
import json,subprocess,sys,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[2];SDK='/home/pranav/.local/share/uv/tools/kaggle/bin/python'
TARGET='pranavagrawai/arc3-ac-competition-20260917-v1';VERSION=sys.argv[1];MAX=int(sys.argv[2]) if len(sys.argv)>2 else 10800;INTERVAL=int(sys.argv[3]) if len(sys.argv)>3 else 60
CODE=r'''
import json,sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest,ApiGetKernelRequest
api=KaggleApi();api.authenticate();out={}
with api.build_kaggle_client() as c:
    for scope,label in (('EXACT_VERSION_LABEL',sys.argv[2]),('CURRENT_IDENTITY',None)):
        r=ApiGetKernelSessionStatusRequest();r.user_name,r.kernel_slug=sys.argv[1].split('/')
        if label:r.version_label=label
        try:s=c.kernels.kernels_api_client.get_kernel_session_status(r);out[scope]=dict(status=s.status.name,failure_message=s.failure_message)
        except Exception as e:out[scope]=dict(error=type(e).__name__,detail=str(e)[:100])
    g=ApiGetKernelRequest();g.user_name,g.kernel_slug=sys.argv[1].split('/')
    try:m=c.kernels.kernels_api_client.get_kernel(g).metadata;out['current_version_number']=m.current_version_number
    except Exception as e:out['current_version_number']=dict(error=type(e).__name__)
print(json.dumps(out))
'''
log=RUN/'launch'/f'monitor-commit-v{VERSION}.jsonl';t0=time.time();errors=0
TERMINAL={'COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'}
while True:
    r=subprocess.run([SDK,'-B','-c',CODE,TARGET,VERSION],text=True,capture_output=True,timeout=120)
    rec=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),elapsed=round(time.time()-t0),returncode=r.returncode,stderr=r.stderr.strip()[-300:])
    status=None
    try:
        d=json.loads(r.stdout);rec.update(d)
        if d.get('current_version_number')!=int(VERSION):rec['identity_warning']='current_version_number != '+VERSION
        status=d['EXACT_VERSION_LABEL'].get('status') or d['CURRENT_IDENTITY'].get('status');rec['status']=status
        rec['status_scope']='EXACT_VERSION_LABEL' if d['EXACT_VERSION_LABEL'].get('status') else 'CURRENT_IDENTITY_ONLY'
    except Exception as e:errors+=1;rec['parse_error']=str(e)
    with log.open('a') as f:f.write(json.dumps(rec)+'\n')
    print(rec.get('utc'),rec.get('status'),rec.get('status_scope'),rec.get('current_version_number'),flush=True)
    if status in TERMINAL:print('TERMINAL',json.dumps(rec));break
    if errors>=10:print('READ-FAILURES',json.dumps(rec));sys.exit(3)
    if time.time()-t0>MAX:print('MONITOR-BOUND',json.dumps(rec));sys.exit(4)
    time.sleep(INTERVAL)
