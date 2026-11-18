"""Sole authorized launch, fail closed, never retry a consumed identity."""
from datetime import datetime, timezone
import hashlib,json,subprocess,time
from pathlib import Path
from verify_package import HERE,PACKAGE,verify
CLI='/home/pranav/.local/bin/kaggle'
def receipt(args,name):
    p=subprocess.run([CLI,*args],stdin=subprocess.DEVNULL,close_fds=True,capture_output=True,text=True,timeout=90)
    r={'arguments':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'utc':datetime.now(timezone.utc).isoformat()}
    (HERE/name).write_text(json.dumps(r,indent=2)+'\n')
    assert p.returncode==0,'CLI_FAILED:'+name
    return r

def reconcile():
    m=json.loads((HERE/'MANIFEST.json').read_text())
    for path,h in m['files'].items():assert hashlib.sha256((HERE.parents[1]/path).read_bytes()).hexdigest()==h,path
    verify()
    return m

def approval_valid(approval, digest):
    return approval.get('status')=='resolved_for_bounded_probe' and approval.get('manifest_sha256')==digest

def main():
    m=reconcile(); b=json.loads((PACKAGE/'BUILD.json').read_text()); kernel=b['kernel']
    assert b['launch_authorized'] is True
    assert approval_valid(json.loads((HERE/'REVIEW-RESOLVED.json').read_text()),hashlib.sha256((HERE/'MANIFEST.json').read_bytes()).hexdigest()),'REVIEW_NOT_APPROVED'
    assert not (HERE/'LAUNCH-ATTEMPT.json').exists(),'NO_RETRY'
    q=receipt(['quota','--format','json'],'quota-before.json')
    gpu=next(x for x in json.loads(q['stdout']) if x['resource']=='GPU')
    assert gpu['remaining'].endswith('h') and float(gpu['remaining'][:-1])>=.5
    r=receipt(['kernels','list','--mine','--search',kernel.split('/')[1],'--page-size','100','--format','json'],'target-absence.json')
    assert r['stdout'].strip() in ('Not found','No kernels found','[]'),'TARGET_NOT_ABSENT'
    receipt(['kernels','pull','pranavagrawai/containment-q-20260913t160618z','--metadata','-p',str(HERE/'image-preflight')],'image-preflight-receipt.json')
    meta=json.loads((HERE/'image-preflight/kernel-metadata.json').read_text())
    request=json.loads((HERE/'PROVIDER-CONTRACT.json').read_text())['request']
    assert meta['docker_image']==request['docker_image'] and meta['machine_shape']==request['machine_shape']
    assert request['is_private'] is True and request['enable_internet'] is False
    reconcile()
    with (HERE/'LAUNCH-ATTEMPT.json').open('x') as f:json.dump({'kernel':kernel,'epoch':time.time(),'utc':datetime.now(timezone.utc).isoformat(),'requested_version':1,'timeout_seconds':300,'accelerator':'NvidiaRtxPro6000','no_retry':True},f,indent=2)
    r=receipt(['kernels','push','--path',str(PACKAGE),'--timeout','300','--accelerator','NvidiaRtxPro6000'],'launch.json')
    assert 'Kernel version 1 successfully pushed.' in r['stdout'] and 'https://www.kaggle.com/code/'+kernel in r['stdout'],'UNCERTAIN_ADMISSION_NO_RETRY'
    print(r['stdout'])
if __name__=='__main__':main()
