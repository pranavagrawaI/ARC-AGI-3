"""One explicit push; a consumed launch identity cannot be retried."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from verify_package import verify, HERE, PACKAGE


def cli(arguments, output):
    process=subprocess.run(['kaggle',*arguments],stdin=subprocess.DEVNULL,close_fds=True,
                           capture_output=True,text=True,timeout=90)
    receipt={'arguments':arguments,'returncode':process.returncode,'stdout':process.stdout,
             'stderr':process.stderr,'utc':datetime.now(timezone.utc).isoformat()}
    (HERE/output).write_text(json.dumps(receipt,indent=2)+'\n')
    if process.returncode:
        raise RuntimeError('CLI_FAILED:'+output)
    return receipt


def main():
    frozen=json.loads((HERE/'PRELAUNCH.json').read_text())
    assert verify()==frozen,'FROZEN_MANIFEST_DRIFT'
    approval=json.loads((HERE/'REVIEW-RESOLVED.json').read_text())
    assert approval['status']=='resolved_for_bounded_probe'
    assert approval['reviewed_source_hashes']=={n:frozen['files'][str(HERE/n)] for n in ('runner.py','probe.py','test_qualification.py','EXPERIMENT-CARD.md')}
    assert not (HERE/'LAUNCH-ATTEMPT.json').exists(),'NO_RETRY'
    quota=cli(['quota','--format','json'],'quota-before.json')
    gpu=next(r for r in json.loads(quota['stdout']) if r['resource']=='GPU')
    assert gpu['remaining'].endswith('h') and float(gpu['remaining'][:-1])>=.5,'INSUFFICIENT_QUOTA'
    absence=cli(['kernels','list','--mine','--search','containment-q-20260913t160618z','--page-size','100','--format','json'],'target-absence.json')
    text=absence['stdout'].strip()
    rows=[] if text in ('Not found','No kernels found') else json.loads(text)
    assert rows==[],'TARGET_NOT_ABSENT'
    # Reconcile immediately before the irreversible operation, then consume ID
    # atomically before invoking CLI. Exceptions/uncertain admission never retry.
    assert verify()==frozen
    with (HERE/'LAUNCH-ATTEMPT.json').open('x') as f:
        json.dump({'utc':datetime.now(timezone.utc).isoformat(),'epoch':time.time(),
                   'kernel':frozen['kernel'],'requested_version':1,'timeout_seconds':300,
                   'accelerator':'NvidiaRtxPro6000','no_retry':True},f,indent=2)
    result=cli(['kernels','push','--path',str(PACKAGE),'--timeout','300','--accelerator','NvidiaRtxPro6000'],'launch.json')
    assert 'Kernel version 1 successfully pushed.' in result['stdout'], 'UNCERTAIN_OR_WRONG_VERSION'
    assert 'https://www.kaggle.com/code/pranavagrawai/containment-q-20260913t160618z' in result['stdout']
    print(result['stdout'])

if __name__=='__main__': main()
