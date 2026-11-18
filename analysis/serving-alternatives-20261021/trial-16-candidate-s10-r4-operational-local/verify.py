"""Operational-only CPU/SDK qualification; no runtime61/65 or API/GPU run."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
SDK=Path('/home/pranav/.local/share/uv/tools/kaggle/bin/python')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(destination):
    destination=Path(destination).resolve();destination.mkdir(exist_ok=False);started=time.monotonic();end=started+600
    paths=[*HERE.glob('*.py'),*HERE.glob('*.json'),*HERE.glob('tests/*.py'),*HERE.glob('collection_tools/*.py')]
    before={str(p.relative_to(HERE)):sha(p) for p in paths if p.name not in ('freeze.json','review-packet.json') and not p.name.startswith('HANDOFF')}
    env=dict(os.environ,PYTHONPATH=str(HERE/'guard')+os.pathsep+str(HERE),LOCAL_ADAPTER_PROOF='1',PACKET_PROOF_DIR=str(destination),PYTHONDONTWRITEBYTECODE='1')
    steps=[]
    commands=[(name,[str(SDK),'-B','-m','unittest','discover','-s','tests','-p',pattern,'-v']) for name,pattern in [('quota-http','test_quota_http.py'),('quota-window','test_quota_window.py'),('reconciliation','test_reconciliation.py'),('collection-reporting','test_collection_reporting.py')]]
    commands.append(('SDK',[str(SDK),'-B',str(HERE/'verify_sdk.py'),str(destination/'SDK.json')]))
    for name,argv in commands:
        begin=time.monotonic()
        with (destination/(name+'.log')).open('x') as stream:value=subprocess.run(argv,cwd=HERE,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=max(.001,end-time.monotonic()))
        steps.append(dict(name=name,argv=argv,returncode=value.returncode,seconds=time.monotonic()-begin))
    after={name:sha(HERE/name) for name in before}
    proof=dict(pass_=all(s['returncode']==0 for s in steps) and before==after,seconds=time.monotonic()-started,steps=steps,source_hashes=before,source_stable=before==after,
               SDK_python=str(SDK),CPU_cap_seconds=600,API_calls=0,GPU_work=False,runtime61_rerun=False,baseline65_rerun=False,inherited_baseline='64/65 conservative identity abort remains disclosed',
               inherited_runtime_qualification='approved trial16-r2 exact runtime/42payloads unchanged; full selected qualification retained by hash/reference')
    (destination/'PROOF.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(dict(pass_=proof['pass_'],seconds=proof['seconds'],sha256=sha(destination/'PROOF.json'))));return 0 if proof['pass_'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination');a=p.parse_args();raise SystemExit(run(a.destination))
