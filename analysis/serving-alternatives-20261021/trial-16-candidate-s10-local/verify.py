"""Bounded changed-path CPU qualification. Never launches or contacts Kaggle."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
CPU=Path('/home/pranav/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12')
SDK=Path('/home/pranav/.local/share/uv/tools/kaggle/bin/python')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(destination):
    destination=Path(destination).resolve();destination.mkdir(exist_ok=False)
    started=time.monotonic();end=started+600
    bound=[*HERE.glob('*.py'),*HERE.glob('*.json'),*HERE.glob('tests/*.py'),*HERE.glob('collection_tools/*.py')]
    before={str(path.relative_to(HERE)):sha(path) for path in bound if path.name not in ('freeze.json','review-packet.json')}
    env=dict(os.environ,PYTHONPATH=str(HERE/'guard')+os.pathsep+str(HERE),LOCAL_ADAPTER_PROOF='1',PACKET_PROOF_DIR=str(destination),PYTHONDONTWRITEBYTECODE='1')
    rows=[]
    steps=[('tests',[str(CPU),'-B','-m','unittest','discover','-s','tests','-v']),
        ('sdk-primary',[str(SDK),'-B','-m','unittest','discover','-s','tests','-p','test_primary_collection.py','-v']),
        ('sdk-operational',[str(SDK),'-B','-m','unittest','discover','-s','tests','-p','test_reconciliation.py','-v']),
        ('sdk',[str(SDK),'-B',str(HERE/'verify_sdk.py'),str(destination/'SDK.json')]),
        ('bodies',[str(CPU),'-B',str(HERE/'verify_bodies.py'),str(destination/'BODIES.json')])]
    for name,argv in steps:
        begin=time.monotonic()
        with (destination/(name+'.log')).open('x') as stream:
            value=subprocess.run(argv,cwd=HERE,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=max(.001,end-time.monotonic()))
        rows.append(dict(name=name,argv=argv,returncode=value.returncode,seconds=time.monotonic()-begin))
    after={name:sha(HERE/name) for name in before}
    report=dict(pass_=all(row['returncode']==0 for row in rows) and before==after,seconds=time.monotonic()-started,
        cpu_seconds_cap=600,steps=rows,source_hashes=before,source_stable=before==after,CPU_python=str(CPU),SDK_python=str(SDK),
        baseline65_rerun=False,inherited_baseline='64/65 disclosed conservative identity abort',network_calls=0,gpu_work=False,
        proof_scope='explicit CPU/analytical package/window substitutions; no GPU/engine/throughput claim')
    (destination/'PROOF.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(pass_=report['pass_'],seconds=report['seconds'],proof_sha256=sha(destination/'PROOF.json'))))
    return 0 if report['pass_'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination');a=p.parse_args();raise SystemExit(run(a.destination))
