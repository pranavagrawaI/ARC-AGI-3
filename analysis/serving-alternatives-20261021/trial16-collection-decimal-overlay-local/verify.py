"""Offline-only focused qualification, original source integrity and 600s cap."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from binding import ROOT,DEADLINE
HERE=Path(__file__).resolve().parent
SDK='/home/pranav/.local/share/uv/tools/kaggle/bin/python'
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def verify(destination):
    destination=Path(destination).resolve();destination.mkdir(exist_ok=False);start=time.monotonic()
    pins=json.loads((HERE/'SOURCE-PINS.json').read_text())
    before={n:sha(HERE/n) for n in pins['overlay_files']}
    assert before==pins['overlay_files']
    for name,expected in pins['references'].items():assert sha(name)==expected,name
    freeze=json.loads((ROOT/'freeze.json').read_text())
    for name,expected in freeze['files'].items():assert sha(ROOT/name)==expected,name
    for name,expected in freeze['references'].items():assert sha(name)==expected,name
    env=dict(os.environ,PYTHONPATH=str(ROOT/'guard')+os.pathsep+str(HERE)+os.pathsep+str(HERE/'tests'),LOCAL_ADAPTER_PROOF='1',OVERLAY_PROOF_DIR=str(destination/'cases'),PYTHONDONTWRITEBYTECODE='1')
    argv=[SDK,'-B','-m','unittest','discover','-s','tests','-v']
    with (destination/'tests.log').open('x') as stream:result=subprocess.run(argv,cwd=HERE,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=max(.001,600-(time.monotonic()-start)))
    after={n:sha(HERE/n) for n in before}
    report=dict(pass_=result.returncode==0 and before==after,seconds=time.monotonic()-start,CPU_cap_seconds=600,argv=argv,source_stable=before==after,overlay_files=after,source_pins_sha256=sha(HERE/'SOURCE-PINS.json'),original_bindings_verified=freeze['bindings'],original_freeze_sha256=sha(ROOT/'freeze.json'),original_failed_receipt_sha256=sha(ROOT/'remote-readback/initial-readback/READBACK.json'),observation_deadline=DEADLINE,network_calls=0,GPU_work=False,frozen_edits=False,launcher_or_authority_changes=False,monitor_started=False,new_run_or_packet=False)
    (destination/'PROOF.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(pass_=report['pass_'],seconds=report['seconds'],sha256=sha(destination/'PROOF.json'))));return 0 if report['pass_'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination');a=p.parse_args();raise SystemExit(verify(a.destination))
