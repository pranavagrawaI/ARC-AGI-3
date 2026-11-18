"""One bounded full current-packet CPU proof; inherited64/65 never rerun."""
import argparse,hashlib,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
SDK='/home/pranav/.local/share/uv/tools/kaggle/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bindings():
 v=json.loads((HERE/'freeze.json').read_text())
 for n,h in v['files'].items():
  if sha(HERE/n)!=h:raise ValueError('input drift '+n)
 for n,h in v['references'].items():
  if sha(n)!=h:raise ValueError('dependency drift '+n)
 return dict(freeze_sha256=sha(HERE/'freeze.json'),files=len(v['files']),references=len(v['references']))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False);signal.alarm(600);started=time.monotonic();before=bindings()
 env=dict(os.environ,PYTHONPATH=str(HERE)+os.pathsep+str(HERE/'guard'),LOCAL_ADAPTER_PROOF='1',PACKET_PROOF_DIR=str(out/'cases'),LOCAL_EVIDENCE=str(out/'retained-wire-witnesses'))
 commands=[('tests',[sys.executable,'-m','unittest','discover','-s',str(HERE/'tests'),'-v']),('bodies',[sys.executable,str(HERE/'verify_bodies.py'),str(out/'BODY-PROOF.json')]),('sdk',[SDK,str(HERE/'verify_sdk.py'),str(out/'SDK-PROOF.json')])]
 for label,cmd in commands:
  value=subprocess.run(cmd,cwd=HERE,env=env if label!='sdk' else dict(os.environ),capture_output=True,text=True,timeout=max(1,590-(time.monotonic()-started)))
  (out/(label+'.stdout')).write_text(value.stdout);(out/(label+'.log')).write_text(value.stderr)
  if value.returncode:raise ValueError(label+' failed; original logs/cases retained')
 import packet,launch,release_runner,capacity_runner,release_core,phase_tools,provenance,capacity_evidence
 for m in (packet,launch,release_runner,capacity_runner,release_core,phase_tools,provenance,capacity_evidence):
  if Path(m.__file__).resolve().parent!=HERE:raise ValueError('actual import origin '+m.__name__)
 with tempfile.TemporaryDirectory() as td:report=packet.prepare(Path(td)/'review')
 if report!=json.loads((HERE/'review-packet.json').read_text()):raise ValueError('review packet changed')
 try:launch.clearance()
 except ValueError as error:
  if not str(error).startswith('no reviewed serialized assignment'):raise
 else:raise ValueError('unexpected authority')
 sdk=json.loads((out/'SDK-PROOF.json').read_text());body=json.loads((out/'BODY-PROOF.json').read_text())
 if sdk['network_calls'] or sdk['network_attempts'] or sdk['decoded_payloads']!=43:raise ValueError('SDK scope/source drift')
 baseline=(HERE/'development-evidence/runtime-first/tests.log').read_text()
 if 'Ran 65 tests' not in baseline or 'FAILED (failures=1)' not in baseline:raise ValueError('inherited64/65 disclosure drift')
 after=bindings()
 if before!=after:raise ValueError('bindings changed during qualification')
 proof=dict(status='UNASSIGNED_EXECUTABLE_PACKET_LOCAL_QUALIFIED_NOT_LAUNCH_AUTHORITY',seconds=time.monotonic()-started,bindings=after,source_sha256=report['sources_sha256'],config_sha256=report['config_sha256'],policy_sha256=sha(HERE/'prior-reconciliation-policy.json'),payloads=43,SDK_network_calls=0,SDK_network_attempts=0,body_records=body['records'],body_bytes=body['body_bytes'],baseline='Inherited64/65 preserved; no baseline rerun or full65pass claim',actual_import_origins_checked=True,no_authority=True,gpu_work=False,external_network=0,inventory={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
 with (out/'PROOF.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in proof.items() if k!='inventory'}));signal.alarm(0)
if __name__=='__main__':main()
