"""Bounded CPU/loopback-only packet qualification; no remote API/auth/model work."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
SDK=Path('/home/pranav/.local/share/uv/tools/kaggle/bin/python')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def bindings():
 frozen=json.loads((HERE/'freeze.json').read_text())
 for name,expected in frozen['files'].items():
  if sha(HERE/name)!=expected:raise ValueError('frozen input drift '+name)
 for name,expected in frozen['references'].items():
  if sha(name)!=expected:raise ValueError('reference drift '+name)
 for name in ('slot-confirmation.json','launch-attempt.json','launch-response.json','kernel'):
  if (HERE/name).exists():raise ValueError('unexpected real authority/attempt '+name)
 return dict(freeze_sha256=sha(HERE/'freeze.json'),files=len(frozen['files']),references=len(frozen['references']))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
 out=args.output.resolve();out.mkdir(exist_ok=False);signal.alarm(600);started=time.monotonic();before=bindings()
 def run(argv,name,env=None):
  value=subprocess.run(argv,cwd=HERE,env=env,capture_output=True,text=True,timeout=max(1,590-(time.monotonic()-started)))
  (out/(name+'.stdout')).write_text(value.stdout);(out/(name+'.stderr')).write_text(value.stderr)
  if value.returncode:raise RuntimeError(name+' failed: '+value.stderr[-3000:])
  return value
 # Preserve full accepted runtime regression coverage through the separately
 # accepted test-only revision, without importing its local modules into this process.
 r6=HERE.parent/'release-adapters-r6-test-local-20261023'
 run([sys.executable,str(r6/'verify.py'),'--output',str(out/'r6-full-suite')],'r6-full-suite')
 env=dict(os.environ,LOCAL_ADAPTER_PROOF='1',PYTHONPATH=str(HERE/'guard'),PACKET_PROOF_DIR=str(out/'packet-cases'))
 run([sys.executable,'-m','unittest','discover','-s',str(HERE/'tests'),'-v'],'packet-suite',env)
 run([str(SDK),str(HERE/'verify_sdk.py'),str(out/'sdk-proof.json')],'sdk-proof')
 import packet,transport,launch
 with __import__('tempfile').TemporaryDirectory() as directory:
  report=packet.prepare(Path(directory)/'review')
 if report!=json.loads((HERE/'review-packet.json').read_text()):raise ValueError('review packet differs from fresh build')
 try:launch.main()
 except ValueError as error:
  if not str(error).startswith('no reviewed serialized assignment'):raise
 else:raise ValueError('unassigned launcher did not refuse')
 after=bindings()
 if before!=after:raise ValueError('immutable inputs changed during tests/preflight fixtures')
 base=json.loads((out/'r6-full-suite/PROOF.json').read_text());sdk=json.loads((out/'sdk-proof.json').read_text())
 if sdk['network_calls'] or sdk['network_attempts']:raise ValueError('network attempted')
 inventory={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
 report=dict(status='UNASSIGNED_LOCAL_PACKET_QUALIFIED_NOT_GPU_RELEASE',bindings=after,
  seconds=time.monotonic()-started,cpu_cap_s=600,base_tests=base['tests'],base_artifact_refs=base['artifact_references_checked'],
  packet_tests_log_sha256=sha(out/'packet-suite.stderr'),sdk_proof_sha256=sha(out/'sdk-proof.json'),sdk_network_calls=0,
  sources_sha256=sdk['sources_sha256'],config_sha256=sdk['config_sha256'],unassigned_launcher_refused=True,
  platform_s=3900,cooperative_s=3600,work_s=3300,cleanup_s=300,evidence_inventory=inventory,
  gpu_or_model_work=False,external_network=0,loopback_only=True,
  remaining=['actual nvidia/Kaggle/native integration and release unproved','140GiB provenance/eligibility unresolved','exact review/new serialized assignment required'])
 (out/'PROOF.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ('status','bindings','seconds','base_tests','sources_sha256','config_sha256')}));signal.alarm(0)

if __name__=='__main__':main()
