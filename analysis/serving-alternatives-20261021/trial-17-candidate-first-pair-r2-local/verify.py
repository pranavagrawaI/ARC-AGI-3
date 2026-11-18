"""Fresh-output-only local qualification. No launcher, authority or remote actions."""
import ast,hashlib,importlib,inspect,io,json,os,socket,sys,time,trace,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
PYTHON='/home/pranav/.local/share/uv/tools/kaggle/bin/python'

def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def original_bindings():
 old=HERE.parent/'trial-16-candidate-s10-r4-operational-local';f=json.loads((old/'freeze.json').read_bytes())
 if sha(old/'freeze.json')!='b51120fedef231e6b5fdecd8882f03bb78653e4e221fa9b256cdd7d908c2ed5b':raise ValueError('predecessor freeze drift')
 pairs={str(old/name):digest for name,digest in f['files'].items()}|f['references']
 for path,digest in pairs.items():
  if sha(path)!=digest:raise ValueError('preserved binding changed '+path)
 return dict(verified=len(pairs),freeze_sha256=sha(old/'freeze.json'))

def risk(counts):
 files=('monitor.py','launch.py','reconciliation.py','build.py','packet.py','collection_tools/owned_cli.py','collection_tools/plan_required.py','collection_tools/space.py','candidate_evidence.py','native_support.py','control_prepare.py','candidate_reduce.py','reduce_api.py','decimal_quota.py','collection_tools/remote.py','collection_tools/primary.py','candidate_runner.py','phase_tools.py','primary_bundle.py','pair_runner.py','pair_evidence.py','pair_resources.py','provenance.py','observation.py','release_runner.py','collection_tools/required_files.py','collection_tools/collector.py','collection_tools/retry_transport.py','collection_tools/run_required.py','collection_tools/audit_required.py','collection_tools/publication.py')
 rows=[]
 for name in files:
  path=(HERE/name).resolve();tree=ast.parse(path.read_text())
  for node in ast.walk(tree):
   if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
   cc=1+sum(isinstance(x,(ast.If,ast.For,ast.While,ast.ExceptHandler,ast.IfExp,ast.comprehension)) for x in ast.walk(node))+sum(max(0,len(x.values)-1) for x in ast.walk(node) if isinstance(x,ast.BoolOp))
   lines={x.lineno for x in ast.walk(node) if isinstance(x,ast.stmt) and not isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
   hit=sum(counts.get((str(path),line),0)>0 for line in lines);coverage=hit/len(lines) if lines else 0
   rows.append(dict(file=name,function=node.name,line=node.lineno,CC_ast=cc,statement_coverage=coverage,CRAP_indicator=cc*cc*(1-coverage)**3+cc))
 return dict(measurement='stdlib trace function statement lines + same AST decision rule as baseline; no branch coverage/GPU execution or CRAP gate claim',
   comparison='Baseline recorded CC only; no comparable baseline coverage/improvement claim. No dependency installed or threshold invented.',
   tests='This packet changed-path tests only; child-process execution is not traced; covered lines alone do not establish assertion strength.',rows=sorted(rows,key=lambda x:-x['CRAP_indicator']))

def qualify(destination):
 destination=Path(destination).resolve();destination.mkdir(exist_ok=False);started=time.monotonic()
 if sys.executable!=PYTHON and Path(sys.executable).resolve()!=Path(PYTHON).resolve():raise ValueError('installed Kaggle interpreter required')
 sources={str(p.relative_to(HERE)):sha(p) for p in [*HERE.glob('*.py'),*HERE.glob('*.json'),*HERE.glob('collection_tools/*.py'),*HERE.glob('tests/*.py'),*HERE.glob('tests/*.json')] if p.name not in ('freeze.json','review-packet.json','HANDOFF.json')}
 external=[];GPU=[]
 def guard(event,args):
  if event=='socket.connect':
   address=args[1]
   if not isinstance(address,tuple) or address[0] not in ('127.0.0.1','localhost','::1'):
    external.append(event);raise RuntimeError('qualification forbids external network')
  if event=='subprocess.Popen' and Path(str(args[0])).name=='nvidia-smi':GPU.append(event);raise RuntimeError('qualification forbids GPU command')
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and os.fsdecode(args[0]).startswith(('/dev/nvidia','/dev/dri','/proc/driver/nvidia')):
   GPU.append(event);raise RuntimeError('qualification forbids GPU device')
 sys.addaudithook(guard)
 os.environ['PACKET_PROOF_DIR']=str(destination/'cpu-artifacts')
 loader=unittest.TestLoader();suite=loader.discover(str(HERE/'tests'));output=io.StringIO();runner=unittest.TextTestRunner(stream=output,verbosity=2)
 tracer=trace.Trace(count=True,trace=False,ignoredirs=[sys.prefix,'/usr/lib'])
 result=tracer.runfunc(runner.run,suite)
 (destination/'TESTS.txt').write_text(output.getvalue())
 if not result.wasSuccessful():raise ValueError('changed-path qualification failed; retained TESTS.txt')
 import scientific_delta,verify_sdk,qualify_primary_size,qualify_wire,qualify_mutations
 science=scientific_delta.check();sdk=verify_sdk.qualify()
 (destination/'SCIENTIFIC-DELTA.json').write_text(json.dumps(science,indent=2)+'\n');(destination/'SDK.json').write_text(json.dumps(sdk,indent=2)+'\n')
 qualify_primary_size.main(destination/'primary-size');wire=qualify_wire.qualify(destination/'wire');mutations=qualify_mutations.qualify(destination/'isolated-RED-witnesses')
 bindings=original_bindings()
 (destination/'RISK.json').write_text(json.dumps(risk(tracer.results().counts),indent=2)+'\n')
 from kaggle.api.kaggle_api_extended import KaggleApi
 from kagglesdk.kaggle_http_client import KaggleHttpClient
 import kagglesdk.kaggle_object as objects,requests.adapters
 pinned={inspect.getfile(module):sha(inspect.getfile(module)) for module in (KaggleApi,KaggleHttpClient,objects,requests.adapters)}
 (destination/'SDK-BINDINGS.json').write_text(json.dumps(dict(files=pinned,default_HTTPAdapter_retries=requests.adapters.HTTPAdapter().max_retries.total,serializer_projection='integer microsecond projection; exact Decimal wire authority; no SDK edits'),indent=2)+'\n')
 if external or GPU:raise ValueError('forbidden qualification action attempted')
 for name,digest in sources.items():
  if sha(HERE/name)!=digest:raise ValueError('qualification changed its input '+name)
 proof=dict(status='LOCAL_EXECUTABLE_QUALIFICATION_ONLY_UNASSIGNED',tests=result.testsRun,tests_passed=True,network_attempts=external,GPU_attempts=GPU,
   source_inputs=sources,original_bindings=bindings,wire_records=wire['records'],mutations_detected=len(mutations),baseline65_rerun=False,
   SDK_intercepted_SaveKernel=sdk['intercepted_SaveKernel'],SDK_network_attempts=sdk['network_attempts'],outer_bytes=sdk['outer_bytes'],payloads=sdk['decoded_payloads'],elapsed_seconds=time.monotonic()-started)
 (destination/'PROOF.json').write_text(json.dumps(proof,indent=2)+'\n');return proof

if __name__=='__main__':
 if len(sys.argv)!=2:raise SystemExit('verify.py FRESH_ABSOLUTE_OUTPUT_DIRECTORY')
 print(json.dumps(qualify(sys.argv[1]),indent=2))
