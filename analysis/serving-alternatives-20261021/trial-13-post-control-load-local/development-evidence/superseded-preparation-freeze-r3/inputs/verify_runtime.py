"""Run all accepted65 regressions against THIS packet's actual runtime modules."""
import argparse,hashlib,importlib.util,io,json,os,signal,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'release-adapters-r6-test-local-20261023'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
 out=a.output.resolve();out.mkdir(exist_ok=False);signal.alarm(600);started=time.monotonic()
 os.environ.update(ADAPTER_PROOF_DIR=str(out/'cases'),SHUTDOWN_PROOF_DIR=str(out/'core-cases'),LOCAL_ADAPTER_PROOF='1',PYTHONPATH=str(BASE/'guard'))
 sys.path.insert(0,str(BASE/'guard'));import sitecustomize
 import core,adapters,capture,artifacts,command_child,no_processes,registration,runtime_scope,resources,probe_ipc
 for module in (adapters,capture,artifacts,command_child,no_processes,registration,runtime_scope,resources,probe_ipc,core.shutdown):
  assert Path(module.__file__).resolve().parent==HERE,module.__file__
 spec=importlib.util.spec_from_file_location('prior_qualification',BASE/'verify.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior);prior.prior()
 fixture_root=Path(json.loads((BASE/'ACCEPTED-CORE.json').read_text())['directory'])
 sys.modules['shutdown']=core.shutdown;sys.path.append(str(fixture_root))
 suite=unittest.TestSuite();loaded=[]
 for name,path in [('test_adapters',BASE/'test_adapters.py')]+[(name,fixture_root/(name+'.py')) for name in ('test_shutdown','test_resources','test_ipc')]:
  spec=importlib.util.spec_from_file_location('current_runtime_'+name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  if hasattr(module,'ROOT'):module.ROOT=fixture_root
  if hasattr(module,'shutdown'):module.shutdown=core.shutdown
  suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module));loaded.append(dict(path=str(path),sha256=sha(path)))
 stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 (out/'tests.log').write_text(stream.getvalue());print(stream.getvalue(),file=sys.stderr)
 if not result.wasSuccessful():raise RuntimeError('actual packet runtime regression failure; retained tests/receipts')
 checked=0
 for case in (out/'cases').iterdir():
  for ref in prior.refs(json.loads((case/'RESULT.json').read_text())):
   artifacts.verify(case/'raw',ref,time.monotonic()+2);checked+=1
 prior.prior()
 report=dict(status='CURRENT_PACKET_RUNTIME_CPU_REGRESSIONS_ONLY',tests=result.testsRun,seconds=time.monotonic()-started,artifact_references_checked=checked,
  runtime_sources={name:sha(HERE/name) for name in json.loads((HERE/'runtime-pins.json').read_text())},accepted_test_sources=loaded,
  actual_runtime_modules_bound=True,external_network=0,gpu_queries=0,
  inventory={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
 with (out/'PROOF.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({k:report[k] for k in ('status','tests','seconds','artifact_references_checked')}));signal.alarm(0)
if __name__=='__main__':main()
