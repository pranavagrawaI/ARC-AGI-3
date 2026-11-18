"""Unchanged receipt tests against each local proposal; CPU-only."""
import argparse,fcntl,hashlib,importlib.util,io,json,os,signal,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('variant');p.add_argument('output',type=Path);p.add_argument('--high-fd',action='store_true');a=p.parse_args();a.output.mkdir(exist_ok=False);signal.alarm(600)
sys.path.insert(0,str(ROOT/a.variant));import probe_ipc
source=ROOT.parent/'shutdown-release-prototype-r2-local-20261023/test_ipc.py';spec=importlib.util.spec_from_file_location('original_ipc_tests',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if a.high_fd:
 class High(module.ReceiptTests):
  def setUp(self):
   super().setUp();r=fcntl.fcntl(self.read,fcntl.F_DUPFD_CLOEXEC,2048);w=fcntl.fcntl(self.write,fcntl.F_DUPFD_CLOEXEC,2048);os.close(self.read);os.close(self.write);self.read=r;self.write=w
 cls=High
else:cls=module.ReceiptTests
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(cls));(a.output/'tests.log').write_text(stream.getvalue());report=dict(pass_=result.wasSuccessful(),tests=result.testsRun,high_fd=a.high_fd,variant=a.variant,source_sha256=hashlib.sha256((ROOT/a.variant/'probe_ipc.py').read_bytes()).hexdigest(),test_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),python=sys.version,source_test_unchanged=True)
(a.output/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));signal.alarm(0)
if not result.wasSuccessful():raise SystemExit(1)
