"""Public phase lifecycle contracts: logs/errors/active ownership/cleanup."""
import hashlib,io,json,os,signal,sys,tempfile,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import phase_tools
class Lifecycle(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();c=json.loads((ROOT/'packet-spec.json').read_text())['config'];c.update(mode='CPU_FIXTURE_ONLY',cwd=str(ROOT),total_seconds=10,cleanup_seconds=2)
  self.output=Path(self.tmp.name)/'out';self.t=phase_tools.PhaseTools(c,self.output)
 def tearDown(self):
  try:self.t.cleanup()
  finally:
   for h in self.t.handles:h.close()
   self.t.trace.close();self.tmp.cleanup()
 def spec(self,code):return dict(argv=[sys.executable,'-c',code],seconds=2)
 def test_completed_stdout_stderr_preserved_and_closed(self):
  self.t.command('cpu.step',self.spec("import sys;print('OK');print('ERR',file=sys.stderr)"))
  self.assertEqual((self.output/'cpu.step.stdout').read_text(),'OK\n');self.assertEqual((self.output/'cpu.step.stderr').read_text(),'ERR\n')
  self.assertEqual(self.t.handles,[]);self.assertEqual(self.t.phase_handles,{});self.assertEqual(self.t.groups,{})
 def test_live_server_handles_survive_other_phase_until_group_cleanup(self):
  p=self.t.start('cpu.server',self.spec("import time;print('LIVE',flush=True);time.sleep(20)"),self.t.deadline(4));pair=self.t.phase_handles[p.pid]
  self.t.command('cpu.step',self.spec("print('OTHER')"));self.assertIsNone(p.poll());self.assertTrue(all(not h.closed for h in pair));self.assertEqual(len(self.t.handles),2)
  self.t.cleanup([p.pid]);self.assertTrue(all(h.closed for h in pair));self.assertEqual(self.t.groups,{})
  self.assertEqual((self.output/'cpu.server.stdout').read_text(),'LIVE\n')
 def test_repeated_whole_cleanup_does_not_mask_or_resurrect_streams(self):
  self.t.command('cpu.step',self.spec("print('DONE')"));self.t.cleanup();self.t.cleanup();self.assertEqual(self.t.handles,[]);self.assertEqual(self.t.phase_handles,{})
 def test_nonzero_exit_keeps_original_error_and_raw_logs(self):
  with self.assertRaisesRegex(ValueError,'cpu.fail: exit 7'):self.t.command('cpu.fail',self.spec("print('BEFORE');raise SystemExit(7)"))
  self.assertEqual((self.output/'cpu.fail.stdout').read_text(),'BEFORE\n');self.assertEqual(self.t.handles,[]);self.assertEqual(self.t.phase_handles,{});self.assertEqual(self.t.groups,{})
 def test_injected_spawn_failure_keeps_original_exception_and_releases_streams(self):
  missing=str(self.output/'not-an-executable')
  from unittest.mock import patch
  failure=FileNotFoundError(2,'CPU fixture missing executable',missing)
  with patch.object(phase_tools.subprocess,'Popen',side_effect=failure),self.assertRaises(FileNotFoundError) as c:self.t.command('cpu.missing',dict(argv=[missing],seconds=1))
  self.assertIs(c.exception,failure)
  self.assertEqual(c.exception.filename,missing);self.assertEqual(self.t.handles,[]);self.assertEqual(self.t.phase_handles,{});self.assertEqual(self.t.groups,{})
  self.assertEqual((self.output/'cpu.missing.stdout').read_bytes(),b'')
 def test_second_open_failure_preserves_existing_file_and_closes_first(self):
  (self.output/'cpu.open.stderr').write_text('KEEP')
  error=None
  try:self.t.start('cpu.open',self.spec("print('NEVER')"),self.t.deadline(1))
  except FileExistsError as e:error=e
  self.assertIsNotNone(error);self.assertEqual((self.output/'cpu.open.stderr').read_text(),'KEEP')
  links=[]
  for p in Path('/proc/self/fd').iterdir():
   try:links.append(os.readlink(p))
   except OSError:pass
  self.assertNotIn(str(self.output/'cpu.open.stdout'),links);self.assertEqual(self.t.handles,[])
 def test_readiness_http_failure_preserves_status_body_and_live_server_handles(self):
  # A single failed helper through the real ready path, no model or native engine.
  p=self.t.start('cpu.server',self.spec("import time;time.sleep(20)"),self.t.deadline(4));pair=self.t.phase_handles[p.pid]
  import socket,threading
  from http.server import BaseHTTPRequestHandler,HTTPServer
  class Handler(BaseHTTPRequestHandler):
   def do_GET(self):self.send_response(500);self.send_header('Content-Length','3');self.end_headers();self.wfile.write(b'BAD')
   def log_message(self,*a):pass
  server=HTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever);thread.start()
  try:
   a=dict(ready_url=f'http://127.0.0.1:{server.server_port}/health',ready_seconds=1,request_seconds=.5,ready_poll_seconds=.01,ready_retry_http_statuses=[503])
   with self.assertRaisesRegex(ValueError,'health HTTP/error status 1'):self.t.ready('control',a,p,self.t.deadline(2))
   data=(self.output/'control.health-0.stdout').read_bytes();header,body=data.split(b'\n',1);self.assertEqual(json.loads(header)['status'],500);self.assertEqual(body,b'BAD')
   self.assertTrue(all(not h.closed for h in pair));self.assertEqual(len(self.t.handles),2)
  finally:server.shutdown();thread.join();server.server_close()

 def test_candidate_gate_rejection_retains_exact_error_and_live_streams(self):
  from unittest.mock import patch
  active=self.t.start('cpu.server',self.spec("import time;time.sleep(20)"),self.t.deadline(4));pair=self.t.phase_handles[active.pid]
  sentinel=ValueError('EXACT_ADMISSION_REJECTION')
  with patch.object(self.t,'candidate_gate',side_effect=sentinel,create=True),self.assertRaises(ValueError) as caught:
   self.t.start('candidate.prepare',self.spec("raise RuntimeError('NEVER')"),self.t.deadline(1))
  self.assertIs(caught.exception,sentinel);self.assertTrue(all(not h.closed for h in pair));self.assertEqual(self.t.handles,list(pair))
  self.assertEqual(set(self.t.groups),{active.pid});self.assertEqual(set(self.t.phase_handles),{active.pid})
