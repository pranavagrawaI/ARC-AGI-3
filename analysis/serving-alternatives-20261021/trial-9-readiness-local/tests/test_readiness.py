"""CPU-only lifecycle regressions; no SDK/model/GPU. Production trial8 stays immutable."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import test_supervisor as old

ROOT=Path(__file__).resolve().parents[1]
source=Path(os.environ.get('SUPERVISOR_SOURCE',str(ROOT/'supervisor.py')))
spec=importlib.util.spec_from_file_location('readiness_subject',source);S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
SERVER='''import http.server,sys,os,time
count=0
class H(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  global count
  count+=1
  mode=sys.argv[2]
  if mode=='die':os._exit(7)
  if mode=='malformed':
   self.wfile.write(b'BROKEN STATUS\\r\\n\\r\\n');self.wfile.flush();self.close_connection=True;return
  status=503 if mode=='persistent' or mode=='sequence' and count<=2 else 500 if mode=='other' else 200
  if mode=='slow200':time.sleep(.15)
  self.send_response(status);self.send_header('Content-Length','0');self.end_headers()
 def log_message(self,*args):pass
http.server.HTTPServer(('127.0.0.1',int(sys.argv[1])),H).serve_forever()
'''

class Readiness(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.config=old.fixture(self.root);(self.root/'server.py').write_text(SERVER)
  for arm in ('control','candidate'):
   self.config[arm]['ready_retry_http_statuses']=[503]
   self.config[arm]['ready_seconds']=.7;self.config[arm]['request_seconds']=.4
   self.config[arm]['server']['argv'][-1]='sequence'
 def run_case(self):
  self.s=S.Supervisor(self.config,self.root/'out');result=self.s.run()
  events=[json.loads(x) for x in (self.root/'out/supervisor.jsonl').read_text().splitlines()]
  return result,events
 def test_bounded_503_to_200_then_replay_and_release(self):
  result,events=self.run_case();self.assertTrue(result['pass_'],result)
  for arm in ('control','candidate'):
   ready=next(x for x in events if x['event']=='ready' and x['arm']==arm)
   exits=[x for x in events if x['event']=='exit' and x.get('phase','').startswith(arm+'.health-') and x['returncode']!=75]
   statuses=[json.loads((self.root/'out'/(x['phase']+'.stdout')).read_bytes().split(b'\n',1)[0])['status'] for x in exits]
   self.assertEqual(statuses,[503,503,200])
   replay=next(x for x in events if x['event']=='spawn' and x['phase']==arm+'.replay')
   self.assertGreater(replay['seq'],ready['seq'])
  self.assertEqual(result['surviving_groups'],[])
 def test_persistent_503_expires_original_startup_deadline(self):
  self.config['control']['server']['argv'][-1]='persistent'
  self.config['control']['ready_seconds']=.35
  self.config['control']['request_seconds']=25  # Must be clipped by original absolute startup end.
  result,events=self.run_case();self.assertFalse(result['pass_']);self.assertIn('deadline',result['error'])
  self.assertFalse(any(x['event']=='ready' or x.get('phase')=='control.replay' or x.get('phase','').startswith('candidate') for x in events))
  checks=[x for x in events if x['event']=='startup_not_ready'];self.assertTrue(checks)
  self.assertEqual(len({x['deadline'] for x in checks}),1)
  self.assertLess(result['elapsed_seconds'],1.5)
 def test_server_death_is_fatal(self):
  self.config['control']['server']['argv'][-1]='die'
  result,events=self.run_case();self.assertFalse(result['pass_']);self.assertRegex(result['error'],'server died|health HTTP/error')
  self.assertFalse(any(x.get('phase')=='control.replay' or x.get('phase','').startswith('candidate') for x in events))
 def test_other_http_failure_is_not_retried(self):
  self.config['control']['server']['argv'][-1]='other'
  result,events=self.run_case();self.assertFalse(result['pass_']);self.assertIn('health HTTP/error',result['error'])
  self.assertFalse(any(x['event']=='startup_not_ready' for x in events))
 def test_malformed_http_is_not_retried(self):
  self.config['control']['server']['argv'][-1]='malformed'
  result,events=self.run_case();self.assertFalse(result['pass_']);self.assertIn('health HTTP/error',result['error'])
  self.assertFalse(any(x['event']=='startup_not_ready' for x in events))
 def test_candidate_stays_blocked_when_existing_release_fails(self):
  self.config['release_check']=old.command('raise SystemExit(9)')
  result,events=self.run_case();self.assertFalse(result['pass_']);self.assertIn('control.release: exit 9',result['error'])
  self.assertFalse(any(x.get('phase','').startswith('candidate') for x in events))
 def test_original_http_failure_survives_cleanup_failure(self):
  self.config['control']['server']['argv'][-1]='other'
  s=S.Supervisor(self.config,self.root/'out');cleanup=s.cleanup;injected=[]
  def fault(groups=None):
   if groups is None and not injected:
    injected.append(True);raise ValueError('INJECTED delayed cleanup failure')
   return cleanup(groups)
  with patch.object(s,'cleanup',fault):result=s.run()
  self.assertIn('control: health HTTP/error status 1',result['error'])
  self.assertIn('INJECTED delayed cleanup failure',json.dumps(result['cleanup_failures']))
  self.assertEqual(result['surviving_groups'],[])
 def test_cleanup_waits_for_delayed_sigkill_group_disappearance(self):
  self.config['cleanup_seconds']=.4
  s=S.Supervisor(self.config,self.root/'out');pidfile=self.root/'child.pid'
  child="import os,pathlib,signal,sys,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(30)"
  parent="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]);time.sleep(30)"
  p=s.start('fixture.server',{'argv':[sys.executable,'-c',parent,child,str(pidfile)],'seconds':2},s.deadline(2))
  limit=time.monotonic()+1
  while not pidfile.exists() and time.monotonic()<limit:time.sleep(.005)
  self.assertTrue(pidfile.exists())
  kill=os.killpg;timers=[]
  def delayed(pgid,sig):
   if sig==signal.SIGKILL:
    timer=threading.Timer(.05,kill,args=(pgid,sig));timer.start();timers.append(timer)
   else:kill(pgid,sig)
  try:
   start=time.monotonic()
   with patch.object(S.os,'killpg',delayed):s.cleanup()
   self.assertEqual(S.group_members(p.pid),[])
   self.assertEqual(s.groups,{})
   self.assertLess(time.monotonic()-start,.7)
  finally:
   for timer in timers:timer.join()
   try:kill(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   p.wait(timeout=1)
   for h in s.handles:h.close()
   s.trace.close()
 def test_empty503_policy_does_not_accept_truncated_or_other_shapes(self):
  if not hasattr(S,'startup_http_not_ready'):self.skipTest('baseline has no explicit503 policy')
  capture=self.root/'health.raw'
  good={'status':503,'headers':[['Content-Length','0']],'body_limit':65536,'body_format':'raw bytes after first newline'}
  for change,body in (({'status':500},b''),({},b'partial'),({'headers':[]},b''),({'headers':[['Content-Length','0'],['Transfer-Encoding','chunked']]},b''),({'headers':[['Content-Length','0'],['Content-Length','0']]},b'')):
   capture.write_bytes(json.dumps(good|change).encode()+b'\n'+body)
   self.assertFalse(S.startup_http_not_ready(1,capture,[503]))
  capture.write_bytes(b'BadStatusLine(\'BROKEN\')\n')
  self.assertFalse(S.startup_http_not_ready(1,capture,[503]))
 def test_cleanup_stall_fails_at_bounded_deadline(self):
  self.config['cleanup_seconds']=.2
  s=S.Supervisor(self.config,self.root/'out');marker=self.root/'ready'
  code="import pathlib,signal,sys,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path(sys.argv[1]).write_text('ready');time.sleep(30)"
  p=s.start('fixture.server',{'argv':[sys.executable,'-c',code,str(marker)],'seconds':2},s.deadline(2))
  limit=time.monotonic()+1
  while not marker.exists() and time.monotonic()<limit:time.sleep(.005)
  self.assertTrue(marker.exists())
  kill=os.killpg
  def withheld(pgid,sig):
   if sig!=signal.SIGKILL:kill(pgid,sig)
  try:
   start=time.monotonic()
   with patch.object(S.os,'killpg',withheld),self.assertRaisesRegex(ValueError,'cleanup incomplete'):s.cleanup()
   self.assertLess(time.monotonic()-start,.5)
   self.assertTrue(S.group_members(p.pid));self.assertIn(p.pid,s.groups)
  finally:
   kill(p.pid,signal.SIGKILL);p.wait(timeout=1)
   for h in s.handles:h.close()
   s.trace.close()
 def test_captured_actual_empty503_requires_explicit_opt_in(self):
  capture=ROOT.parent/'trial-8-wire-capacity-local/remote-readback/archive-extracted/serving-wire-capacity8/output/control.health-615.stdout'
  self.assertTrue(S.startup_http_not_ready(1,capture,[503]))
  self.assertFalse(S.startup_http_not_ready(1,capture,[]))
 def test_cleanup_failure_without_phase_error_is_fatal_under_ambient_exception(self):
  self.config['control']['server']['argv'][-1]='ok'
  s=S.Supervisor(self.config,self.root/'out');cleanup=s.cleanup;injected=[]
  def fault(groups=None):
   if groups is None and not injected:
    injected.append(True);raise ValueError('INJECTED cleanup with no phase error')
   return cleanup(groups)
  try:raise RuntimeError('unrelated caller exception')
  except RuntimeError:
   with patch.object(s,'cleanup',fault):result=s.run()
  self.assertFalse(result['pass_'],result)
  self.assertIn('INJECTED cleanup with no phase error',result['error'])
  events=[json.loads(x) for x in (self.root/'out/supervisor.jsonl').read_text().splitlines()]
  self.assertFalse(any(x.get('phase','').startswith('candidate') for x in events))
