import copy,json,os,shutil,socket,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import release_runner,registration,adapters,resources,capture,host_observer
HERE=Path(__file__).resolve().parents[1]

class CpuRunner(release_runner.Runner):
 def env(self,end):
  env=super().env(end);data=json.loads((self.output/'control-package/control-prepared.json').read_text());env.update(LOCAL_CPU_ENTRY=data['argv'][0],LOCAL_CPU_ENTRY_SHA=registration.sha(data['argv'][0]));return env
 def hardware_and_queries(self):self.gpu_uuid='GPU-LOCAL-CPU';self.emit('CPU_ONLY_hardware_substitute')
 def runtime_contract(self):
  out=self.output/'control-package';self.expected=registration.contract(out/'control-prepared.json',json.loads((out/'cpu-sources.json').read_text()),self.config['run_id']);return self.expected
 def make_adapter(self,commands,paths,log,scope,end):
  root=self.output/'cpu-query-inputs';root.mkdir(exist_ok=True)
  (root/'gpu').write_text(resources.GPU_NAME+',0,GPU-LOCAL-CPU\n');(root/'apps').write_text('')
  fixture=HERE.parent/'release-adapters-r6-test-local-20261023/probe_fixture.py'
  commands={name:[sys.executable,str(fixture),'file',str(root/name)] for name in ('gpu','apps')}
  real=adapters.Adapters(self.output/'query-artifacts',self.binding,self.gpu_uuid,commands,paths,int(self.config['control']['ready_url'].split(':')[2].split('/')[0]),log,'CPU_JSON_EVENTS',end,scope=scope)
  # The host140 historical predicate is intentionally false in this fixture.
  actual=real.cgroup
  def host():
   r=actual();r.update(eligible=False,computed_bytes=10);return r
  real.cgroup=host
  real.disk=lambda:dict(eligible=True,free_bytes=100*1024**3,threshold_bytes=100*1024**3,origin='EXPLICIT_CPU_DISK_FIXTURE_NOT_HOST_CAPACITY')
  return real

class Lifecycle(unittest.TestCase):
 def config(self):
  with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
  c=copy.deepcopy(json.loads((HERE/'packet-spec.json').read_text())['config']);c.update(mode='CPU_FIXTURE_ONLY',cwd=str(HERE),run_id='CPU_POST_CONTROL_LOAD_ONLY',total_seconds=20,cleanup_seconds=3)
  for arm,script in [('control','cpu_control.py'),('candidate','cpu_candidate.py')]:
   for name in ['prepare','identity','server']:c[arm][name]=dict(argv=[sys.executable,str(HERE/'tests'/script),'serve' if name=='server' else name,str(port)],seconds=4)
   c[arm].update(ready_url=f'http://127.0.0.1:{port}/health',ready_seconds=4,request_seconds=1,ready_poll_seconds=.05)
  c['release']=dict(seconds=5,grful=2) if False else dict(seconds=5,graceful_seconds=3,fallback_reserve=1,probe_seconds=1,poll=.02)
  c['verify_seconds']=1;c['observer']['interval_seconds']=.1;c['admission']['fresh_probe_seconds']=1
  return c
 def save(self,path,label):
  if os.environ.get('PACKET_PROOF_DIR'):shutil.copytree(path,Path(os.environ['PACKET_PROOF_DIR'])/label)
 def test_actual_cpu_control_release_candidate_health_with_original_host_false(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'output';r=CpuRunner(self.config(),out).run();self.save(out,'cpu-two-loads')
   self.assertTrue(r['pass_'],r);self.assertEqual(r['qualification'],'POST_CONTROL_CANDIDATE_LOAD_TO_HEALTH_ONLY')
   self.assertTrue(r['candidate_health200']);self.assertFalse(json.loads((out/'candidate-admission.json').read_text())['original140_result'])
   self.assertEqual(r['surviving_phase_groups'],[]);self.assertEqual(r['surviving_native_group'],[]);self.assertEqual(r['retained_queries'],[])
   events=[json.loads(s) for s in (out/'supervisor.jsonl').read_text().splitlines()]
   names=[e.get('phase') for e in events if e['event']=='spawn'];self.assertEqual(names.count('candidate.prepare'),1);self.assertFalse(any('replay' in str(n) or 'functional' in str(n) for n in names))
   self.assertLess(next(e['t'] for e in events if e['event']=='release_result'),next(e['t'] for e in events if e.get('phase')=='candidate.prepare'))
 def test_observer_failure_prevents_candidate_and_invalidates(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'output'
   with patch.object(host_observer,'snapshot',side_effect=OSError('measurement persistence/input failure')):r=CpuRunner(self.config(),out).run()
   self.save(out,'cpu-measurement-failure');self.assertFalse(r['pass_']);self.assertFalse(r['candidate_health200']);self.assertIn('measurement',r['error'])
   self.assertFalse((out/'candidate-admission.json').exists())
 def test_oom_delta_is_not_silently_a_health_success(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'output';runner=CpuRunner(self.config(),out);runner.observer_spec=host_observer.bind('cpu-oom');runner.initial_events={'oom':0,'oom_kill':0}
   original=host_observer.snapshot
   def oom(*args):
    r=original(*args);r['oom_events'].update(oom=1,oom_kill=1);return r
   try:
    with patch.object(host_observer,'snapshot',side_effect=oom),self.assertRaisesRegex(ValueError,'OOM'):runner.boundary('injected_oom')
   finally:runner.trace.close()
