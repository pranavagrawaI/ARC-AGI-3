"""Actual CPU process/port orchestration; source-bound model substitutions only."""
import importlib.util,json,os,shutil,socket,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import capacity_runner,release_runner,phase_tools
HERE=Path(__file__).resolve().parents[1];OLD=HERE.parent/'trial-14-phase-handles-local'
spec=importlib.util.spec_from_file_location('previous_cpu_orchestration',OLD/'tests/test_lifecycle.py');fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
class CpuRunner(fixtures.CpuRunner):pass
class CapacityLifecycle(unittest.TestCase):
 def config(self):
  c=json.loads((HERE/'packet-spec.json').read_text())['config'];c.update(mode='CPU_FIXTURE_ONLY',cwd=str(HERE),run_id='CPU_CAPACITY_ORCHESTRATION',total_seconds=30,cleanup_seconds=3)
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  for arm,name in [('control','cpu_control.py'),('candidate','cpu_candidate.py')]:
   for phase in ('prepare','identity','server'):c[arm][phase]=dict(argv=[sys.executable,str(OLD/'tests'/name),'serve' if phase=='server' else phase,str(port)],seconds=4)
   c[arm].update(ready_url=f'http://127.0.0.1:{port}/health',ready_seconds=4,request_seconds=1,ready_poll_seconds=.05)
   for phase in ('replay','qualify'):c[arm][phase]=dict(argv=[sys.executable,str(HERE/'tests/cpu_stage.py'),arm+'.'+phase],seconds=2)
  for label in ('corpus_prepare','paired_reduce'):c[label]=dict(argv=[sys.executable,str(HERE/'tests/cpu_stage.py'),'corpus.prepare' if label=='corpus_prepare' else 'pair.reduce'],seconds=2)
  c['release']=dict(seconds=5,graceful_seconds=3,fallback_reserve=1,probe_seconds=1,poll=.02);c['admission']['fresh_probe_seconds']=1;c['verify_seconds']=1;c['final_resource_seconds']=2;c['observer']['interval_seconds']=.1
  return c
 def keep(self,out,name):
  if os.environ.get('PACKET_PROOF_DIR'):shutil.copytree(out,Path(os.environ['PACKET_PROOF_DIR'])/name)
 def test_actual_two_cpu_servers_replay_release_and_observer_complete(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'output';result=CpuRunner(self.config(),out).run();self.keep(out,'cpu-capacity-success');self.assertTrue(result['pass_'],result)
   e=[json.loads(s) for s in (out/'supervisor.jsonl').read_text().splitlines()];self.assertEqual(result['qualification'],'CLOSE_TESTED_CONFIG')
   names=[r['phase'] for r in e if r['event']=='spawn'];self.assertEqual(names.count('candidate.prepare'),1);self.assertIn('control.replay',names);self.assertIn('candidate.replay',names)
   release=next(r['t'] for r in e if r['event']=='release_start');control_qual=next(r['t'] for r in e if r.get('phase')=='control.qualify' and r['event']=='exit');self.assertLess(control_qual,release)
   self.assertEqual(result['surviving_phase_groups'],[]);self.assertEqual(result['surviving_native_group'],[]);self.assertEqual(result['retained_queries'],[])
   self.assertTrue((out/'host-series.result.json').exists());self.assertEqual(len([r for r in e if r['event']=='host_boundary' and r['label']=='final_cleanup']),1)
 def test_failed_control_qualification_never_admits_candidate(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'output';c=self.config();c['control']['qualify']['env']={'CPU_FAIL_PHASE':'control.qualify'};result=CpuRunner(c,out).run();self.keep(out,'cpu-control-failure')
   self.assertFalse(result['pass_']);self.assertIn('exit 9',result['error']);e=[json.loads(s) for s in (out/'supervisor.jsonl').read_text().splitlines()];self.assertFalse(any(r.get('phase')=='candidate.prepare' for r in e))
 def test_delayed_candidate_cleanup_cannot_qualify_or_pair_reduce(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'output';c=self.config();c['final_resource_seconds']=.05;runner=CpuRunner(c,out);actual=runner.cleanup
   def delayed(groups=None,deadline=None):
    if deadline is not None:time.sleep(.08)
    return actual(groups,deadline=deadline)
   with patch.object(runner,'cleanup',side_effect=delayed):result=runner.run()
   self.keep(out,'cpu-late-candidate-cleanup');self.assertFalse(result['pass_']);self.assertIn('deadline',result['error']);self.assertFalse((out/'paired-result.json').exists())
 def test_cleanup_returning_late_even_without_helper_error_cannot_qualify(self):
  # Isolate the candidate cleanup boundary from unrelated observer/proc races.
  # Real CPU candidate process+HTTP200+whole-group cleanup remain in the path.
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'output';c=self.config();c['final_resource_seconds']=.05;runner=CpuRunner(c,out);actual=runner.cleanup;delayed=[]
   def ignores_phase(groups=None,deadline=None):
    if deadline is not None:delayed.append(deadline);time.sleep(.08)
    return actual(groups)
   try:
    with patch.object(runner,'candidate_gate'),patch.object(runner,'boundary'),patch.object(runner,'replay_and_qualify'),patch.object(runner,'cleanup',side_effect=ignores_phase):
     with self.assertRaisesRegex(ValueError,'late candidate cleanup'):runner.candidate_phase()
    self.assertEqual(len(delayed),1);self.assertFalse((out/'paired-result.json').exists());self.assertEqual(runner.groups,{})
   finally:
    runner.cleanup();runner.trace.close();self.keep(out,'cpu-late-cleanup-return')
 def test_persistent_final_gpu_resource_failure_never_reaches_pair(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'output';c=self.config();c['final_resource_seconds']=.35;runner=CpuRunner(c,out);actual=runner.make_adapter;calls=[]
   def bad_final(*args):
    adapter=actual(*args);calls.append(1)
    if len(calls)==2:
     file=out/'cpu-query-inputs/gpu';name,_,uuid=file.read_text().strip().split(',');file.write_text(f'{name},5000,{uuid}\n')
    return adapter
   with patch.object(runner,'make_adapter',side_effect=bad_final):result=runner.run()
   self.keep(out,'cpu-persistent-final-resource');self.assertFalse(result['pass_']);self.assertFalse((out/'paired-result.json').exists());receipt=json.loads((out/'candidate-release.json').read_text());self.assertFalse(receipt['pass_']);self.assertTrue(receipt['error'])
