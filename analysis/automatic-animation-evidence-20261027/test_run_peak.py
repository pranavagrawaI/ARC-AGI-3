import json,tempfile,unittest
from pathlib import Path
import run_pair as pair
import run_peak as peak

T0='2026-10-30T09:00:00+00:00'

class Clock:
 def __init__(self):self.t=peak.ts(T0)
 def __call__(self):return self.t
 def sleep(self,s):self.t+=s

def script(clock,schedule):
 """schedule: list of (until_offset_s, status) evaluated against elapsed time since T0."""
 def status():
  elapsed=clock()-peak.ts(T0)
  for until,value in schedule:
   if elapsed<until:
    if isinstance(value,BaseException):raise value
    return value
  return schedule[-1][1]
 return status

class Observation(unittest.TestCase):
 def test_85_minute_queue_then_healthy_run_reaches_terminal(self):
  # The closed pair's submission-based 7200 s deadline would have expired at 120 min, mid-run.
  with tempfile.TemporaryDirectory() as d:
   clock=Clock();h=Path(d)/'h.jsonl'
   status=script(clock,[(85*60,'KernelWorkerStatus.QUEUED'),(85*60+4500,'KernelWorkerStatus.RUNNING'),(10**9,'KernelWorkerStatus.COMPLETE')])
   self.assertEqual(peak.observe(status,T0,h,clock=clock,sleep=clock.sleep),'KernelWorkerStatus.COMPLETE')
   self.assertGreater(clock()-peak.ts(T0),7200)

 def test_interruption_resumes_with_original_runtime_clock(self):
  with tempfile.TemporaryDirectory() as d:
   clock=Clock();h=Path(d)/'h.jsonl'
   status=script(clock,[(600,'QUEUED'),(3000,'RUNNING'),(3001,KeyboardInterrupt()),(10**9,'RUNNING')])
   with self.assertRaises(KeyboardInterrupt):peak.observe(status,T0,h,clock=clock,sleep=clock.sleep)
   # Observer down for two hours; the runtime clock must not restart from the new invocation.
   clock.t+=7200
   with self.assertRaisesRegex(RuntimeError,'runtime-observation bound'):peak.observe(status,T0,h,clock=clock,sleep=clock.sleep)
   self.assertEqual(peak.clocks(peak.history(h),T0)[1],peak.ts(T0)+600)

 def test_queue_bound_is_separate_and_bounded(self):
  with tempfile.TemporaryDirectory() as d:
   clock=Clock();h=Path(d)/'h.jsonl'
   with self.assertRaisesRegex(RuntimeError,'queue-observation bound'):peak.observe(lambda:'QUEUED',T0,h,clock=clock,sleep=clock.sleep)
   self.assertGreaterEqual(clock()-peak.ts(T0),peak.QUEUE_CAP_S)

 def test_transient_status_errors_reauth_and_persistent_errors_stop(self):
  with tempfile.TemporaryDirectory() as d:
   clock=Clock();h=Path(d)/'h.jsonl';auths=[]
   status=script(clock,[(600,TimeoutError()),(1200,'RUNNING'),(10**9,'COMPLETE')])
   self.assertEqual(peak.observe(status,T0,h,reauth=lambda:auths.append(1),clock=clock,sleep=clock.sleep),'COMPLETE')
   self.assertTrue(auths)
   # Errors never start the runtime clock.
   self.assertEqual(peak.clocks(peak.history(h),T0)[1],peak.ts(T0)+600)
  with tempfile.TemporaryDirectory() as d:
   clock=Clock();h=Path(d)/'h.jsonl'
   def down():raise ConnectionError('api down')
   with self.assertRaisesRegex(RuntimeError,'failed repeatedly'):peak.observe(down,T0,h,clock=clock,sleep=clock.sleep)

 def test_no_harm_floor_uses_recovered_control(self):
  control=json.loads(peak.CONTROL_HEALTH.read_text())['levels']
  self.assertEqual(control,74)
  self.assertTrue(peak.no_harm(67,control)['pass']);self.assertFalse(peak.no_harm(66,control)['pass'])

 def test_slug_absence_accepts_only_denied_or_missing(self):
  class Api:
   def __init__(self,e):self.e=e
   def kernels_status(self,ref):
    if self.e:raise self.e
    return object()
  self.assertTrue(peak.slug_absent(Api(ValueError("Cannot access kernel 'x' (Permission 'kernels.get' was denied).")),'x'))
  self.assertFalse(peak.slug_absent(Api(None),'x'))
  self.assertFalse(peak.slug_absent(Api(TimeoutError('slow')),'x'))

class FakeApi:
 def kernels_status(self,ref):
  class S:status='KernelWorkerStatus.COMPLETE'
  return S()

class ResumeNeverRepushes(unittest.TestCase):
 def run_main(self,marker_state):
  gold=pair.ROOT/'solvers/m2_f1x_reap/experiment/screen-1/merge/output'
  with tempfile.TemporaryDirectory() as d:
   run=Path(d)/'run';folder=run/'peak';folder.mkdir(parents=True)
   saved=(peak.RUN,pair.RUN,pair.verify_sources,pair.api_setup,peak.push_once)
   pushes=[]
   class LW:
    @staticmethod
    def quota_used_s(api):return 0,{}
   try:
    peak.RUN=pair.RUN=run
    pair.verify_sources=lambda:'bound';pair.api_setup=lambda:(FakeApi(),LW)
    peak.push_once=lambda f:pushes.append(f)
    if marker_state:
     (folder/'launch-attempt.json').write_text(json.dumps({'utc':pair.now(),'state':marker_state}))  # real clock drives observe()
     (folder/'launch-response.json').write_text(json.dumps({'ref':'/code/pranavagrawai/x','versionNumber':1}))
     (folder/'source-readback').mkdir()
     out=folder/'output';out.mkdir()
     bench=json.loads((gold/'benchmark.json').read_text());(out/'benchmark.json').write_text(json.dumps(bench))
     (out/'serve.log').write_text('ok')
     (out/'kernel.log').write_text('GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition capability: (12, 0)\nSERVER READY at 700s')
     for g in bench['game_runs']:(out/(g['game_id']+'_p0_requests.jsonl')).write_text('{"event":"response","usage":{"completion_tokens":1}}\n')
    with self.assertRaises(SystemExit):peak.main()
    return pushes,json.loads((run/'PEAK.json').read_text())
   finally:
    peak.RUN,pair.RUN,pair.verify_sources,pair.api_setup,peak.push_once=saved

 def test_unresolved_marker_refuses_push(self):
  pushes,result=self.run_main('acceptance unresolved; never retry')
  self.assertEqual(pushes,[]);self.assertIn('refusing any push',result['reason'])

 def test_accepted_marker_resumes_to_collection_without_push(self):
  # Archived D16 traces carry no automatic images, so the flag-effectiveness gate must stop it.
  pushes,result=self.run_main('accepted')
  self.assertEqual(pushes,[]);self.assertIn('flag ineffective',result['reason'])

if __name__=='__main__':unittest.main()
