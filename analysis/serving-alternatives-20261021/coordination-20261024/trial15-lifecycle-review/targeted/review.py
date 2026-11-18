import json,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path('/home/pranav/projects/arc-agi-3/analysis/serving-alternatives-20261021/trial-15-wire-capacity-local')
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import test_capacity_lifecycle as t
from core import shutdown
import admission
OUT=Path(__file__).resolve().parent
class Extra(unittest.TestCase):
 def test_primary_replay_failure_survives_secondary_cleanup_error(self):
  r=t.CpuRunner(t.CapacityLifecycle().config(),OUT/'primary-error')
  actual=r.cleanup;primary=RuntimeError('independent replay failure');secondary=RuntimeError('independent cleanup failure')
  state={'replay_entered':False}
  def replay(*args):
   state['replay_entered']=True
   raise primary
  def cleanup(groups=None,deadline=None):
   actual(groups,deadline=deadline)
   if state['replay_entered']:raise secondary
  try:
   with patch.object(r,'candidate_gate'),patch.object(r,'boundary'),patch.object(r,'replay_and_qualify',side_effect=replay),patch.object(r,'cleanup',side_effect=cleanup):
    try:r.candidate_phase()
    except BaseException as caught:self.assertIs(caught,primary)
    else:self.fail('primary error lost')
   self.assertEqual(r.groups,{})
   self.assertEqual(len(r.cleanup_failures),1)
   self.assertEqual(r.cleanup_failures[0]['primary_error'],repr(primary))
   self.assertEqual(r.cleanup_failures[0]['error'],repr(secondary))
   self.assertFalse((r.output/'candidate-release.json').exists())
   (OUT/'primary-preservation.json').write_text(json.dumps(r.cleanup_failures,indent=2))
  finally:actual();r.trace.close()
 def test_delayed_final_identity_never_spawns_candidate_prepare(self):
  r=t.CpuRunner(t.CapacityLifecycle().config(),OUT/'stale-admission')
  r.binding={};r.candidate_admission={};r.config['admission']['maximum_age_seconds']=.01
  # Reach the actual final candidate_gate with a real expired work deadline.
  end=time.monotonic()+.01
  def delayed(_):time.sleep(.03);return {}
  try:
   with patch.object(shutdown,'owned',side_effect=delayed),patch.object(r,'check_observer'),patch('phase_tools.subprocess.Popen') as spawn:
    with self.assertRaisesRegex(ValueError,'candidate work deadline'):r.start('candidate.prepare',r.config['candidate']['prepare'],end)
    spawn.assert_not_called()
   self.assertEqual(r.groups,{})
   self.assertEqual(r.handles,[])
  finally:r.trace.close()
suite=unittest.TestSuite([t.CapacityLifecycle('test_cleanup_returning_late_even_without_helper_error_cannot_qualify'),Extra('test_primary_replay_failure_survives_secondary_cleanup_error'),Extra('test_delayed_final_identity_never_spawns_candidate_prepare')])
with (OUT/'tests.log').open('w') as log:
 result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
(OUT/'review.json').write_text(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'ok':result.wasSuccessful(),'scope':'CPU only: actual candidate server cleanup, injected primary/secondary errors, actual final admission gate'},indent=2))
print(OUT)
print((OUT/'tests.log').read_text())
raise SystemExit(not result.wasSuccessful())
