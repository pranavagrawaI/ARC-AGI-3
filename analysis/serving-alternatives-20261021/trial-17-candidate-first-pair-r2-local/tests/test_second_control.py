import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import control_prepare,pair_evidence
HERE=Path(__file__).resolve().parents[1]
NATIVE=HERE.parent/'trial-14-phase-handles-local/remote-readback/archive-v1-kernel140449504/members/serving-post-control-load14/output'
class Control(unittest.TestCase):
 def test_prepared_second_environment_clears_candidate_pythonpath_and_keeps_health1_20(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);value=dict(argv=['PINNED_SHIPPING_ARGV'],python='/tmp/sgl-intel/venv/bin/python',env_overrides=control_prepare.PLAN['health_policy']['raw'])
   (root/'control-prepared.json').write_text(json.dumps(value))
   with patch.dict(os.environ,dict(SCREEN_OUTPUT=str(root),PYTHONPATH='/tmp/candidate-untrusted-imports')):state=control_prepare.state()
   self.assertNotIn('PYTHONPATH',state['env']);self.assertEqual(state['argv'],value['argv'])
   self.assertEqual(state['env']['SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION'],'1');self.assertEqual(state['env']['SGLANG_HEALTH_CHECK_TIMEOUT'],'20')
 def test_actual_shipping_health_source_and_effective_environment(self):
  record=json.loads((NATIVE/'control-package/control-health-serve.json').read_bytes());observed=control_prepare.verify_health_record(record,'serve')
  self.assertEqual(observed['effective'],dict(generation_enabled=True,timeout_s=20))
  for key,value in [('generation_enabled',False),('timeout_s',19)]:
   wrong=copy.deepcopy(record);wrong['effective'][key]=value
   with self.assertRaises(ValueError):control_prepare.verify_health_record(wrong,'serve')
 def test_actual_shipping_preparation_gpu_and_total_capacity_raw(self):
  trace=[json.loads(row) for row in (NATIVE/'supervisor.jsonl').read_text().splitlines()];hardware=next(row for row in trace if row['event']=='hardware_pass')
  self.assertTrue(pair_evidence.preparation_hardware(NATIVE,'control',hardware))
