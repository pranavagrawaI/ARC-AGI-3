import ast,copy,hashlib,json,tempfile,unittest
from pathlib import Path
import packet,release_runner,transport
HERE=Path(__file__).resolve().parents[1]
class Contract(unittest.TestCase):
 def test_only_declared_candidate_identity_changes_and_engine_unchanged(self):
  prior=HERE.parent/'trial-7-serving-feasibility-guard/sources'
  for name in ('candidate_library.py','draft_vocab_32k.json','chat_template.normalized.jinja'):
   self.assertEqual((HERE/name).read_bytes(),(prior/name).read_bytes())
  old=json.loads((prior/'plan.json').read_text());new=json.loads((HERE/'candidate-plan.json').read_text());new['run_id']=old['run_id'];self.assertEqual(new,old)
  expected=(prior/'prepare.py').read_text().replace("HERE/'plan.json'","HERE/'candidate-plan.json'").replace('serving-candidate-feas7-runtime','serving-post-control-load13-runtime')
  self.assertEqual((HERE/'candidate_prepare.py').read_text(),expected)
  for name in ('control_prepare.py','control_support.py','control_notebook_setup.py','control_prefetch.py','control-argv.reference.json','capture.py','adapters.py','registration.py','runtime_scope.py','resources.py','probe_ipc.py','bootstrap_patch.py'):
   self.assertEqual((HERE/name).read_bytes(),(HERE.parent/'trial-12-bootstrap-local'/name).read_bytes(),name)
 def test_caps_no_replay_no_extra_and_metadata(self):
  s=json.loads((HERE/'packet-spec.json').read_text());c=s['config'];release_runner.validate(c,False)
  self.assertEqual(c['arm_order'],['control','candidate']);self.assertEqual((c['platform_seconds'],c['total_seconds'],c['cleanup_seconds']),(6300,6000,300))
  for mutate in (lambda v:v.update(replay={}),lambda v:v['candidate'].update(functional={}),lambda v:v.update(total_seconds=10800),lambda v:v['admission'].update(maximum_age_seconds=300),lambda v:v['candidate']['server'].update(seconds=3600)):
   value=copy.deepcopy(c);mutate(value)
   with self.assertRaises(ValueError):release_runner.validate(value,False)
  self.assertEqual(s['metadata']['dataset_sources'],['dfranzen/pennyroyal-v253','lordhansolo/vllm-main-e975732-arc3'])
  self.assertEqual(len(s['metadata']['model_sources']),3)
 def test_packet_payloads_have_no_cases_or_replay(self):
  with tempfile.TemporaryDirectory() as td:
   m=packet.prepare(Path(td)/'review');s=json.loads((HERE/'packet-spec.json').read_text());raw=transport.decode((Path(td)/'review'/s['metadata']['code_file']).read_bytes());v=transport.literals(transport.cell_source(json.loads(raw)))
   self.assertIsNone(v['ASSIGNMENT']);self.assertEqual(v['CONFIG'],s['config']);self.assertFalse(any('replay' in name or 'cases' in name for name in v['PAYLOAD']))
