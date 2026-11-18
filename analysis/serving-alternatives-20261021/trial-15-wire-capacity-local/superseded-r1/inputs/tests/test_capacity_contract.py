import hashlib,json,unittest
from pathlib import Path
import release_runner
HERE=Path(__file__).resolve().parents[1];BASE=HERE.parent
class Contract(unittest.TestCase):
 def test_exact_wire_and_ownership_runtime_unchanged(self):
  for name in ('replay.py','accounting.py','load_contract.py','catalog.py','catalog.json','catalog-proof.json','corpus_base.py','corpus_prepare.py'):
   self.assertEqual((HERE/name).read_bytes(),(BASE/'trial-10-exact-wire-local'/name).read_bytes(),name)
  for name in ('capture.py','controller_shutdown.py','core.py','registration.py','runtime_scope.py','probe_ipc.py','resources.py','host_observer.py','adapters.py','control_prepare.py','control_support.py','control_notebook_setup.py','control_prefetch.py'):
   self.assertEqual((HERE/name).read_bytes(),(BASE/'trial-14-phase-handles-local'/name).read_bytes(),name)
  self.assertEqual((HERE/'release_core.py').read_bytes(),(BASE/'trial-14-phase-handles-local/release_runner.py').read_bytes())
 def test_caps_and_required_observer_horizon(self):
  spec=json.loads((HERE/'packet-spec.json').read_text());c=spec['config'];release_runner.validate(c,False)
  self.assertEqual((c['total_seconds'],c['cleanup_seconds'],c['platform_seconds']),(10200,300,10500));self.assertGreaterEqual(c['observer']['max_samples'],int(c['total_seconds']/.5)+1)
  self.assertIn('pranavagrawai/mtp-depth-replay-data-20261006',spec['metadata']['dataset_sources'])
  bad=dict(c,total_seconds=10800)
  with self.assertRaises(ValueError):release_runner.validate(bad,False)
 def test_package_and_request_identity_deltas_only(self):
  old=BASE/'trial-14-phase-handles-local'
  for name in ('candidate_library.py','draft_vocab_32k.json','chat_template.normalized.jinja','control-argv.reference.json'):self.assertEqual((HERE/name).read_bytes(),(old/name).read_bytes())
  for name in ('candidate-plan.json','control-plan.json'):
   a=json.loads((old/name).read_text());b=json.loads((HERE/name).read_text());b['run_id']=a['run_id'];self.assertEqual(a,b)
  self.assertEqual((HERE/'candidate_prepare.py').read_text(),(old/'candidate_prepare.py').read_text().replace('serving-post-control-load14-runtime','serving-wire-capacity15-runtime'))
