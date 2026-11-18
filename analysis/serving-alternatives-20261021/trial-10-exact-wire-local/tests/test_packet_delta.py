"""Only identity, exact-byte reference/binding and accepted cutoff correction change."""
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'trial-9-readiness-local'
class PacketDelta(unittest.TestCase):
 def test_config_metadata_identity_only(self):
  a=json.loads((ROOT/'packet-spec.json').read_text());b=json.loads((OLD/'packet-spec.json').read_text())
  self.assertEqual(json.loads(json.dumps(a['config']).replace('serving-wire-capacity10','serving-wire-capacity9')),b['config'])
  self.assertEqual(json.loads(json.dumps(a['metadata']).replace('serving-wire-capacity10','serving-wire-capacity9')),b['metadata'])
 def test_candidate_control_runtime_sources_and_science_exact(self):
  for name in ('candidate_prepare.py','candidate_library.py','draft_vocab_32k.json','chat_template.normalized.jinja','load_contract.py','accounting.py','release_check.py','http_once.py','supervisor.py','control_prepare.py','control_notebook_setup.py','control_prefetch.py','control-argv.reference.json','corpus_base.py','corpus_prepare.py','provenance.py'):
   self.assertEqual((ROOT/name).read_bytes(),(OLD/name).read_bytes(),name)
  for name in ('control-plan.json','candidate-plan.json'):
   a=json.loads((ROOT/name).read_text());a['run_id']='serving-wire-capacity9-20261023';self.assertEqual(a,json.loads((OLD/name).read_text()))
  a=json.loads((ROOT/'capacity-contract.json').read_text());b=json.loads((OLD/'capacity-contract.json').read_text());a['config_sha256']=b['config_sha256'];self.assertEqual(a,b)
 def test_exact_accepted_catalog(self):
  self.assertEqual((ROOT/'catalog.json').read_bytes(),(ROOT.parent/'trial9-count-reviewed-r1-20261023/catalog.exact-wire-v2.json').read_bytes())
