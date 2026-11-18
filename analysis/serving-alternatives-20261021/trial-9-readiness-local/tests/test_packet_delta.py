"""Independent allowed delta checks: candidate/science/caps cannot drift during health repair."""
import copy,json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'trial-8-wire-capacity-local'
class PacketDelta(unittest.TestCase):
 def test_only_declared_config_identity_and_control_health_changes(self):
  old=json.loads((OLD/'packet-spec.json').read_text());new=json.loads((ROOT/'packet-spec.json').read_text())
  normalized=json.loads(json.dumps(new['config']).replace('serving-wire-capacity9','serving-wire-capacity8'))
  actual=normalized['control'];self.assertEqual(actual.pop('ready_retry_http_statuses'),[503]);self.assertEqual(actual['request_seconds'],25);actual['request_seconds']=10
  for phase in ('identity','server'):
   self.assertEqual(actual[phase].pop('env'),{'SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION':'1','SGLANG_HEALTH_CHECK_TIMEOUT':'20'})
  self.assertEqual(normalized,old['config'])
  self.assertEqual(json.loads(json.dumps(new['metadata']).replace('serving-wire-capacity9','serving-wire-capacity8')),old['metadata'])
 def test_candidate_corpus_reducer_and_accepted_supervisor_are_exact(self):
  for name in ('candidate_prepare.py','candidate_library.py','draft_vocab_32k.json','chat_template.normalized.jinja','catalog.json','catalog-proof.json','catalog.py','replay.py','reduce_api.py','load_contract.py','accounting.py','release_check.py','http_once.py','control_notebook_setup.py','control_prefetch.py','control-argv.reference.json'):
   self.assertEqual((ROOT/name).read_bytes(),(OLD/name).read_bytes(),name)
  self.assertEqual((ROOT/'supervisor.py').read_bytes(),(ROOT/'accepted-local-b48996d7/supervisor.py').read_bytes())
  a=json.loads((ROOT/'candidate-plan.json').read_text());a['run_id']='serving-wire-capacity8-20261023';self.assertEqual(a,json.loads((OLD/'candidate-plan.json').read_text()))
  a=json.loads((ROOT/'capacity-contract.json').read_text());b=json.loads((OLD/'capacity-contract.json').read_text());a['config_sha256']=b['config_sha256'];self.assertEqual(a,b)
 def test_health_settings_and_probe_budget_remain_inside_existing_startup(self):
  c=json.loads((ROOT/'packet-spec.json').read_text())['config']['control'];p=json.loads((ROOT/'control-plan.json').read_text())['health_policy']
  self.assertEqual(c['identity']['env'],p['raw']);self.assertEqual(c['server']['env'],p['raw']);self.assertEqual(c['request_seconds'],p['effective']['timeout_s']+5)
  self.assertEqual(c['ready_seconds'],2700);self.assertEqual(c['server']['seconds'],2700)
