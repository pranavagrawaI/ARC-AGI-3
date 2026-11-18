import copy, hashlib, json, os, sys, tempfile, unittest
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from integration import FixtureTransport, IntegrationHost, ProtocolError, load_case
from request_contract import validate_request

CANDIDATE='''
def _obs(o):
 g=o["grid"]; cells=[(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v==12]
 return [min(r for r,c in cells),min(c for r,c in cells)],next(c for c,v in enumerate(g[61]) if v==11)
def initialize(observation):
 b,m=_obs(observation); return {"block":b,"bar_left":m}
def predict(state,action):
 dr,dc={"UP":(-5,0),"DOWN":(5,0),"LEFT":(0,-5),"RIGHT":(0,5)}.get(action,(0,0))
 n={"block":[state["block"][0]+dr,state["block"][1]+dc],"bar_left":state["bar_left"]+1}
 return [{"state":n,"commitments":dict(n),"declared_dependencies":["block","bar_left"]}]
def assimilate(predicted_state,observation,verdict):
 b,m=_obs(observation); return {"block":b,"bar_left":m}
'''
WORK={"decision":"next move","obstacle":"transition uncertainty","operation":"test","claim":"joint transition","evidence_refs":[1,2],"dependencies":["block","bar_left"],"completion_checks":[1,2],"remaining_scope":"outside trace","reasoning_need":"needs_computation"}

def call(i,name,args):
 return {"id":i if isinstance(i,str) else f"call-{i}","type":"function","function":{"name":name,"arguments":json.dumps(args)}}
def response(calls,p=100,c=30):
 return {"id":"fixture","provider":"fixture/provider","choices":[{"message":{"role":"assistant","content":None,"tool_calls":calls}}],"usage":{"prompt_tokens":p,"completion_tokens":c,"completion_tokens_details":{"reasoning_tokens":0}}}

class IntegrationTests(unittest.TestCase):
 def run_host(self,responses):
  with tempfile.TemporaryDirectory() as d:
   h=IntegrationHost(load_case(),FixtureTransport(responses),Path(d)); result=h.run(); return result,Path(d)
 def test_full_two_inspections_candidate_methods_freeze_one_suffix(self):
  rs=[response([call(1,"execute_python",{"code":"import json; d=json.load(open('/evidence/case.json')); print(len(d['visible_transitions']))"})]),response([call(2,"execute_python",{"code":"import json; d=json.load(open('/evidence/case.json')); print(d['frontier'])"})]),response([call(3,"submit_candidate",{"work_item":WORK,"candidate_source":CANDIDATE})])]
  with tempfile.TemporaryDirectory() as d:
   out=Path(d); result=IntegrationHost(load_case(),FixtureTransport(rs),out).run()
   self.assertEqual(result['classification'],'INTEGRATION_VALID_CANDIDATE_SUPPORTED')
   self.assertEqual(result['inspections'],2); self.assertEqual(result['suffix_assessments'],1)
   tool_events=[json.loads(x) for x in (out/'tool-events.jsonl').read_text().splitlines()]
   inspection_payloads=[json.loads(x['rendered_result_message']['content']) for x in tool_events[:2]]
   self.assertEqual([x['status'] for x in inspection_payloads],['completed','completed'])
   self.assertEqual([x['stdout'].strip() for x in inspection_payloads],['5','5'])
   self.assertGreaterEqual(len((out/'lifecycle.jsonl').read_text().splitlines()),5)
   frozen=json.loads((out/'FROZEN.json').read_text()); self.assertEqual(frozen['model_sha256'],hashlib.sha256(CANDIDATE.encode()).hexdigest())
   self.assertEqual((out/'frozen_candidate.py').read_text(),CANDIDATE)
   methods=[x['method'] for x in json.loads((out/'sealed_suffix_full_evaluation.json').read_text())['rpc_methods']]
   self.assertEqual(methods,['initialize']+['predict','assimilate']*5+['predict']*3)
 def test_duplicate_tool_id_fails_before_second_execution(self):
  rs=[response([call(1,"execute_python",{"code":"print('one')"})]),response([call(1,"execute_python",{"code":"print('two')"})])]
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex((ProtocolError,ValueError),'duplicate'):
    IntegrationHost(load_case(),FixtureTransport(rs),Path(d)).run()
   events=[json.loads(x) for x in (Path(d)/'tool-events.jsonl').read_text().splitlines()]
   self.assertEqual(len(events),1)
 def test_opaque_ids_are_preserved_and_dispatched_in_response_order(self):
  bad=response([call("call_abc123","execute_python",{"code":"print(2)"}),call("opaque/Z9","execute_python",{"code":"print(1)"})])
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ProtocolError,'exhausted'):
    IntegrationHost(load_case(),FixtureTransport([bad]),Path(d)).run()
   events=[json.loads(x) for x in (Path(d)/'tool-events.jsonl').read_text().splitlines()]
   self.assertEqual([event['call']['id'] for event in events],["call_abc123","opaque/Z9"])
 def test_missing_usage_and_oversize_fail_closed(self):
  missing=response([call(1,"request_evidence",{"obstacle":"x","needed_observation":"y","decision_relevance":"z"})]); missing.pop('usage')
  with tempfile.TemporaryDirectory() as d:
   self.assertEqual(IntegrationHost(load_case(),FixtureTransport([missing]),Path(d)).run()['construction_status'],'INVALID_USAGE')
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ProtocolError): IntegrationHost(load_case(),FixtureTransport([response([call(1,"execute_python",{"code":"x"*32001})])]),Path(d)).run()
   event=json.loads((Path(d)/'tool-events.jsonl').read_text().splitlines()[0]); self.assertIn('32,000',event['rendered_result_message']['content'])
 def test_provider_config_drift_rejected(self):
  h=IntegrationHost(load_case(),FixtureTransport([]),Path(tempfile.mkdtemp()))
  body=h.next_request([]); body['provider']['allow_fallbacks']=True
  with self.assertRaisesRegex(ValueError,'routing drift'): validate_request(body)
 def test_transport_never_executes_tool(self):
  r=response([call(1,"request_evidence",{"obstacle":"x","needed_observation":"y","decision_relevance":"z"})])
  t=FixtureTransport([r]); body={'sentinel':'host dispatch only'}
  self.assertEqual(t(body),r); self.assertEqual(t.executed_tools,0)
 def test_candidate_cannot_see_host_canaries_or_prefix(self):
  attack=CANDIDATE+'''\ndef initialize(observation):\n import os\n denied=[]\n for p in ("/evidence/case.json","/work/evaluator_payload.json","/work/suffix.json","/work/verifier.py","/proc/1/environ"):\n  try: open(p).read(); denied.append(p)\n  except OSError: pass\n return {"env":dict(os.environ),"opened":denied,"stdin":os.read(0,10).decode()}\n'''
  case=load_case()
  with tempfile.TemporaryDirectory() as d:
   value=__import__('integration_boundary').SealedBoundary().invoke(attack,'initialize',[{"grid":case.visible[0]['before']['grid']}])
   self.assertEqual(value['stdin'],''); self.assertEqual(value['opened'],[]); self.assertNotIn('SYNTHETIC_SECRET',value['env'])
 def test_pre_freeze_only_and_no_assimilation_in_suffix(self):
  source=CANDIDATE.replace('def assimilate(', 'def assimilate(')
  rs=[response([call(1,"submit_candidate",{"work_item":WORK,"candidate_source":source})])]
  with tempfile.TemporaryDirectory() as d:
   out=Path(d); IntegrationHost(load_case(),FixtureTransport(rs),out).run()
   events=[json.loads(x) for x in (out/'lifecycle.jsonl').read_text().splitlines()]
   freeze=next(i for i,e in enumerate(events) if e['event']=='frozen')
   self.assertFalse(any(e['event']=='tool_dispatch' for e in events[freeze+1:]))
   self.assertEqual([e['event'] for e in events].count('suffix_assessment'),1)
 def test_chronological_replay_and_ancestry(self):
  case=load_case(); self.assertEqual([x['step'] for x in case.visible],[1,2,3,4,5]); self.assertEqual([x['step'] for x in case.suffix],[6,7,8])
 def test_malformed_tool_args_fail_without_execution(self):
  r=response([{"id":"call-1","type":"function","function":{"name":"execute_python","arguments":"{"}}])
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ProtocolError,'arguments'): IntegrationHost(load_case(),FixtureTransport([r]),Path(d)).run()
   event=json.loads((Path(d)/'tool-events.jsonl').read_text().splitlines()[0]); self.assertIn('rejected',event['rendered_result_message']['content'])

if __name__=='__main__': unittest.main()
