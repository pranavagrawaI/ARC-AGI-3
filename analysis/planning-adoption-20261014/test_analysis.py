"""Separate attempted helper names, persisted outcomes and reasoning-only cutoffs."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from analyze import mechanisms,receipt_integrity,core


class MechanismTests(unittest.TestCase):
    def test_native_replay_preserves_live_level_reset_policy(self):
        receipt=json.loads((Path(__file__).parent/'evidence/audit-reset-policy.json').read_text())
        actions=receipt['prefix_actions']+receipt['appended_actions']
        with patch.dict(os.environ,{'ONLY_RESET_LEVELS':'false'}):
            audit=core()
            self.assertEqual(audit.native_level_count(actions),1)
            os.environ['ONLY_RESET_LEVELS']='false'
            self.assertEqual(audit.native_level_count(actions),0)

    def test_source_and_action_receipts_cannot_rely_on_worker_success_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            identity={'limits':{'tokens':32768,'model_calls':32,'native_actions':150,'episode_seconds':900,'request_seconds':120},
                      'variant':'planner','protected_files':{'/remote/planning-package/sources/b1/actor.py':'frozen',
                                                            '/native/s5i5/18d95033/s5i5.py':'engine'}}
            result={'variant':'planner','checkpoint':'b1','new_native_actions':1,'protected_inputs_unchanged':True}
            def write():
                for name,value in [('identity',identity),('result',result),('checkpoint',{'prefix_actions':[{}]}),('actions',[{},{}])]:
                    (root/(name+'.json')).write_text(json.dumps(value))
            manifest={'sources/b1/actor.py':'frozen'}
            write();self.assertEqual(receipt_integrity(root,manifest,'engine'),[])
            identity['protected_files']['/remote/planning-package/sources/b1/actor.py']='changed'
            write();self.assertIn('protected source differs',receipt_integrity(root,manifest,'engine')[0])
            identity['protected_files']['/remote/planning-package/sources/b1/actor.py']='frozen'
            del identity['protected_files']['/native/s5i5/18d95033/s5i5.py']
            write();self.assertIn('incomplete protected source coverage',receipt_integrity(root,manifest,'engine'))
            identity['protected_files']['/native/s5i5/18d95033/s5i5.py']='engine'
            result['new_native_actions']=151
            write();self.assertIn('native action budget/accounting mismatch',receipt_integrity(root,manifest,'engine'))

    def test_attempts_are_distinct_from_saved_execution_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result={'saved_data':{'probe':{'value':{'predicted':{},'before':{},'receipt':{'executed':True},'matched':False,'status':'MISMATCH'}},
                                 'plan':{'value':{'stored_nodes':7,'depth_frontier_states':2,'expanded':4,'status':'RESOURCE_LIMIT'}}}}
            (root/'result.json').write_text(json.dumps(result))
            def response(code,usage=3):
                return {'event':'response','body':{'usage':{'completion_tokens':usage},'choices':[{'finish_reason':'tool_calls',
                    'message':{'tool_calls':[{'function':{'name':'python','arguments':json.dumps({'code':code})}}]}}]}}
            events=[response('r=bounded_search(0, successors, state_key=key, is_goal=goal)'),
                    response('def bounded_search(x): return x\nr=bounded_search(0)'),
                    response("probe_transition('UP',observe=o,predict=p,state_key=k,name='p')"),
                    {'event':'response','body':{'usage':{'completion_tokens':12},'choices':[{'finish_reason':'length','message':{'content':'thinking','tool_calls':[]}}]}}]
            (root/'wire.jsonl').write_text('\n'.join(json.dumps(row) for row in events))
            report=mechanisms(root)
            self.assertEqual(report['attempted_name_calls'],{'bounded_search':1,'probe_transition':1})
            self.assertEqual(report['persisted_probe_records'][0]['status'],'MISMATCH')
            self.assertEqual(report['persisted_search_records'][0]['status'],'RESOURCE_LIMIT')
            self.assertEqual(report['length_cutoff_no_tool_tokens'],12)


if __name__=='__main__':unittest.main()
