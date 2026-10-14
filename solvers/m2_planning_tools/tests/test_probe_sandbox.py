"""Probe persistence, ordering and capability boundaries in the real subprocess."""
import copy
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'm2_b1_fixed/tests'))
from test_basics import SandboxCase, STATE

CALLBACKS = '''
def observe(frame): return {'n':frame.grid[0][0]}
def predict(before, action): return {'n':before['n']+1}
def key(state): return state['n']
'''
CALL = "probe_transition('UP', observe=observe, predict=predict, state_key=key, name='probe_1')"


class ProbeSandboxTests(SandboxCase):
    def setUp(self):
        super().setUp()
        self.variant = os.environ.get('ARC3_PLANNING_VARIANT')
        os.environ['ARC3_PLANNING_VARIANT'] = 'both'

    def tearDown(self):
        if self.variant is None:
            os.environ.pop('ARC3_PLANNING_VARIANT', None)
        else:
            os.environ['ARC3_PLANNING_VARIANT'] = self.variant

    def test_host_has_prediction_before_action_and_keeps_mismatch(self):
        store = {}
        def handler(actions, **kwargs):
            self.assertEqual(store['probe_1']['value']['status'], 'PREDICTED')
            self.assertEqual(store['probe_1']['value']['predicted'], {'n':1})
            return self.handler(actions, **kwargs)
        result = self.run_code(CALLBACKS + 'r=' + CALL + "\nassert r['status']=='MISMATCH'", saved_data=store, handler=handler)
        self.assertFalse(result['error'], result)
        self.assertEqual(len(self.receipts),1)
        self.assertEqual(store['probe_1']['value']['actual'], {'n':0})
        self.assertFalse(store['probe_1']['value']['matched'])

    def test_match_uses_refreshed_frame(self):
        state = copy.deepcopy(STATE); state['current_frame']['grid'][0][0] = 1
        def handler(actions, **kwargs):
            row = self.handler(actions, **kwargs); row['state'] = state; return row
        result = self.run_code(CALLBACKS + "r=" + CALL + "\nassert r['status']=='MATCH'", handler=handler)
        self.assertFalse(result['error'],result)

    def test_prediction_cannot_act_and_exception_releases_guard(self):
        code = CALLBACKS + '''
def bad_predict(before, proposed):
    action('UP')
    return {'n':1}
try:
    probe_transition('UP', observe=observe, predict=bad_predict, state_key=key, name='bad')
except RuntimeError as error:
    assert 'must not execute' in str(error)
action('UP')
'''
        result = self.run_code(code)
        self.assertFalse(result['error'],result)
        self.assertEqual(len(self.receipts),1)

    def test_timeout_after_action_retains_acted_record(self):
        store = {}
        code = CALLBACKS + '''
def observe(frame):
    if frame.step > 0:
        while True: pass
    return {'n':0}
''' + CALL
        state = copy.deepcopy(STATE); state['current_frame']['step'] = 1
        def handler(actions, **kwargs):
            row = self.handler(actions, **kwargs); row['state'] = state; return row
        result = self.run_code(code, saved_data=store, handler=handler, timeout=1)
        self.assertTrue(result['timed_out'])
        self.assertEqual(store['probe_1']['value']['status'],'ACTED')
        self.assertEqual(len(self.receipts),1)

    def test_batch_reuse_and_search_probes_cannot_execute(self):
        store = {}
        first = self.run_code(CALLBACKS + CALL, saved_data=store)
        self.assertFalse(first['error'],first)
        for expression in [CALL, CALL.replace("'UP'", "['UP','UP']")]:
            result = self.run_code(CALLBACKS + expression, saved_data=store)
            self.assertTrue(result['error'])
        self.assertEqual(len(self.receipts),1)
        code = CALLBACKS + '''
def successors(state):
    probe_transition('UP',observe=observe,predict=predict,state_key=key,name='nested')
    return []
r=bounded_search(0,successors,state_key=lambda s:s,is_goal=lambda s:False)
assert r['status']=='CALLBACK_ERROR'
'''
        result = self.run_code(code)
        self.assertFalse(result['error'],result)
        self.assertEqual(len(self.receipts),1)

    def test_variants_expose_only_their_named_capability(self):
        for variant, search, probe in [('control',False,False),('planner',True,False),('probing',False,True),('both',True,True)]:
            os.environ['ARC3_PLANNING_VARIANT'] = variant
            result = self.run_code("try:\n    bounded_search\n    print(True)\nexcept Exception:\n    print(False)\ntry:\n    probe_transition\n    print(True)\nexcept Exception:\n    print(False)")
            self.assertFalse(result['error'],result)
            self.assertEqual(result['stdout'].splitlines(),[str(search),str(probe)])
        self.assertEqual(self.receipts,[])


if __name__ == '__main__':
    unittest.main()
