"""Search through the actual subprocess: retention and no speculative actions."""
import unittest
from test_basics import SandboxCase, commit

SETUP = '''
def successors(s):
    if s < 3:
        yield 'advance', s+1
def key(s): return s
def goal(s): return s == 3
def estimate(s): return 3-s
def plan(s):
    return bounded_search(s, successors, state_key=key, is_goal=goal, heuristic=estimate)
'''


class SearchSandboxTests(SandboxCase):
    def test_plan_functions_survive_next_call(self):
        first = self.run_code(SETUP + "save_data('plan', plan(0))", kept_functions=[])
        self.assertFalse(first.get('error'))
        self.assertEqual(self.receipts,[])
        saved,_=commit({},first)
        self.assertTrue({'successors','key','goal','estimate','plan'} <= set(saved))
        again=self.run_code("p=plan(1)\nassert p['status']=='FOUND'\nassert p['path']==['advance','advance']\nprint('RETAINED_SEARCH_OK')",
                            kept_functions=list(saved.values()))
        self.assertFalse(again.get('error'))
        self.assertIn('RETAINED_SEARCH_OK',again['stdout'])
        self.assertEqual(self.receipts,[])

    def test_search_callback_cannot_act_but_later_action_can(self):
        code='''
def bad_goal(s):
    action('UP')
    return True
r=bounded_search(0, lambda s: [], state_key=lambda s:s, is_goal=bad_goal)
assert r['status']=='CALLBACK_ERROR'
assert r['callback']=='is_goal'
assert 'must not execute' in r['error']
action('UP')
'''
        result=self.run_code(code)
        self.assertFalse(result.get('error'))
        self.assertEqual(len(self.receipts),1)
        self.assertEqual(len(result['action_results']),1)

    def test_nested_search_is_error_and_clears_guard(self):
        code=SETUP+'''
def nested(s):
    return bounded_search(s, successors, state_key=key, is_goal=goal)['depth']
r=bounded_search(0, successors, state_key=key, is_goal=goal, heuristic=nested)
assert r['status']=='CALLBACK_ERROR'
assert 'Nested' in r['error']
assert plan(0)['status']=='FOUND'
'''
        result=self.run_code(code)
        self.assertFalse(result.get('error'))
        self.assertEqual(self.receipts,[])

    def test_hung_callback_is_killed_by_outer_tool(self):
        code='''
def hung(s):
    while True: pass
bounded_search(0, hung, state_key=lambda s:s, is_goal=lambda s:False, max_seconds=.01)
'''
        result=self.run_code(code,timeout=1)
        self.assertIn('timed out',result['error'])
        self.assertEqual(self.receipts,[])

    def test_shadowed_search_binding_cannot_silently_change_retained_wrapper(self):
        first=self.run_code("bounded_search=lambda *args,**kwargs: {'status':'FAKE'}\n"
                            "def plan(): return bounded_search(0,None)\nprint(plan()['status'])",kept_functions=[])
        self.assertEqual(first['stdout'].strip(),'FAKE')
        kept,_=commit({},first)
        self.assertNotIn('plan',kept)


if __name__=='__main__':
    unittest.main()
