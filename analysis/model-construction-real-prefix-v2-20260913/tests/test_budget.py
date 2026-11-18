import json
import unittest
from unittest.mock import patch
from constructor import ConstructionLimits, run_construction, construction_messages
from real_trace import load_case
from sandbox_eval import execute_inspection


def response(name='execute_python', args=None, completion=10):
    return {'usage': {'prompt_tokens': 100, 'completion_tokens': completion},
            'choices': [{'message': {'role': 'assistant', 'tool_calls': [
                {'id': 'call', 'type': 'function', 'function': {'name': name,
                 'arguments': json.dumps(args or {'code':'print(1)'})}}]}}]}


class BudgetTests(unittest.TestCase):
    def setUp(self): self.case = load_case()
    def test_actual_usage_not_requested_allowance_and_telemetry(self):
        bodies=[]
        def transport(body): bodies.append(body); return response()
        with patch('constructor.execute_inspection', return_value={'status':'completed'}):
            r=run_construction(self.case, transport=transport)
        self.assertEqual(r.model_requests,16)
        self.assertEqual(r.usage['completion_tokens'],160)
        self.assertGreater(r.requested_tokens,16384)
        budgets=[json.loads(b['messages'][-1]['content'])['budget'] for b in bodies]
        self.assertEqual(budgets[1]['completion_tokens_remaining'],16374)
        self.assertEqual(budgets[1]['requests_remaining'],15)
        self.assertEqual(budgets[1]['inspections_remaining'],9)
        self.assertGreater(budgets[0]['seconds_remaining'],0)
        self.assertTrue(all(b['tool_choice']=='required' for b in bodies))
        for b,t in zip(bodies,budgets):
            self.assertLessEqual(b['max_tokens'],t['completion_tokens_remaining'])
            if t['phase']=='commit': self.assertNotIn('execute_python',[x['function']['name'] for x in b['tools']])
        self.assertEqual(r.inspections,10)
    def test_reserve_under_maximal_completions(self):
        budgets=[]
        def transport(body):
            budgets.append(json.loads(body['messages'][-1]['content'])['budget'])
            return response(completion=body['max_tokens'])
        with patch('constructor.execute_inspection', return_value={}):
            r=run_construction(self.case,transport=transport)
        first=next(b for b in budgets if b['phase']=='commit')
        self.assertGreaterEqual(first['requests_remaining'],2)
        self.assertGreaterEqual(first['completion_tokens_remaining'],4096)
        self.assertLessEqual(r.usage['completion_tokens'],16384)
    def test_missing_malformed_and_excess_usage_stop_before_tools(self):
        for usage in [None, {}, {'prompt_tokens':0,'completion_tokens':True},
                      {'prompt_tokens':0,'completion_tokens':-1},
                      {'prompt_tokens':0,'completion_tokens':'10'},
                      {'prompt_tokens':0,'completion_tokens':4097}]:
            def transport(body):
                r=response(); r['usage']=usage; return r
            with patch('constructor.execute_inspection') as tool:
                r=run_construction(self.case,transport=transport)
            self.assertEqual(r.status,'INVALID_USAGE')
            self.assertEqual(r.model_requests,1)
            tool.assert_not_called()
    def test_reserved_commit_or_explicit_evidence(self):
        seen=[]
        def transport(body):
            seen.append(body)
            return response('request_evidence',{'obstacle':'unknown','needed_observation':'new visible transition','decision_relevance':'test planned move'})
        r=run_construction(self.case,transport=transport,limits=ConstructionLimits(max_model_requests=2))
        self.assertEqual(r.status,'NEEDS_EVIDENCE')
        self.assertNotIn('execute_python',[t['function']['name'] for t in seen[0]['tools']])
    def test_prompt_and_spend_reservation_block_before_transport(self):
        for limits in [ConstructionLimits(max_prompt_tokens=1),ConstructionLimits(max_dollars=0.001)]:
            with patch('builtins.print'):
                with patch('constructor.execute_inspection'), patch('test_budget.response') as transport:
                    r=run_construction(self.case,transport=transport,limits=limits)
            transport.assert_not_called()
            self.assertEqual(r.status,'BUDGET_EXHAUSTED')
    def test_compact_evidence_parity_and_full_chronology_in_sandbox(self):
        compact=self.case.compact_view(); original=self.case.constructor_view()
        for key in original:
            if key not in {'initial_observation','current_observation'}: self.assertEqual(compact[key],original[key])
        full=self.case.inspection_view()
        self.assertEqual(full['initial_observation'],original['initial_observation'])
        self.assertEqual(full['current_observation'],original['current_observation'])
        for row,raw in zip(full['visible_transitions'],self.case.visible):
            for side in ['before','after']:
                self.assertEqual([[int(c,16) for c in s] for s in row[side]['grid_rows']],raw[side]['grid'])
        out=execute_inspection("import json; d=json.load(open('/evidence/case.json')); print([x['step'] for x in d['visible_transitions']])", self.case)
        self.assertEqual(out['stdout'],'[1, 2, 3, 4, 5]\n')
        self.assertLess(len(json.dumps(construction_messages(self.case,None))),len(json.dumps(original)))
    def test_repeated_prose_stops_without_wasting_all_requests(self):
        r=run_construction(self.case,transport=lambda b:{'usage':{'prompt_tokens':10,'completion_tokens':10},'choices':[{'message':{'role':'assistant','content':'no'}}]})
        self.assertEqual(r.model_requests,2)
        self.assertEqual(r.status,'INVALID_RESPONSE')

if __name__=='__main__': unittest.main()
