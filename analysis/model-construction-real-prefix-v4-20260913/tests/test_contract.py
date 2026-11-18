import copy
import json
import unittest
from unittest.mock import patch
from pathlib import Path
from constructor import run_construction, ConstructionLimits
from real_trace import load_case
from request_contract import validate_request

HERE=Path(__file__).resolve().parents[1]


def first_request(limits=None):
    bodies=[]
    def transport(body):
        bodies.append(copy.deepcopy(body))
        return {'usage':{'prompt_tokens':100,'completion_tokens':10,'cost':0.0001},'choices':[{'message':{'role':'assistant','tool_calls':[{'id':'preflight-only','type':'function','function':{'name':'request_evidence','arguments':json.dumps({'obstacle':'fixture','needed_observation':'fixture','decision_relevance':'fixture'})}}]}}]}
    run_construction(load_case(),transport=transport,limits=limits)
    return bodies[0]


class ContractTests(unittest.TestCase):
    def test_actual_constructor_request_explore_and_commit(self):
        for limits in (None,ConstructionLimits(max_model_requests=2)):
            body=first_request(limits)
            self.assertEqual(set(body),{'model','messages','provider','tools','tool_choice','max_tokens','reasoning'})
            self.assertTrue(validate_request(body)['passed'])
            self.assertNotIn('parallel_tool_calls',body)
    def test_v3_request_fails_unsupported_parallel_flag(self):
        body=first_request(); body['parallel_tool_calls']=False
        with self.assertRaisesRegex(ValueError,'unsupported optional.*parallel_tool_calls'): validate_request(body)
    def test_every_optional_field_requires_endpoint_advertisement(self):
        body=first_request()
        for field in ('tools','tool_choice','max_tokens','reasoning'):
            selection=json.loads((HERE/'PROVIDER-SELECTION.json').read_text())
            selection['selected']['supported_parameters'].remove(field)
            with self.assertRaisesRegex(ValueError,'unsupported optional'): validate_request(body,selection=selection)
    def test_gratuitous_fields_fail_closed(self):
        for field,value in [('temperature',1),('stream',False),('usage',{'include':True}),('include_reasoning',False),('reasoning_effort','none')]:
            body=first_request(); body[field]=value
            with self.assertRaises(ValueError): validate_request(body)
    def test_essential_values_routing_and_reasoning(self):
        mutations=[('reasoning',{'exclude':True}),('reasoning',{'effort':'none'}),('max_tokens',4097),('max_tokens',True),('tool_choice','auto'),('model','another-model')]
        for field,value in mutations:
            body=first_request(); body[field]=value
            with self.assertRaises(ValueError): validate_request(body)
        body=first_request(); body['provider']['allow_fallbacks']=True
        with self.assertRaises(ValueError): validate_request(body)
        model=json.loads((HERE/'model-metadata.json').read_text()); model['reasoning']['mandatory']=True
        with self.assertRaises(ValueError): validate_request(first_request(),model=model)
    def test_endpoint_value_and_context_limits(self):
        for field,value in [('status',1),('max_completion_tokens',1),('context_length',5000),('max_prompt_tokens',1)]:
            selection=json.loads((HERE/'PROVIDER-SELECTION.json').read_text()); selection['selected'][field]=value
            with self.assertRaises(ValueError): validate_request(first_request(),selection=selection)
        selection=json.loads((HERE/'PROVIDER-SELECTION.json').read_text()); selection['selected']['supports_tool_choice']['required']=False
        with self.assertRaises(ValueError): validate_request(first_request(),selection=selection)
    def test_nested_unreviewed_options_and_unanswered_calls(self):
        body=first_request(); body['tools'][0]['function']['strict']=True
        with self.assertRaises(ValueError): validate_request(body)
        body=first_request(); body['messages'].insert(-1,{'role':'tool','tool_call_id':'unmatched','content':'x'})
        with self.assertRaises(ValueError): validate_request(body)
    def test_reasoning_response_aborts_before_tools(self):
        for field,value in [('reasoning','unexpected'),('reasoning_content','unexpected'),('reasoning_details',[{'type':'reasoning.text','text':'unexpected'}])]:
            def transport(body):
                return {'usage':{'prompt_tokens':100,'completion_tokens':10,'cost':0.0001},'choices':[{'message':{'role':'assistant',field:value,'tool_calls':[{'id':'x','type':'function','function':{'name':'execute_python','arguments':'{"code":"print(1)"}'}}]}}]}
            with patch('constructor.execute_inspection') as inspect:
                result=run_construction(load_case(),transport=transport)
            inspect.assert_not_called()
            self.assertEqual(result.status,'INVALID_REASONING')
            self.assertEqual(result.model_requests,1)
    def test_multiple_tools_executed_sequentially_and_budgeted(self):
        bodies=[]
        def transport(body):
            bodies.append(body)
            if len(bodies)==1:
                calls=[{'id':str(i),'type':'function','function':{'name':'execute_python','arguments':'{"code":"print(1)"}'}} for i in range(2)]
            else:
                calls=[{'id':'end','type':'function','function':{'name':'request_evidence','arguments':'{"obstacle":"x","needed_observation":"x","decision_relevance":"x"}'}}]
            return {'usage':{'prompt_tokens':100,'completion_tokens':10,'cost':0.0001},'choices':[{'message':{'role':'assistant','tool_calls':calls}}]}
        with patch('constructor.execute_inspection',return_value={'status':'completed'}) as inspect:
            result=run_construction(load_case(),transport=transport)
        self.assertEqual(inspect.call_count,2)
        self.assertEqual(result.inspections,2)
        self.assertEqual(result.model_requests,2)
        self.assertTrue(validate_request(bodies[1])['passed'])

if __name__=='__main__': unittest.main()
