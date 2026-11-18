"""Offline tests: provider-shaped fixtures, no paid requests."""
import copy
from email.message import Message
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from typing import Any
import run
import audit


def response(d):
    return {'model':run.base.MODEL,'provider':'DekaLLM','usage':{'prompt_tokens':100,'completion_tokens':30,'cost':.0001},'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[{'id':'test-call','index':0,'type':'function','function':{'name':'decide','arguments':json.dumps(d)}}]}}]}


def choice(readback, actions=None) -> dict[str,Any]:
    actions=['UP'] if actions is None else actions
    return dict(actions=actions,planned_actions=actions[:],readback=readback,rationale='fixture',prediction='fixture',stop_reason='')


class Contracts(unittest.TestCase):
    def test_measure_all_cells_not_diff_union(self):
        self.assertEqual(run.measure([[1,0,1],[0,0,1]]),[
            {'color':0,'count':3,'bbox':[0,1,0,1]}, {'color':1,'count':3,'bbox':[0,1,0,2]}])
        d=choice({'color':1,'count':3,'bbox':[0,1,0,2]})
        self.assertTrue(run.validate(d,[[1,0,1],[0,0,1]])['readback_ok'])
        d['readback']['count']=4
        self.assertFalse(run.validate(d,[[1,0,1],[0,0,1]])['readback_ok'])
        d['actions']=['LEFT']
        self.assertFalse(run.validate(d,[[1,0,1],[0,0,1]])['actions_agree'])

    def test_provider_message_projection(self):
        d=choice({'color':1,'count':1,'bbox':[0,0,0,0]})
        message,_,parsed=run.parse(response(d))
        self.assertEqual(parsed,d)
        self.assertNotIn('index',message['tool_calls'][0])
        b=run.body([message],51)
        self.assertEqual(b['messages'][0],message)
        self.assertEqual(b['provider']['only'],['dekallm'])
        self.assertEqual(b['reasoning'],{'enabled':False})
        self.assertFalse(b['provider']['allow_fallbacks'])

    def test_invalid_readback_shape(self):
        d=choice({'color':True,'count':1,'bbox':[0,0,0,0]})
        with self.assertRaises(ValueError): run.parse(response(d))
        d['readback']['color']=1;d['readback']['bbox']=[0,0,0]
        with self.assertRaises(ValueError): run.parse(response(d))

    def test_gate_rejects_then_repairs_actual_actions(self):
        transitions,packet=run.base.evidence()
        for name in ('baseline','consistency'):
            with self.subTest(arm=name),tempfile.TemporaryDirectory() as temp:
                seen=[]
                def opener(req,timeout):
                    b=json.loads(req.data);seen.append(b)
                    current=json.loads(b['messages'][-1]['content'])['current_color_measurements']
                    d=choice(next(x for x in current if x['color']==12),['UP','DOWN'])
                    if len(seen)==1:
                        d['readback']['count']+=1
                        d['planned_actions']=['RIGHT']
                    return io.BytesIO(json.dumps(response(d)).encode())
                t=run.base.Transport('test-only',Path(temp),opener)
                result=run.play(t,transitions,packet,name,51,Path(temp))
                self.assertEqual(result['actions'],6)
                self.assertEqual(result['readback_failures'],1)
                self.assertEqual(result['action_mismatches'],1)
                self.assertEqual(result['rejections'],1 if name=='consistency' else 0)
                self.assertEqual(result['calls'],4 if name=='consistency' else 3)
                self.assertEqual(result['repaired_executed'],1 if name=='consistency' else 0)
                self.assertEqual(run.TREATMENT in [m.get('content') for m in seen[0]['messages']],name=='consistency')
                second_tool=[m for m in seen[1]['messages'] if m['role']=='tool'][-1]
                self.assertEqual('consistency_failure' in second_tool['content'],name=='consistency')
                self.assertTrue(all('index' not in c for b in seen for m in b['messages'] for c in m.get('tool_calls',[])))
                self.assertTrue(audit.audit(Path(temp),full=False)['passed'])

    def test_rejected_stop_does_not_terminate_or_act(self):
        transitions,packet=run.base.evidence()
        with tempfile.TemporaryDirectory() as temp:
            def opener(req,timeout):
                return io.BytesIO(json.dumps(response(choice({'color':12,'count':1,'bbox':[0,0,0,0]},[]))).encode())
            t=run.base.Transport('test',Path(temp),opener)
            s=run.play(t,transitions,packet,'consistency',51,Path(temp))
            self.assertEqual((s['actions'],s['calls'],s['rejections'],s['stop']),(0,4,4,'call_cap'))

    def test_http_failure_auditable_no_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            seen=[]
            def opener(req,timeout):
                seen.append(1)
                raise urllib.error.HTTPError(run.base.ENDPOINT,429,'test',Message(),io.BytesIO(b'{"error":"fixture"}'))
            t=run.base.Transport('test-only',Path(temp),opener)
            with self.assertRaises(urllib.error.HTTPError): t.call(run.body([],51),'51-baseline-0')
            a=audit.audit(Path(temp),full=False)
            self.assertTrue(a['passed']);self.assertFalse(a['performance_valid'])
            self.assertIsNone(a['metrics']['total_cost']);self.assertEqual(a['metrics']['completed'],0)
            self.assertEqual(len(seen),1)

    def test_absent_usage_aborts(self):
        p=response(choice({'color':1,'count':1,'bbox':[0,0,0,0]}));del p['usage']
        with self.assertRaises(ValueError): run.base.receipt(p,10000,.1)


if __name__=='__main__': unittest.main()
