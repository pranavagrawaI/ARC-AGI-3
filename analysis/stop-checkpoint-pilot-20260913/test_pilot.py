"""Offline qualification; fake responses are test fixtures, never run evidence."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from email.message import Message
import urllib.error
import pilot


def response():
    return {'model': pilot.MODEL, 'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'cost': .0001},
            'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'x', 'type': 'function', 'function': {'name': 'decide', 'arguments': json.dumps({'actions':['UP'], 'rationale':'test', 'prediction':'move', 'stop_reason':''})}}]}}]}


class Contracts(unittest.TestCase):
    def test_grid_roundtrip(self):
        grid = [[1,1,2],[1,1,2],[3,3,3]]
        self.assertEqual(pilot.encode(grid), {'rows_rle': [[2,[[2,1],[1,2]]],[1,[[3,3]]]]})
        self.assertEqual(pilot.changes([[1,2]], [[1,3]]), [[0,1,2,3]])

    def test_real_replay(self):
        transitions, packet = pilot.evidence()
        env, raw = pilot.environment(transitions)
        self.assertEqual(raw.frame[-1].tolist(), transitions[-1]['after']['grid'])
        bad = copy.deepcopy(transitions)
        bad[2]['after']['grid'][0][0] = 15
        with self.assertRaises(AssertionError):
            pilot.environment(bad)

    def test_completed_transport_and_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            seen = []
            def opener(req, timeout):
                seen.append(json.loads(req.data))
                return io.BytesIO(json.dumps(response()).encode())
            t = pilot.Transport('test-only-key', Path(temp), opener)
            t.call(pilot.body_for([{'role':'user','content':pilot.CHECKPOINT}],41), 'checkpoint-test')
            self.assertEqual(seen[0]['reasoning'], {'enabled':False})
            self.assertFalse(seen[0]['provider']['allow_fallbacks'])
            self.assertEqual(seen[0]['tool_choice']['function']['name'], 'decide')
            self.assertEqual(seen[0]['messages'][0]['content'], pilot.CHECKPOINT)
            self.assertEqual(t.spent, .0001)
            a = pilot.audit(Path(temp))
            self.assertTrue(a['passed'])
            self.assertEqual(a['total_cost'], .0001)

    def test_first_http_failure_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []
            def opener(req, timeout):
                calls.append(1)
                raise urllib.error.HTTPError(pilot.ENDPOINT,429,'rate limit',Message(),io.BytesIO(b'{"error":"test rate limit"}'))
            t = pilot.Transport('test-only-key', Path(temp), opener)
            with self.assertRaises(urllib.error.HTTPError):
                t.call(pilot.body_for([],41), 'baseline-test')
            self.assertEqual(len(calls), 1)
            a = pilot.audit(Path(temp))
            self.assertTrue(a['passed'])
            self.assertFalse(a['performance_valid'])
            self.assertIsNone(a['total_cost'])
            self.assertEqual(a['completed'], 0)

    def test_actual_arm_routing_and_action_cap(self):
        transitions, packet = pilot.evidence()
        for name in ('baseline','checkpoint'):
            with self.subTest(arm=name), tempfile.TemporaryDirectory() as temp:
                seen = []
                def opener(req, timeout):
                    seen.append(json.loads(req.data))
                    p = response()
                    p['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = json.dumps({'actions':['UP','DOWN'],'rationale':'fixture','prediction':'fixture','stop_reason':''})
                    return io.BytesIO(json.dumps(p).encode())
                t = pilot.Transport('test-only', Path(temp), opener)
                with patch.object(pilot,'HERE',Path(temp)):
                    result = pilot.arm(t,transitions,packet,name,41)
                self.assertEqual(result['actions'],6)
                self.assertEqual(result['calls'],3)
                self.assertEqual(result['stop'],'action_cap')
                self.assertEqual(pilot.CHECKPOINT in [m['content'] for m in seen[0]['messages']], name=='checkpoint')
                events = [json.loads(l) for l in (Path(temp)/'trajectories.jsonl').read_text().splitlines()]
                self.assertEqual(sum(r['event']=='action' for r in events),6)
                self.assertTrue(all(r['before']!=r['after'] for r in events if r['event']=='action'))

    def test_invalid_usage_stops(self):
        for value in (None, -1, True, 10000000):
            p = response()
            p['usage']['prompt_tokens'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                pilot.receipt(p,1000,.1)
        for value in (None,-1,True,float('nan'),10):
            p = response()
            p['usage']['cost'] = value
            with self.subTest(cost=value), self.assertRaises(ValueError):
                pilot.receipt(p,1000,.1)

    def test_wrong_model_reasoning(self):
        p = response()
        p['model'] = 'other'
        with self.assertRaises(ValueError): pilot.receipt(p,1000,.1)
        p = response()
        p['choices'][0]['message']['reasoning'] = 'hidden thought'
        with self.assertRaises(ValueError): pilot.receipt(p,1000,.1)

    def test_cap_prevents_network(self):
        with tempfile.TemporaryDirectory() as temp:
            t = pilot.Transport('test', Path(temp), lambda *a,**k: self.fail('network called'))
            t.spent = 1
            with self.assertRaises(RuntimeError): t.call(pilot.body_for([],41),'test')
            self.assertEqual(t.index,0)

    def test_action_validation_and_stop(self):
        self.assertEqual(pilot.decision(response())[2]['actions'], ['UP'])
        for actions in (['BAD'],['UP']*3,[True],None):
            p = response()
            p['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = json.dumps({'actions':actions,'rationale':'','prediction':'','stop_reason':''})
            with self.subTest(actions=actions), self.assertRaises(ValueError): pilot.decision(p)
        p = response()
        p['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = json.dumps({'actions':[],'rationale':'','prediction':'','stop_reason':'stop'})
        self.assertEqual(pilot.decision(p)[2]['actions'], [])


if __name__ == '__main__':
    unittest.main()
