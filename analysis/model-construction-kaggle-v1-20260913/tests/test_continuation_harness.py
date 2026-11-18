"""Offline full harness regressions. No actual provider/network calls."""
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import constructor
import run

HERE = Path(__file__).resolve().parents[1]
PRIOR = HERE.parent/'model-construction-real-prefix-v4-20260913'


def receipt(name, args):
    return {'model':run.MODEL, 'provider':'Reka', 'usage':{'prompt_tokens':100,'completion_tokens':10,'cost':0.0001},
            'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[{'index':0,'id':'second','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}}]}


def evidence():
    return receipt('request_evidence',dict(obstacle='unknown',needed_observation='new transition',decision_relevance='next decision'))


class HarnessTests(unittest.TestCase):
    @contextlib.contextmanager
    def episode(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d); target=parent/'episode'; target.mkdir()
            for p in HERE.iterdir():
                if p.is_file() and p.suffix in {'.py','.json'}: shutil.copy2(p,target/p.name)
            for name in ('model-construction-real-prefix-20260913','model-construction-real-prefix-v2-20260913','model-construction-real-prefix-v3-20260913'):
                (parent/name).symlink_to(HERE.parent/name,target_is_directory=True)
            (target/'PRELAUNCH.json').write_text(json.dumps({'passed':True,'subject_sha256':{},'incumbent_suffix':{'compatible_depth':1}}))
            with patch.object(run,'HERE',target),patch.object(run,'RAW',target/'raw'),patch.object(run,'AUTHORIZED_KEY','fixture-only'):
                yield target

    def mock_http(self, responses):
        return patch('urllib.request.urlopen',side_effect=[io.BytesIO(json.dumps(x).encode()) for x in responses])

    def audit(self, target):
        p=subprocess.run([sys.executable,str(target/'audit.py')],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        return json.loads((target/'AUDIT.json').read_text())

    def test_captured_real_response_executes_tool_then_second_wire(self):
        raw=json.loads((PRIOR/'raw/responses/00.json').read_text())
        with self.episode() as target, self.mock_http([raw,evidence()]) as http:
            r=run.execute()
            self.assertEqual(r['classification'],'INTEGRATION_VALID_NEEDS_EVIDENCE')
            self.assertEqual(http.call_count,2)
            transcript=json.loads((target/'raw/transcript.json').read_text())
            self.assertEqual(transcript[0]['response'],raw)
            second=json.loads((target/'raw/requests/01.body').read_bytes())
            call=next(x for x in second['messages'] if x['role']=='assistant')['tool_calls'][0]
            self.assertNotIn('index',call)
            events=[json.loads(x) for x in (target/'raw/tool_events.jsonl').read_text().splitlines()]
            self.assertEqual(json.loads(events[0]['rendered_result_message']['content'])['status'],'completed')
            self.assertIn(events[0]['rendered_result_message'],second['messages'])
            self.assertEqual(self.audit(target)['completed_requests'],2)

    def test_local_second_presend_preserves_receipts_and_progress(self):
        raw=json.loads((PRIOR/'raw/responses/00.json').read_text())
        original=constructor.validate_request; calls=0
        def validate(body):
            nonlocal calls
            calls+=1
            if calls==2: raise ValueError('fixture local pre-send failure')
            return original(body)
        with self.episode() as target,self.mock_http([raw]) as http,patch.object(constructor,'validate_request',side_effect=validate):
            r=run.execute()
            self.assertEqual(http.call_count,1)
            self.assertEqual(r['failure_boundary'],'local_after_receipts')
            self.assertEqual(r['usage'],{k:raw['usage'][k] for k in ('prompt_tokens','completion_tokens')})
            self.assertEqual(r['requested_tokens'],4096)
            self.assertEqual(r['inspections'],1)
            self.assertEqual(len(json.loads((target/'raw/transcript.json').read_text())),1)
            self.audit(target)

    def test_successful_submission_freeze_and_single_suffix_check(self):
        source=(HERE/'tests/witness_candidate.py').read_text()
        item=dict(decision='fixed rollout',obstacle='test fixture',operation='test',claim='fixture only',evidence_refs=[1],dependencies=['fixture'],completion_checks=[1,2,3,4,5],remaining_scope='all other cases',reasoning_need='needs_computation')
        with self.episode() as target,self.mock_http([receipt('submit_candidate',dict(work_item=item,candidate_source=source))]):
            r=run.execute()
            self.assertEqual(r['construction_status'],'CANDIDATE_READY')
            self.assertEqual(r['sealed_suffix_evaluation']['compatible_depth'],3)
            self.assertEqual((target/'raw/frozen_candidate.py').read_text(),source)
            self.audit(target)

    def test_artifact_failure_retains_completed_usage(self):
        original=run.write_json
        def write(path,value):
            if path.name=='construction.json': raise OSError('fixture artifact failure')
            return original(path,value)
        with self.episode() as target,self.mock_http([evidence()]),patch.object(run,'write_json',side_effect=write):
            r=run.execute()
            self.assertEqual(r['failure_boundary'],'local_after_receipts')
            self.assertEqual(r['usage'],{'prompt_tokens':100,'completion_tokens':10})
            self.audit(target)

    def test_missing_usage_stops_full_harness_and_audits_unknown(self):
        response=evidence(); response.pop('usage')
        with self.episode() as target,self.mock_http([response]) as http:
            r=run.execute()
            self.assertEqual(r['construction_status'],'INVALID_USAGE')
            self.assertIsNone(r['usage'])
            self.assertEqual(http.call_count,1)
            self.assertEqual(r['inspections'],0)
            self.audit(target)

    def test_first_local_presend_failure_has_zero_attempts(self):
        with self.episode() as target,patch.object(constructor,'validate_request',side_effect=ValueError('local fixture')),patch('urllib.request.urlopen') as http:
            r=run.execute()
            self.assertEqual(r['failure_boundary'],'local_pre_send')
            self.assertEqual(r['model_requests'],0)
            self.assertEqual(r['requested_tokens'],0)
            self.assertEqual(r['usage'],{'prompt_tokens':0,'completion_tokens':0})
            http.assert_not_called()
            self.audit(target)

    def test_exact_completion_cap_and_reserved_commit_in_full_harness(self):
        budgets=[]
        def respond(request,**kwargs):
            body=json.loads(request.data)
            budget=json.loads(body['messages'][-1]['content'])['budget']; budgets.append(budget)
            if budget['completion_tokens_remaining']==body['max_tokens']:
                response=evidence()
            elif budget['phase']=='commit':
                response=receipt('submit_candidate',{})
            else:
                response=receipt('execute_python',{'code':'print(1)'})
            response['usage']['completion_tokens']=body['max_tokens']
            return io.BytesIO(json.dumps(response).encode())
        with self.episode() as target,patch('urllib.request.urlopen',side_effect=respond):
            r=run.execute()
            self.assertEqual(r['usage']['completion_tokens'],16384)
            self.assertEqual(r['classification'],'INTEGRATION_VALID_NEEDS_EVIDENCE')
            first=next(b for b in budgets if b['phase']=='commit')
            self.assertGreaterEqual(first['requests_remaining'],2)
            self.assertGreaterEqual(first['completion_tokens_remaining'],4096)
            self.audit(target)

    def test_audit_detects_tampering_without_relaunch(self):
        with self.episode() as target,self.mock_http([evidence()]) as http:
            run.execute()
            self.audit(target)
            (target/'raw/requests/00.body').write_text('{}')
            p=subprocess.run([sys.executable,str(target/'audit.py')],capture_output=True,text=True)
            self.assertEqual(p.returncode,1)
            self.assertFalse(json.loads((target/'AUDIT.json').read_text())['checks']['request_wire_bodies_match'])
            self.assertEqual(http.call_count,1)

if __name__=='__main__': unittest.main()
