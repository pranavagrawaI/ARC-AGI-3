import io
from email.message import Message
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
import run
from accounting import receipts

HERE = Path(__file__).resolve().parents[1]

class AuditTests(unittest.TestCase):
    def exercise(self, fail):
        # Fully mocked HTTP only; run real transport, constructor, persistence, audit.
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / 'test-episode'
            shutil.copytree(HERE, target, ignore=shutil.ignore_patterns('raw', 'STARTED.json', '__pycache__'))
            for name in ('model-construction-real-prefix-20260913','model-construction-real-prefix-v2-20260913','model-construction-real-prefix-v3-20260913'):
                (parent/name).symlink_to(HERE.parent/name, target_is_directory=True)
            (target/'PRELAUNCH.json').write_text(json.dumps({'passed':True,'subject_sha256':{}}))
            receipt = {'model':run.MODEL, 'provider':'Reka', 'usage':{'prompt_tokens':100,'completion_tokens':10,'cost':0.0001},
                       'choices':[{'message':{'role':'assistant','tool_calls':[{'id':'test','type':'function','function':{'name':'request_evidence','arguments':json.dumps({'obstacle':'unknown','needed_observation':'new transition','decision_relevance':'decision'})}}]}}]}
            error = urllib.error.HTTPError(run.ENDPOINT,429,'fixture rate limit',Message(),io.BytesIO(b'{"error":{"code":429}}'))
            with patch.object(run,'HERE',target), patch.object(run,'RAW',target/'raw'), patch.object(run,'AUTHORIZED_KEY','test-only-not-a-secret'):
                with patch('urllib.request.urlopen', side_effect=error if fail else None, return_value=io.BytesIO(json.dumps(receipt).encode())) as http:
                    result=run.execute()
                error.close()
                self.assertEqual(http.call_count,1)
            proc=subprocess.run([sys.executable,str(target/'audit.py')],capture_output=True,text=True)
            audit=json.loads((target/'AUDIT.json').read_text())
            self.assertEqual(proc.returncode,0,(proc.stderr,audit['checks']))
            self.assertTrue(audit['passed'])
            self.assertEqual(audit['request_count'],1)
            self.assertEqual(audit['completed_requests'],0 if fail else 1)
            self.assertGreaterEqual(audit['attempt_latencies'][0]['wall_seconds'],0)
            self.assertEqual(audit['prompt_tokens'],None if fail else 100)
            self.assertEqual(audit['completion_tokens'],None if fail else 10)
            self.assertEqual(audit['provider_reported_cost_usd'],None if fail else '0.0001')
            self.assertEqual(audit['semantic_evidence']['actual_provider_matches_pin'],'NOT_ESTABLISHED' if fail else 'ESTABLISHED')
            self.assertNotIn('actual_provider_matches_pin',audit['checks'])
            self.assertEqual(result['classification'],'INTEGRATION_INVALID' if fail else 'INTEGRATION_VALID_NEEDS_EVIDENCE')
    def test_http_failure_audit_and_latency(self): self.exercise(True)
    def test_completed_response_audit_and_latency(self): self.exercise(False)
    def test_partial_and_empty_receipts_do_not_claim_zero(self):
        for responses in ([],[{'usage':{'prompt_tokens':100,'completion_tokens':10,'cost':0.1}}]):
            a=receipts(responses,2)
            self.assertFalse(a['complete'])
            for key in ('prompt_tokens','completion_tokens','cost'): self.assertIsNone(a[key])
    def test_malformed_usage_and_cost(self):
        self.assertIsNone(receipts([{'usage':{'prompt_tokens':True,'completion_tokens':0}}],1)['prompt_tokens'])
        for cost in ('NaN',-1,True):
            self.assertIsNone(receipts([{'usage':{'prompt_tokens':1,'completion_tokens':0,'cost':cost}}],1)['cost'])

if __name__=='__main__': unittest.main()
