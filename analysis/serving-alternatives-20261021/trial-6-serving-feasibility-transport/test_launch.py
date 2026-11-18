import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import launch

class AuthorityTests(unittest.TestCase):
    def fixture(self,root):
        now=dt.datetime.now(dt.timezone.utc)
        (root/'sentinel').write_text('reviewed bytes')
        (root/'freeze.json').write_text(json.dumps({'files':{'sentinel':hashlib.sha256(b'reviewed bytes').hexdigest()}}))
        report=dict(sources_sha256='source',config_sha256='config')
        (root/'review-packet.json').write_text(json.dumps(report))
        slot=dict(launch_authorized=True,local_only=False,coordinator_thread_id=launch.COORDINATOR,
                  launch_owner_thread_id=launch.OWNER,kernel_ref=launch.KERNEL_REF,
                  approved_freeze_sha256=launch.sha(root/'freeze.json'),confirmed_utc=now.isoformat(),
                  hard_timeout_s=5400,supervisor_timeout_s=5100,maximum_push_attempts=1,
                  maximum_model_loads=1,maximum_requests=6,concurrency=2,retry_authorized=False,
                  rate_or_adoption_authorized=False,sources_sha256='source',config_sha256='config',
                  serialized_assignment='UNIT_TEST_AUTHORITY_NOT_REAL')
        return now,slot,report

    def test_missing_authority_stops_before_any_sdk_or_packet_work(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(launch,'HERE',Path(directory)):
            with self.assertRaisesRegex(ValueError,'assignment'): launch.clearance()

    def test_assignment_failures_are_closed_at_public_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);now,slot,report=self.fixture(root)
            mutations=[{'launch_authorized':False},{'local_only':True},{'local_only':None},
                       {'confirmed_utc':(now-dt.timedelta(seconds=601)).isoformat()},
                       {'confirmed_utc':(now+dt.timedelta(seconds=1)).isoformat()},
                       {'approved_freeze_sha256':'wrong'},{'kernel_ref':'pranavagrawai/serving-candidate-feas5-20261023'},
                       {'serialized_assignment':'LOCAL_ONLY_SERIALIZATION_FIXTURE_NOT_LAUNCH_AUTHORITY'},
                       {'hard_timeout_s':5100},{'supervisor_timeout_s':5400},{'maximum_push_attempts':2},
                       {'maximum_requests':7},{'retry_authorized':True},{'rate_or_adoption_authorized':True},
                       {'sources_sha256':'oldsource'},{'config_sha256':'oldconfig'},{'launch_owner_thread_id':'other'}]
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    (root/'slot-confirmation.json').write_text(json.dumps(slot|mutation))
                    with patch.object(launch,'HERE',root):
                        with self.assertRaises(ValueError):launch.clearance(now)
            (root/'slot-confirmation.json').write_text(json.dumps(slot))
            with patch.object(launch,'HERE',root):self.assertEqual(launch.clearance(now),(slot,report))

    def test_source_drift_and_prior_attempt_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);now,slot,_=self.fixture(root)
            (root/'slot-confirmation.json').write_text(json.dumps(slot))
            with patch.object(launch,'HERE',root):
                (root/'sentinel').write_text('changed')
                with self.assertRaisesRegex(ValueError,'source changed'):launch.clearance(now)
                (root/'sentinel').write_text('reviewed bytes');(root/'launch-attempt.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'one attempt'):launch.clearance(now)

    def test_main_without_assignment_has_zero_sdk_and_no_arming(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(launch,'HERE',Path(directory)), patch.object(launch,'arm',side_effect=AssertionError('armed before clearance')):
            with self.assertRaises(ValueError):launch.main()
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_cancel_requested_redacted_active_unknown_are_not_terminal(self):
        launch.require_terminal_sessions([dict(status='COMPLETE'),dict(status='ERROR'),dict(status='CANCEL_ACKNOWLEDGED')])
        for status in ('CANCEL_REQUESTED','RUNNING','QUEUED','UNSPECIFIED','REDACTED_UNKNOWN','5'):
            with self.subTest(status=status),self.assertRaises(ValueError):launch.require_terminal_sessions([dict(status=status)])
        with self.assertRaises(ValueError):launch.require_terminal_sessions([])

    def test_failed_push_records_response_then_protects_one_attempt(self):
        import requests
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'kernel').mkdir();(root/'kernel/test.ipynb').write_text('reviewed outer')
            (root/'packet-spec.json').write_text(json.dumps({'metadata':{'code_file':'test.ipynb'}}))
            response=requests.Response();response.status_code=400;response._content=b'{"error":{"message":"specific server rejection"}}'
            response.headers['Content-Type']='application/json';response.headers['Set-Cookie']='do-not-copy'
            class Endpoint:
                calls=0
                def kernels_push(self,path,timeout):
                    self.calls+=1
                    if timeout!='5400':raise AssertionError('wrong platform cap')
                    raise requests.HTTPError('400 SaveKernel',response=response)
            api=Endpoint();manifest={'inner_notebook_sha256':'newinner','outer_notebook_bytes':14}
            with patch.object(launch,'HERE',root):
                with self.assertRaises(requests.HTTPError):launch.push_once(api,{'serialized_assignment':'fixture'},manifest)
                got=json.loads((root/'launch-error.json').read_text())
                self.assertEqual(got['status'],400);self.assertIn('specific server rejection',got['body'])
                self.assertNotIn('do-not-copy',json.dumps(got))
                with self.assertRaises(FileExistsError):launch.push_once(api,{},manifest)
            self.assertEqual(api.calls,1)
            self.assertTrue((root/'launch-attempt.json').exists())
            self.assertFalse((root/'launch-response.json').exists())

if __name__=='__main__':unittest.main()
