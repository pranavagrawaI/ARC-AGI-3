import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import launch

class Authority(unittest.TestCase):
    def fixture(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);p=Path(t.name)
        (p/launch.PREPARATION_FREEZE).write_text(json.dumps({'files':{}}));(p/'prior-reconciliation-policy.json').write_text('{}')
        report={'sources_sha256':'sources','config_sha256':'config'};(p/'review-packet.json').write_text(json.dumps(report))
        slot=dict(launch_authorized=True,local_only=False,coordinator_thread_id=launch.COORDINATOR,launch_owner_thread_id=launch.OWNER,kernel_ref=launch.KERNEL_REF,
                  approved_freeze_sha256=launch.sha(p/launch.PREPARATION_FREEZE),confirmed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),hard_timeout_s=10200,supervisor_timeout_s=9900,
                  maximum_push_attempts=1,maximum_model_loads=2,maximum_requests=2738,concurrency=10,retry_authorized=False,adoption_authorized=False,capacity_screen_authorized=True,
                  sources_sha256='sources',config_sha256='config',serialized_assignment=launch.KERNEL_REF.split('/')[1]+'-CPU_TEST',reconciliation_policy_sha256=launch.sha(p/'prior-reconciliation-policy.json'))
        return p,slot
    def test_missing_false_stale_wrong_freeze_old_identity_and_one_attempt(self):
        p,slot=self.fixture()
        with patch.object(launch,'HERE',p):
            with self.assertRaisesRegex(ValueError,'no reviewed'):launch.clearance()
            changes=[{'launch_authorized':False},{'local_only':True},{'confirmed_utc':(dt.datetime.now(dt.timezone.utc)-dt.timedelta(hours=1)).isoformat()},
                     {'approved_freeze_sha256':'wrong'},{'kernel_ref':'pranavagrawai/serving-wire-capacity8-20261023'},{'supervisor_timeout_s':10800},{'reconciliation_policy_sha256':'wrong'}]
            for change in changes:
                (p/'slot-confirmation.json').write_text(json.dumps(dict(slot,**change)))
                with self.assertRaises(ValueError):launch.clearance()
            (p/'slot-confirmation.json').write_text(json.dumps(slot));self.assertEqual(launch.clearance()[0],slot)
            (p/'launch-attempt.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'one attempt'):launch.clearance()
    def test_pending_cancellation_is_not_terminal(self):
        launch.require_terminal_sessions([{'status':'CANCEL_ACKNOWLEDGED'}])
        for state in ('CANCEL_REQUESTED','RUNNING','QUEUED','REDACTED_UNKNOWN'):
            with self.assertRaises(ValueError):launch.require_terminal_sessions([{'status':state}])
