import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import launch


class LaunchTests(unittest.TestCase):
    def test_explicit_false_slot_cannot_arm_source_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);now=dt.datetime.now(dt.timezone.utc)
            freeze=root/'freeze.json';freeze.write_text(json.dumps(dict(files={})))
            packet=dict(sources_sha256='source',config_sha256='config')
            (root/'review-packet.json').write_text(json.dumps(packet))
            slot=dict(coordinator_thread_id=launch.COORDINATOR,approved_freeze_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest(),
                      confirmed_utc=now.isoformat(),hard_timeout_s=5400,supervisor_timeout_s=5100,
                      sources_sha256='source',config_sha256='config',serialized_assignment='explicit denial',launch_authorized=False)
            (root/'slot-confirmation.json').write_text(json.dumps(slot))
            with patch.object(launch,'HERE',root):
                with self.assertRaisesRegex(ValueError,'authoriz'):launch.clearance(now)


if __name__=='__main__':unittest.main()
