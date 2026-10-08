"""Independent negative cases against the current raw-output acceptance gate."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / 'serving-headroom-v2-20261008'
sys.path[:0] = [str(TARGET), str(HERE.parent / 'mtp-depth-20261006')]
import audit
import test_audit as native_test


class AuditBoundary(unittest.TestCase):
    def test_missing_start_end_are_not_complete_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            root = native_test.fixture(raw)
            self.assertTrue(audit.audit(raw)['qualified'], 'positive fixture must qualify before corruption')
            events = [json.loads(x) for x in (root/'N10A/events.jsonl').read_text().splitlines()]
            events = [e for e in events if e['kind'] not in ('metrics_before','metrics_after')]
            (root/'N10A/events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            native_test.manifest(root)
            result = audit.audit(raw)
            self.assertFalse(result['qualified'], result.get('error', 'incomplete accounting accepted'))

    def test_missing_profile_response_is_not_http_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            root = native_test.fixture(raw)
            self.assertTrue(audit.audit(raw)['qualified'], 'positive fixture must qualify before corruption')
            path = root/'PROFILE10/events.jsonl'
            events = [json.loads(x) for x in path.read_text().splitlines()]
            events = [e for e in events if e['kind'] != 'profile_response']
            path.write_text(''.join(json.dumps(e)+'\n' for e in events))
            native_test.manifest(root)
            result = audit.audit(raw)
            self.assertFalse(result['qualified'], result.get('error', 'missing HTTP receipt accepted'))


if __name__ == '__main__': unittest.main(verbosity=2)
