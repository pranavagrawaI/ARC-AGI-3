"""Real loopback read followed by disk failure across the fixed cutoff."""
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import test_replay as replay_tests

class LateFailingBody:
    def __init__(self,stream):self.stream=stream
    def __enter__(self):self.stream.__enter__();return self
    def __exit__(self,*args):return self.stream.__exit__(*args)
    def close(self):return self.stream.close()
    def write(self,data):
        time.sleep(.3)
        raise OSError('persistence failed after final body read')

def failing_body_open(original):
    def opened(path,*args,**kwargs):
        stream=original(path,*args,**kwargs)
        return LateFailingBody(stream) if path.name=='body.bin' and args and args[0]=='xb' else stream
    return opened

class PersistenceTests(unittest.TestCase):
    run_fixture=replay_tests.ReplayTests.run_fixture
    def test_late_body_write_failure_preserves_clock_and_invalidates(self):
        with patch.object(Path,'open',failing_body_open(Path.open)):
            root,server,obj,events,ok,elapsed=self.run_fixture('ok',(.05,.05))
        self.assertFalse(ok,'completed read plus lost persistence must not report successful cutoff')
        self.assertTrue(obj.failed)
        self.assertEqual(events[-1]['threads_alive'],0)
        receipts=[json.loads(p.read_text()) for p in (root/'out').glob('*/http.json')]
        self.assertEqual(len(receipts),2)
        for receipt in receipts:
            self.assertEqual(receipt['status'],200)
            length=int(dict((k.lower(),v) for k,v in receipt['headers'])['content-length'])
            self.assertEqual(receipt['received_bytes'],length)
            self.assertLess(receipt['body_completed_at'],obj.end)
            self.assertEqual(receipt['ended'],receipt['body_completed_at'])
            self.assertEqual(receipt['termination'],'ERROR')
            self.assertEqual(receipt['error_kind'],'PERSISTENCE')
    def test_partial_disk_failure_after_cutoff_is_not_cancellation(self):
        with patch.object(Path,'open',failing_body_open(Path.open)):
            root,server,obj,events,ok,elapsed=self.run_fixture('slow',(.05,.05))
        self.assertFalse(ok);self.assertTrue(obj.failed)
        receipts=[json.loads(p.read_text()) for p in (root/'out').glob('*/http.json')]
        self.assertEqual(len(receipts),2)
        for receipt in receipts:
            self.assertIsNone(receipt['body_completed_at'])
            self.assertGreater(receipt['received_bytes'],0)
            self.assertEqual(receipt['termination'],'ERROR')
            self.assertEqual(receipt['error_kind'],'PERSISTENCE')
