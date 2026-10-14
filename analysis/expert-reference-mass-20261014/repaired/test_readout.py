"""Reject missing qualification and stale clocks even if a self-report says valid."""
import json
from pathlib import Path
import tempfile
import unittest
import readout


class ReadoutTests(unittest.TestCase):
    def test_stale_nonfinite_and_empty_clock_receipts_are_invalid(self):
        good={'gate_is_fresh':True,'play_s':900,'child_started':1500,'clock_start':1501,'clock_deadline':2401,
              'clock_is_fresh_and_bounded':True}
        self.assertTrue(readout.clock_valid(good))
        for patch in [{'clock_start':100,'clock_deadline':1000}, {'gate_is_fresh':False},
                      {'clock_start':float('nan')}, {'clock_deadline':2403}, {'play_s':800}]:
            self.assertFalse(readout.clock_valid({**good,**patch}),patch)
        self.assertFalse(readout.clock_valid(None))

    def test_graph_marker_alone_cannot_establish_all_tests_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'routing-tests.json').write_text(json.dumps({'returncode':0,'gpu_graph_executed':True}))
            (out/'routing-tests.stdout.txt').write_text('REFERENCE_GRAPH_TEST_PASSED\n')
            self.assertFalse(readout.qualification_valid(out))
            stderr=out/'routing-tests.stderr.txt'
            for text in ['Ran 7 tests in 0.1s\n\nOK (skipped=1)\n','Ran 6 tests in 0.1s\n\nOK\n',
                         'Ran 7 tests in 0.1s\n\nFAILED (failures=1)\n']:
                stderr.write_text(text);self.assertFalse(readout.qualification_valid(out))
            stderr.write_text('Ran 7 tests in 0.1s\n\nOK\n');self.assertTrue(readout.qualification_valid(out))


if __name__=='__main__':unittest.main(verbosity=2)
