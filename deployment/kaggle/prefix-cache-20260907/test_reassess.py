import json
from pathlib import Path
import tempfile
import unittest

import reassess
import cache_checkpoint_probe

ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/'analysis/deployment-probes/20260907-prefix-cache/raw/prefix-cache-20260907'
ARM=RAW/'arm-3-chunk-2048'


class ReassessmentTests(unittest.TestCase):
    def test_pinned_checkpoint_arithmetic_matches_all_retained_pairs(self):
        wheel=next(RAW.parent.rglob('sglang*.whl'))
        result=cache_checkpoint_probe.inspect(wheel,RAW)
        self.assertEqual(len(result['matches']),12)
        self.assertEqual(result['branching_control']['tracked_checkpoint'],19648)

    def test_real_directory_is_not_four_glob_matches(self):
        self.assertGreater(len(list(ARM.glob('batch-*.json'))),4)
        self.assertEqual(set(reassess.batch_summaries(ARM)),set(reassess.BATCH_NAMES))

    def test_all_retained_arms_reassess_without_mutation(self):
        before=reassess.snapshot(RAW)
        result=reassess.assess(RAW)
        self.assertEqual(reassess.snapshot(RAW),before)
        self.assertFalse(result['promotion'])
        self.assertEqual(len(result['conditions']),6)
        self.assertEqual(len(result['batches']),4)
        self.assertEqual([len(r['changed_entries']) for r in result['lifecycle']],[0,0,0,1])
        self.assertEqual(result['lifecycle'][3]['original_lifecycle']['status'],'failed')

    def test_only_exact_info_message_is_added(self):
        line='[2026-09-06 23:17:39] INFO:     Waiting for background tasks to complete. (CTRL+C to force quit)\n'
        self.assertIsNotNone(reassess.WAIT_LINE.fullmatch(line))
        for bad in [line.replace('INFO','WARN'),line.replace('complete.','failed.'),line.rstrip()+' traceback\n', 'other unclassified output\n']:
            self.assertIsNone(reassess.WAIT_LINE.fullmatch(bad))

    def fixture(self,root):
        for name in ('lifecycle.json','server.log','teardown-log-classification.json'):
            (root/name).write_bytes((ARM/name).read_bytes())

    def test_other_lifecycle_failures_are_not_excused(self):
        for field,value in [('root_returncode',1),('containment_passed',False),('quiet_window_valid',False),('term_only',False),('failure_classification','different_failure')]:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);self.fixture(root)
                lifecycle=reassess.read(root/'lifecycle.json');lifecycle[field]=value
                (root/'lifecycle.json').write_text(json.dumps(lifecycle))
                self.assertEqual(reassess.lifecycle_review(root)['reassessed_status'],'failed',field)

    def test_unknown_warning_stays_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            classification=reassess.read(root/'teardown-log-classification.json')
            row=next(e for e in classification['entries'] if not e['allowed'])
            before=row['text'];row['text']=before.replace('INFO','WARN')
            data=(root/'server.log').read_bytes().replace(before.encode(),row['text'].encode())
            (root/'server.log').write_bytes(data)
            (root/'teardown-log-classification.json').write_text(json.dumps(classification))
            self.assertEqual(reassess.lifecycle_review(root)['reassessed_status'],'failed')

    def test_unbound_log_entry_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            (root/'server.log').write_bytes(b'not the classified source')
            with self.assertRaisesRegex(ValueError,'retained server bytes'):reassess.lifecycle_review(root)


if __name__=='__main__':unittest.main()
