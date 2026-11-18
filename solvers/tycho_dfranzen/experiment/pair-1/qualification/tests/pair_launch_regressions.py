"""No duplicate launch, no partial reservation, and no silent hash mismatch."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import pair_regressions
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiment'))
from pair_run import launch, collect_arm, watch
from types import SimpleNamespace


class PairLaunchRegressions(unittest.TestCase):
    def setUp(self):
        pair_regressions.PairRegressions.setUp(self)
        identity = json.loads((self.out/'pair-identity.json').read_text())
        self.budget = Path(self.directory.name)/'budget.json'
        self.budget.write_text(json.dumps({'unreserved_gpu_hours':7,'run_reservations':[]}))
        (self.out/'EXPERIMENT-CARD.md').write_text('fixture card')
        (self.out/'LOCAL-QUALIFICATION.json').write_text(json.dumps({'passed':True,
            'payload_sha256': identity['source_payload_sha256'],
            'dataset_readback_sha256': identity['source_payload_sha256']}))

    def tearDown(self):
        self.directory.cleanup()

    def test_reserves_both_before_launch_and_refuses_second_attempt(self):
        calls = []
        class API:
            def kernels_push(api, path, timeout):
                budget = json.loads(self.budget.read_text())
                self.assertEqual(len(budget['run_reservations']),2)
                self.assertEqual(budget['unreserved_gpu_hours'],4)
                calls.append(Path(path).parent.name)
                return json.dumps({'versionNumber':1,'error':None})
        with patch('pair_run.quota',return_value={'fixture':True}):
            launch(self.out, API(), self.budget)
            with self.assertRaises(AssertionError): launch(self.out, API(), self.budget)
        self.assertEqual(calls,json.loads((self.out/'pair-identity.json').read_text())['launch_order'])

    def test_ambiguous_acceptance_is_not_retried_or_released(self):
        calls = []
        class API:
            def kernels_push(api, path, timeout):
                calls.append(path); raise ConnectionError('lost response')
        with patch('pair_run.quota',return_value={'fixture':True}), self.assertRaises(ConnectionError):
            launch(self.out, API(), self.budget)
        self.assertEqual(len(calls),1)
        self.assertEqual(json.loads(self.budget.read_text())['unreserved_gpu_hours'],4)
        self.assertTrue('unresolved' in json.loads((Path(calls[0]).parent/'launch-attempt.json').read_text())['state'])

    def test_tampered_notebook_cannot_consume_budget(self):
        (self.out/'control/kernel/notebook.ipynb').write_text('tampered')
        with self.assertRaises(AssertionError): launch(self.out, None, self.budget)
        self.assertEqual(json.loads(self.budget.read_text())['unreserved_gpu_hours'],7)

    def test_collector_rejects_wrong_artifact_hash(self):
        class Remote:
            def __init__(self,*args): pass
            def fetch(self,name):
                if name=='pair-evidence-index.json':
                    return json.dumps({'files':[{'path':'benchmark.json','critical':True,'sha256':'wrong'}]}).encode()
                return b'{}'
        with patch('collect_outputs.KaggleOutput', Remote): collect_arm(self.out,'control','fixture/kernel',1)
        receipt=json.loads((self.out/'control/collection-receipt.json').read_text())
        self.assertFalse(receipt['critical_complete'])
        self.assertEqual(receipt['errors'][0]['error_type'],'ValueError')

    def test_watcher_collects_each_terminal_arm_once_after_transient_failure(self):
        for arm in ('control','hybrid'):
            (self.out/arm/'launch-response.json').write_text(json.dumps({'versionNumber':1}))
        calls=[]; collected=[]
        class API:
            def kernels_status(api, kernel):
                calls.append(kernel)
                if len(calls)==1: raise ConnectionError('transient status lookup')
                return SimpleNamespace(status='RUNNING' if len(calls)==2 else 'COMPLETE')
        with patch('pair_run.quota',return_value={'fixture':True}), \
             patch('pair_run.time.sleep'), \
             patch('pair_run.collect_arm',side_effect=lambda directory,arm,kernel,version: collected.append((arm,version))):
            watch(self.out,API())
        self.assertEqual(sorted(collected),[('control',1),('hybrid',1)])
        self.assertEqual(json.loads((self.out/'monitor-result.json').read_text())['collected_arms'],['control','hybrid'])


if __name__=='__main__': unittest.main()
