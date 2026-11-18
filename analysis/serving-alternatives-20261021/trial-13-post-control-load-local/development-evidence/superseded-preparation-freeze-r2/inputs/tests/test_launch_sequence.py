"""Local full launch lifecycle: real checks/clearance, fake SDK and no remote work."""
import contextlib
import datetime as dt
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import requests
import launch
import reconciliation
import test_authority

OUTPUTS = ('quota-prelaunch.json', 'sessions-prelaunch.json',
           'predecessor-status-prelaunch.json', 'failed-save-reconciliation-prelaunch.json')


class Record:
    def __init__(self, raw):
        self.raw = raw
        self.ref = raw.get('ref', '')

    def to_dict(self):
        return self.raw


class LaunchSequence(unittest.TestCase):
    def fixture(self):
        helper = test_authority.Authority()
        directory, slot = helper.fixture()
        self.addCleanup(helper.doCleanups)
        (directory / 'input.py').write_text('immutable input\n')
        (directory / 'packet-spec.json').write_text(json.dumps({'metadata': {'code_file': 'main.ipynb'}}))
        for name in OUTPUTS:
            (directory / name).write_text('earlier audit receipt\n')
        # Reflect the ACTUAL reviewed manifest boundary, rather than assuming it is correct.
        reviewed = json.loads((launch.HERE / launch.PREPARATION_FREEZE).read_text())['files']
        inputs = ['input.py', 'packet-spec.json', 'review-packet.json', 'prior-reconciliation-policy.json']
        inputs += [name for name in OUTPUTS if name in reviewed]
        freeze = directory / launch.PREPARATION_FREEZE
        freeze.write_text(json.dumps({'files': {name: launch.sha(directory / name) for name in inputs}}))
        slot['approved_freeze_sha256'] = launch.sha(freeze)
        (directory / 'slot-confirmation.json').write_text(json.dumps(slot))
        return directory

    def execute(self, directory, drift=False):
        class FakeApi:
            def __init__(self):
                self.pushed = []
                self.authenticated = False

            def authenticate(self):
                self.authenticated = True

            @contextlib.contextmanager
            def build_kaggle_client(self):
                quota = SimpleNamespace(gpu_quota=SimpleNamespace(
                    time_used=dt.timedelta(seconds=68903.943), time_reserved=dt.timedelta()))
                endpoint = SimpleNamespace(get_accelerator_quota_statistics=lambda: quota)
                yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=endpoint))

            def kernels_list(self, **kwargs):
                return [Record(dict(reconciliation.STUB)), Record({'ref': reconciliation.PREDECESSOR_REF})]

            def kernels_list_with_response(self, **kwargs):
                if drift:
                    (directory / 'input.py').write_text('unexpected input drift\n')
                return Record({'kernels': [dict(reconciliation.STUB)]}
                              if kwargs['search'] == 'serving-candidate-feas5-20261023' else {})

            def kernels_status(self, ref):
                if ref == reconciliation.PRIOR_REF:
                    response = requests.Response()
                    response.status_code = 404
                    response._content = b'{"error":{"code":404,"message":"No runs found for this kernel."}}'
                    raise requests.HTTPError('404', response=response)
                return SimpleNamespace(status=SimpleNamespace(name='COMPLETE'))

            def kernels_push(self, path, **kwargs):
                self.pushed.append((path, kwargs))
                return 'LOCAL_FAKE_ACCEPTED'

        api = FakeApi()
        policy = reconciliation.load_policy()

        def fake_arm(slot, report, destination):
            destination.mkdir()
            (destination / 'main.ipynb').write_text('local fake, no executable payload')
            return {'inner_notebook_sha256': 'LOCAL_FAKE_INNER', 'outer_notebook_bytes': 33}

        sdk = SimpleNamespace(KaggleApi=lambda: api)
        with patch.object(launch, 'HERE', directory), patch.object(launch, 'arm', fake_arm), \
             patch.object(reconciliation, 'load_policy', return_value=policy), \
             patch.dict(sys.modules, {'kaggle.api.kaggle_api_extended': sdk}), \
             patch.object(socket.socket, 'connect', side_effect=AssertionError('NETWORK FORBIDDEN')), \
             patch.object(requests.Session, 'send', side_effect=AssertionError('NETWORK FORBIDDEN')):
            launch.main()
            self.assertTrue(api.authenticated)
            self.assertEqual(api.pushed, [(str(directory / 'kernel'), {'timeout': '6300'})])
            for name in OUTPUTS:
                self.assertNotEqual((directory / name).read_text(), 'earlier audit receipt\n')
            receipt = json.loads((directory / 'failed-save-reconciliation-prelaunch.json').read_text())
            self.assertEqual(receipt['judgment']['classification'], 'RECONCILED_FAILED_SAVE_NO_RUN_OBSERVED')
            self.assertEqual(launch.sha(directory / launch.PREPARATION_FREEZE),
                             json.loads((directory / 'slot-confirmation.json').read_text())['approved_freeze_sha256'])
            with self.assertRaisesRegex(ValueError, 'one attempt already'):
                launch.main()
            self.assertEqual(len(api.pushed), 1)

    def test_real_fresh_checks_updates_do_not_invalidate_clearance(self):
        directory = self.fixture()
        self.execute(directory)
        self.assertTrue((directory / 'launch-attempt.json').exists())
        self.assertTrue((directory / 'launch-response.json').exists())

    def test_input_drift_during_preflight_still_blocks_before_attempt(self):
        directory = self.fixture()
        with self.assertRaisesRegex(ValueError, 'reviewed source changed: input.py'):
            self.execute(directory, drift=True)
        self.assertFalse((directory / 'launch-attempt.json').exists())
        self.assertFalse((directory / 'launch-response.json').exists())

    def test_manifest_excludes_live_operational_outputs(self):
        frozen = json.loads((launch.HERE / launch.PREPARATION_FREEZE).read_text())
        forbidden = set(OUTPUTS) | {'slot-confirmation.json', 'launch-attempt.json',
                                   'launch-response.json', 'launch-error.json'}
        self.assertFalse(forbidden.intersection(frozen['files']))
        self.assertFalse(any(Path(name).parts[0] in {'kernel', 'output', 'runtime', 'remote-source', 'collection'}
                             for name in frozen['files']))


if __name__ == '__main__':
    unittest.main()
