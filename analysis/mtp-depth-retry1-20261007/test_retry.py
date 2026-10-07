"""Exercise the actual notebook guard and constrain changes to the retry wrapper."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import prepare


class RetryTests(unittest.TestCase):
    def run_guard(self, stdout, returncode=0):
        source = (prepare.HERE / 'hardware_gate.py').read_text()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            result = SimpleNamespace(stdout=stdout, stderr='', returncode=returncode)
            with patch('subprocess.run', return_value=result), patch('pathlib.Path', return_value=out):
                error = None
                try:
                    exec(compile(source, 'hardware_gate.py', 'exec'), {})
                except RuntimeError as exc:
                    error = exc
            return json.loads((out / 'hardware.json').read_text()), error

    def test_valid_rtx_passes(self):
        receipt, error = self.run_guard('NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 580.178.04\n')
        self.assertTrue(receipt['valid'])
        self.assertIsNone(error)

    def test_original_dual_t4_aborts(self):
        receipt, error = self.run_guard('Tesla T4, 15360, 580.178.04\n' * 2)
        self.assertFalse(receipt['valid'])
        self.assertIsNotNone(error)

    def test_wrong_count_memory_or_unknown_aborts(self):
        for inventory, code in [('NVIDIA RTX PRO 6000, 97887, 580\n' * 2, 0),
                                ('NVIDIA RTX PRO 6000, 15360, 580\n', 0),
                                ('NVIDIA RTX PRO 6000, N/A, 580\n', 0), ('', 1)]:
            with self.subTest(inventory=inventory):
                receipt, error = self.run_guard(inventory, code)
                self.assertFalse(receipt['valid'])
                self.assertIsNotNone(error)

    def test_generated_notebook_keeps_frozen_cells(self):
        prepare.build()
        old = json.loads((prepare.ORIGINAL / 'kaggle/notebook/mtp-depth-sweep-20261006.ipynb').read_text())
        new = json.loads((prepare.HERE / f'kernel/{prepare.SLUG}.ipynb').read_text())
        self.assertEqual(''.join(new['cells'][1]['source']), (prepare.HERE / 'hardware_gate.py').read_text())
        self.assertEqual(new['cells'][2:5], old['cells'][1:4])
        source = ''.join(new['cells'][-1]['source'])
        addition = f"PROVENANCE.update(run_id='{prepare.RUN_ID}', retry_prelaunch_tag='{prepare.TAG}')\n"
        self.assertEqual(source.replace(addition, ''), ''.join(old['cells'][-1]['source']))
        metadata = json.loads((prepare.HERE / 'kernel/kernel-metadata.json').read_text())
        original = json.loads((prepare.ORIGINAL / 'kaggle/notebook/kernel-metadata.json').read_text())
        self.assertEqual(metadata['competition_sources'], [prepare.COMPETITION])
        self.assertEqual(metadata['machine_shape'], 'NvidiaRtxPro6000')
        self.assertEqual(metadata['dataset_sources'], original['dataset_sources'])
        for cell in new['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), 'generated-notebook', 'exec')

    def test_real_sdk_push_transmits_competition_and_machine(self):
        from kaggle.api.kaggle_api_extended import KaggleApi
        prepare.build()
        client = MagicMock()
        context = MagicMock()
        context.__enter__.return_value = client
        api = KaggleApi()
        with patch.object(api, 'build_kaggle_client', return_value=context):
            api.kernels_push(str(prepare.HERE / 'kernel'), timeout='16200')
        request = client.kernels.kernels_api_client.save_kernel.call_args.args[0]
        wire = json.loads(str(request))
        self.assertEqual(wire['competitionDataSources'], [prepare.COMPETITION])
        self.assertEqual(wire['machineShape'], 'NvidiaRtxPro6000')
        self.assertEqual(wire['sessionTimeoutSeconds'], 16200)
        self.assertTrue(wire['isPrivate'])


if __name__ == '__main__':
    unittest.main()
