"""Detect v2's exact package mismatch locally, then validate the built notebook."""
import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import nbformat
import build
from package_contract import verify_bundle, verify_installed, verify_wheel

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE/'native-contract.json').read_text())
PACKAGES = HERE/'evidence/packages'
WHEEL = PACKAGES/CONTRACT['wheel']['filename']


class PackageContract(unittest.TestCase):
    def test_real_package_matches_observed_installed_runtime(self):
        self.assertEqual(verify_bundle(PACKAGES, PACKAGES/'requirements.lock', CONTRACT), CONTRACT['wheel'])
        observed = HERE.parent/'serving-headroom-v2-20261008/readout/raw/headroom/runtime-source'
        self.assertEqual(verify_installed(observed, CONTRACT), CONTRACT['files'])
        self.assertEqual(verify_installed(HERE/'native-source', CONTRACT), CONTRACT['files'])
        self.assertEqual(len(CONTRACT['provenance']['changed_from_git_fixture']), 1)

    def test_v2_git_fixture_mismatch_is_detected_before_gpu(self):
        old = json.loads(subprocess.check_output(['git','show',
            '2033fba:analysis/serving-headroom-v2-20261008/native-contract.json']))
        wrong = copy.deepcopy(CONTRACT)
        wrong['files'] = old['files']
        with self.assertRaisesRegex(ValueError, 'wheel source mismatch: srt/mem_cache/kv_cache_configurator.py'):
            verify_wheel(WHEEL, wrong)
        with self.assertRaisesRegex(ValueError, 'installed source mismatch'):
            verify_installed(HERE/'native-source', wrong)

    def test_drifted_wheel_or_lock_and_extra_wheel_are_rejected(self):
        wrong = copy.deepcopy(CONTRACT); wrong['wheel']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'wheel identity mismatch'):
            verify_wheel(WHEEL, wrong)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/WHEEL.name).symlink_to(WHEEL.resolve())
            lock = root/'requirements.lock'; lock.write_text('drift')
            with self.assertRaisesRegex(ValueError, 'requirements lock mismatch'):
                verify_bundle(root, lock, CONTRACT)
            (root/'sglang-extra.whl').write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError, 'one packaged SGLang'):
                verify_bundle(root, lock, CONTRACT)

    def test_actual_notebook_schema_and_embedded_guard(self):
        build.build()
        path = HERE/'kernel'/(build.SLUG+'.ipynb')
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'markdown':
                self.assertNotIn('outputs',cell)
                self.assertNotIn('execution_count',cell)
        source = ''.join(notebook.cells[3].source)
        self.assertIn((HERE/'package_contract.py').read_text(), source)
        # Run the actual appended preinstall guard against a local bundle layout.
        prefix = ''.join(json.loads((build.PARENT/'kernel/mtp-depth-sweep-retry1-20261007.ipynb').read_text())['cells'][3]['source'])
        guard = source[len(prefix):]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'wheels').mkdir();out=root/'out';out.mkdir()
            (root/'wheels'/WHEEL.name).symlink_to(WHEEL.resolve())
            shutil.copy2(PACKAGES/'requirements.lock',root/'requirements.lock')
            namespace=dict(WHEELHOUSE_DIR=str(root),OUT=out,json=json)
            exec(compile(guard,'<actual-package-guard>','exec'),namespace)
            self.assertEqual(json.loads((out/'package-identity.json').read_text()), CONTRACT['wheel'])
            (root/'requirements.lock').write_text('drift')
            with self.assertRaisesRegex(ValueError,'requirements lock mismatch'):
                exec(compile(guard,'<actual-package-guard>','exec'),namespace)


if __name__ == '__main__': unittest.main()
