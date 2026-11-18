import copy
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import build
from test_supervisor import fixture
import supervisor


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        config = fixture(self.root)
        config.update(mode='GPU_SCREEN', purpose='CANDIDATE_FEASIBILITY', arm_order=['candidate'])
        config['hardware']['argv'] = supervisor.GPU_ARGV
        config['candidate']['functional'] = config['candidate'].pop('replay')
        config['candidate'].pop('ready_poll_seconds')
        del config['control']
        self.spec = dict(config=config, files={}, metadata=dict(id='owner/unlaunched-test', title='unlaunched test',
            code_file='unlaunched.ipynb', language='python', kernel_type='notebook', is_private=True,
            enable_gpu=True, enable_tpu=False, enable_internet=False, docker_image=build.IMAGE,
            machine_shape='NvidiaRtxPro6000', competition_sources=['arc-prize-2026-arc-agi-3'], dataset_sources=[], model_sources=[], kernel_sources=[]))

    def tearDown(self):
        self.temp.cleanup()

    def test_preparation_packet_is_not_executable_without_assignment(self):
        manifest = build.build(self.spec, self.root / 'packet')
        self.assertFalse(manifest['launch_authorized'])
        notebook = json.loads((self.root / 'packet/unlaunched.ipynb').read_text())
        source = ''.join(notebook['cells'][0]['source'])
        with self.assertRaisesRegex(RuntimeError, 'PREPARATION ONLY'):
            exec(compile(source, '<prepared-notebook>', 'exec'), {})
        self.assertFalse((self.root / 'supervisor.py').exists())
        self.assertEqual(manifest['total_seconds'], 8)

    def test_envelope_drift_rejected(self):
        self.spec['metadata']['enable_internet'] = True
        with self.assertRaisesRegex(ValueError, 'envelope mismatch'):
            build.build(self.spec, self.root / 'packet')

    def test_assignment_config_or_cap_drift_rejected(self):
        with self.assertRaisesRegex(ValueError, 'assignment mismatch'):
            build.build(self.spec, self.root / 'packet', dict(launch_authorized=True,
                serialized_assignment='test-only', config_sha256='wrong', total_seconds=8))

    def test_unsafe_filename_rejected(self):
        self.spec['files']['../escape'] = __file__
        with self.assertRaisesRegex(ValueError, 'unsafe packet filename'):
            build.build(self.spec, self.root / 'packet')

    def test_cannot_replace_reviewed_supervisor(self):
        self.spec['files']['supervisor.py'] = __file__
        with self.assertRaisesRegex(ValueError, 'override reviewed runner'):
            build.build(self.spec, self.root / 'packet')

    def test_assignment_binds_every_embedded_source(self):
        sha = hashlib.sha256(json.dumps(self.spec['config'], sort_keys=True).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'source freeze mismatch'):
            build.build(self.spec, self.root / 'packet', dict(launch_authorized=True,
                serialized_assignment='test-only', config_sha256=sha, total_seconds=8, sources_sha256='wrong'))

    def test_alternate_pinning_type_rejected(self):
        self.spec['metadata']['docker_image_pinning_type'] = 'original'
        with self.assertRaisesRegex(ValueError, 'pinning type'):
            build.build(self.spec, self.root / 'packet')


if __name__ == '__main__':
    unittest.main()
