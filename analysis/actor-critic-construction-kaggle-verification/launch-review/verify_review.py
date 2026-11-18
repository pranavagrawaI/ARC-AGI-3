"""Read-only exact launch review; provider subprocess is a failing recorder."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'analysis/actor-critic-construction-kaggle-20260914T174335Z'
GATE = ROOT / 'analysis/actor-critic-construction-kaggle-verification/launch_control.py'
EXPECTED = '0f01cc456dc452a1895dcd30e8594a7c9deee55e0f90c4cc8c227295d84b8aab'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
m = json.loads((RUN / 'manifest.json').read_text())
assert sha(RUN / 'manifest.json') == EXPECTED
rows = [{'path': name, 'expected': digest, 'actual': sha(ROOT / name)} for name, digest in m['sources'].items()]
assert len(rows) == 106
assert all(row['expected'] == row['actual'] for row in rows)
assert not os.path.lexists(RUN / 'CONSUMED.json')
assert not os.path.lexists(RUN / 'launch.json')
spec = importlib.util.spec_from_file_location('review_exact_gate', GATE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
provider_calls = []
def recorder(*args, **kwargs):
    provider_calls.append(repr(args))
    raise AssertionError('Provider call forbidden during independent review')
real_load = gate.load
prospective = {'status': 'APPROVED', 'manifest_sha256': EXPECTED}
def prospective_load(path):
    return prospective if Path(path) == RUN / 'launch-review.json' else real_load(path)
with patch.object(gate.subprocess, 'run', recorder):
    with patch.object(gate, 'load', prospective_load):
        assert gate.main(['check', '--run-dir', str(RUN)]) == 0
assert provider_calls == []
metadata = json.loads((Path(m['package_dir']) / 'kernel-metadata.json').read_text())
# Import only the installed slugifier, not Kaggle's authenticating package.
from slugify import slugify
assert slugify(metadata['title']) == metadata['id'].split('/')[1]
assert metadata['docker_image'] == 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
assert not os.path.lexists(RUN / 'CONSUMED.json')
print(json.dumps({'status': 'PASS', 'manifest_sha256': EXPECTED, 'source_count': len(rows), 'sources': rows, 'provider_calls': provider_calls, 'prospective_exact_gate_exit': 0, 'title_slug': slugify(metadata['title']), 'consumed': False}, indent=2))
