"""Read-only final R6 delta verification; never calls a provider or approves."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
REVIEW = Path(__file__).resolve().parent
RUN = ROOT / 'analysis/actor-critic-construction-kaggle-20260915T091353Z'
EXPECTED = '4ef8edfec9f589a2ad4331ce0de4a0ce4d47bf46efdf87abbef440039c53ad72'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('reviewed_gate', REVIEW.parent / 'launch_control.py')
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
assert sha(RUN / 'manifest.json') == EXPECTED
manifest = gate.load(RUN / 'manifest.json')
assert len(manifest['sources']) == 114
for name, digest in manifest['sources'].items():
    assert sha(ROOT / name) == digest, name
assert sha(RUN / 'EXPERIMENT-CARD.md') == 'd26e3b6ff7d4267f116d3119cade5735f71764c12cb47863a0a0ee710dc7f5d5'
assert sha(gate.PACKAGE_DIR / 'MANIFEST.json') == gate.LOCAL_MANIFEST_HASH
assert sha(gate.PACKAGE_DIR / 'construction.ipynb') == gate.NOTEBOOK_HASH
allowed = {'IDENTITY', 'PACKAGE_DIR', 'LOCAL_REVIEW', 'TARGET', 'NOTEBOOK_HASH', 'LOCAL_MANIFEST_HASH'}
def normalized(path):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in allowed for t in n.targets))]
    return ast.dump(tree, include_attributes=False)
assert normalized(REVIEW.parent / 'launch_control.py') == normalized(ROOT / 'analysis/actor-critic-construction-kaggle-verification-r5/launch_control.py')
xml = ET.parse(REVIEW / 'TEST.xml').getroot()
suites = list(xml.iter('testsuite'))
counts = {key: sum(int(s.get(key, 0)) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
assert counts == dict(tests=119, failures=0, errors=0, skipped=0), counts
approval = dict(status='APPROVED', manifest_sha256=EXPECTED)
original_load = gate.load
def hypothetical_load(path):
    return approval if Path(path) == RUN / 'launch-review.json' else original_load(path)
with patch.object(gate, 'load', hypothetical_load), patch.object(gate.subprocess, 'run', side_effect=AssertionError('provider prohibited')):
    gate.validate(str(RUN))
assert not os.path.lexists(RUN / 'CONSUMED.json')
print(json.dumps(dict(status='VERIFIED', manifest_sha256=EXPECTED, source_hashes_verified=114, drift=[], gate_logic_and_caps_unchanged=True, exact_card_auth_package_localreview=True, tests=counts, proposed_approval_passes_actual_predicate=True, consumed=False, provider_calls=0), indent=2))
