"""Independent local-only final review; no provider calls or source edits."""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
GATE = EVIDENCE.parent / 'launch_control.py'
spec = importlib.util.spec_from_file_location('independent_r4_gate', GATE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
RUN = gate.RUN_DIR
PREP = ROOT / 'analysis/actor-critic-construction-kaggle-launch-r4-candidate'
EXPECTED = '687e70fdfcf92231b1f31da0ddba3726bbd7b7b99d6652879cc89314e9ccff89'
CARD = '183a078aaf5a1abfe94a3bcfc3a4c79b9dc3888e6b4e5d3cd2f0fe24c53496cf'
assert gate.sha(RUN / 'manifest.json') == EXPECTED
m = gate.load(RUN / 'manifest.json')
assert len(m['sources']) == 105
bindings = [{'path': n, 'expected': h, 'actual': gate.sha(ROOT / n)} for n, h in sorted(m['sources'].items())]
assert all(row['actual'] == row['expected'] for row in bindings)
assert gate.sha(RUN / 'EXPERIMENT-CARD.md') == CARD == m['card_sha256']
local = gate.load(gate.LOCAL_REVIEW)
assert gate.sha(PREP / 'MANIFEST.json') == gate.LOCAL_MANIFEST_HASH == local['manifest_sha256']
assert gate.sha(PREP / 'FREEZE.json') == local['freeze_sha256']
assert gate.sha(PREP / 'EXPERIMENT-CARD.md') == local['card_sha256']
assert gate.sha(gate.PACKAGE_DIR / 'construction.ipynb') == local['notebook_sha256'] == gate.NOTEBOOK_HASH
old = gate.load(ROOT / 'analysis/actor-critic-construction-kaggle-20260914T192100Z/manifest.json')
for key in m['policy']:
    if key not in ('identity', 'target'):
        assert m['policy'][key] == old['policy'][key], key
allowed = {'IDENTITY', 'PACKAGE_DIR', 'LOCAL_REVIEW', 'TARGET', 'NOTEBOOK_HASH', 'LOCAL_MANIFEST_HASH'}
def normalized(path):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in allowed for t in n.targets))]
    return ast.dump(tree, include_attributes=False)
assert normalized(GATE) == normalized(ROOT / 'analysis/actor-critic-construction-kaggle-verification-r3/launch_control.py')
suite = ET.parse(EVIDENCE / 'tests.xml').getroot().find('testsuite')
assert int(suite.attrib['tests']) == 119
assert all(int(suite.attrib[k]) == 0 for k in ('failures', 'errors', 'skipped'))
assert gate.load(EVIDENCE / 'test-command.json')['returncode'] == 0
proposal = dict(status='APPROVED', manifest_sha256=EXPECTED, card_sha256=CARD,
                notebook_sha256=gate.NOTEBOOK_HASH, source_bindings_verified=105,
                gate_tests=dict(required=119, executed=119, passed=119, failed=0, skipped=0),
                gate_logic_and_caps_unchanged_vs_r3=True, provider_calls=0, launches=0,
                scope='Independent final bounded R4 launch delta review; coordinator sole launch owner.',
                limitations=['No R4 hardware/model qualification.', 'Exact-version source/status limitations remain qualified by the card.', 'Provider quota/target absence checks remain coordinator launch-gate preflight; not run by reviewer.', 'Pre-existing pytest temporary-fixture permission cleanup warnings are nonblocking.'])
original_load = gate.load
def proposed_load(path):
    return proposal if Path(path) == RUN / 'launch-review.json' else original_load(path)
# Exercise all actual predicates without placing approval prematurely.
with patch.object(gate, 'load', proposed_load), patch.object(gate.subprocess, 'run', side_effect=AssertionError('provider call forbidden')):
    assert gate.validate(str(RUN)) == m
(EVIDENCE / 'source-bindings.json').write_text(json.dumps(bindings, indent=2) + '\n')
# Re-read mutable bindings and consumption immediately before exclusive approval.
with patch.object(gate, 'load', proposed_load):
    gate.validate(str(RUN))
assert gate.sha(RUN / 'manifest.json') == EXPECTED
assert not os.path.lexists(RUN / 'CONSUMED.json')
assert not os.path.lexists(RUN / 'launch.json')
assert not os.path.lexists(RUN / 'launch-review.json')
proposal['consumption_absent_checked_unix_ns'] = time.time_ns()
assert not os.path.lexists(RUN / 'CONSUMED.json')
gate.exclusive(RUN / 'launch-review.json', proposal)
assert original_load(RUN / 'launch-review.json') == proposal
# Unmocked actual CLI read-only gate; check never invokes provider commands.
cmd = [str(ROOT / 'deployment/kaggle/.venv/bin/python'), '-B', str(GATE), 'check', '--run-dir', str(RUN)]
result = subprocess.run(cmd, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL)
receipt = dict(argv=cmd, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
               manifest_sha256=gate.sha(RUN / 'manifest.json'), consumption_absent=not os.path.lexists(RUN / 'CONSUMED.json'))
(EVIDENCE / 'gate-check.json').write_text(json.dumps(receipt, indent=2) + '\n')
assert result.returncode == 0, receipt
assert json.loads(result.stdout) == dict(status='GREEN_LOCAL_GATE', provider_checks='NOT_RUN')
assert receipt['consumption_absent']
assert receipt['manifest_sha256'] == EXPECTED
print(json.dumps(dict(status='APPROVED', source_bindings=105, gate_tests_passed=119, gate_check=receipt), indent=2))
