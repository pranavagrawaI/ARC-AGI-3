"""Read-only candidate checks; writes only this independent review directory."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREP = ROOT / 'analysis/actor-critic-construction-kaggle-launch-r4-candidate'
PACKAGE = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r4'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

freeze_hash = '08d7f97d979da108b8fcf011fa8feb2ab31c7b8bba4b38a42d55ef259a115451'
assert sha(PREP / 'FREEZE.json') == freeze_hash
freeze = json.loads((PREP / 'FREEZE.json').read_text())
assert len(freeze['files']) == freeze['file_count'] == 100
assert all(sha(ROOT / name) == digest for name, digest in freeze['files'].items())
provenance = load_module(PREP / 'verify.py', 'review_verify').verify()
xml = ET.parse(HERE / 'tests.xml')
cases = xml.findall('.//testcase')
assert len(cases) == 63
assert not xml.findall('.//failure') and not xml.findall('.//error') and not xml.findall('.//skipped')
commands = json.loads((HERE / 'commands.json').read_text())
assert all(row['returncode'] == 0 for row in commands)
old = json.loads((ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r3/ASSET-HASHES.json').read_text())
new = json.loads((PACKAGE / 'ASSET-HASHES.json').read_text())
changed = sorted(n for n in old if old[n] != new[n])
assert changed == ['EXPERIMENT-CARD.md', 'donor_bootstrap.py', 'kaggle_probe.py', 'nooa-join/CHASSIS.json']
assert set(new) - set(old) == {'game_preflight.py'}
gate = load_module(HERE.parent / 'launch_control.py', 'review_gate')
assert not os.path.lexists(gate.RUN_DIR / 'CONSUMED.json'), 'identity already consumed'
state = {name: {'exists': (gate.RUN_DIR / name).exists(), 'sha256': sha(gate.RUN_DIR / name) if (gate.RUN_DIR / name).is_file() else None} for name in ['CONSUMED.json', 'manifest.json', 'authorization.json', 'launch-review.json', 'launch.json']}
decision = {
    'status': 'APPROVED_LOCAL_PACKAGE',
    'manifest_sha256': 'aec9801405a8768a681313e65058dc4b9f679722ef800140fb2aceae49b9c2ad',
    'notebook_sha256': '1b24acd36481f09164ca8b518d724d58dee64c2c3cad1983d8cc7234bc95e8ef',
    'launch_authorized': False,
    'freeze_sha256': freeze_hash,
    'card_sha256': provenance['card_sha256'],
    'tests': {'required': 63, 'executed': 63, 'passed': 63, 'failed': 0, 'skipped': 0},
    'findings': [],
    'scope': 'Independent bounded R4 local package delta review versus R3; no launch approval.',
    'verified': [
        'Exact manifest/notebook/FREEZE hashes; all 100 freeze bindings reconciled.',
        '422 protected sources, 413 predecessor files and five public-identity bindings reconciled without drift.',
        'Public LS20 source and metadata match exact byte lengths and pinned public manifest; coordinator retrieval receipts inspected, not independently fetched.',
        'Generated bootstrap rejects invalid source before installer/model; application independently rechecks before admissions.',
        'Actual packaged worker executes exact public bytes on default read-only competitions mount without --environments override.',
        'Full-ID singleton OFFLINE discovery and initialization failures reject before worker admissions.',
        'Policy, caps, isolation, worker, constructor, accounting, supervisor and trusted driver-loader controls unchanged outside named discovery/identity delta.',
        'Exact decision required fields checked with actual launch_control.matches/load predicate; no full launch authorization asserted.'
    ],
    'limitations': [
        'Hardware/model context is explicitly synthetic or intercepted; no R4 target GPU/model installation or inference qualification.',
        'No atomic hostile concurrent replacement claim for trusted read-only input mount.',
        'Coordinator retains sole launch ownership; launch authorization and launch-review remain separate.'
    ],
    'provider_calls': 0,
    'launches': 0,
    'evidence': ['commands.json', 'tests.stdout', 'tests.stderr', 'tests.xml', 'provenance.stdout', 'provenance.stderr', 'final-check.json']
}
assert provenance['manifest_sha256'] == decision['manifest_sha256']
assert provenance['notebook_sha256'] == decision['notebook_sha256']
expected = dict(status='APPROVED_LOCAL_PACKAGE', manifest_sha256=gate.LOCAL_MANIFEST_HASH, notebook_sha256=gate.NOTEBOOK_HASH, launch_authorized=False)
gate.matches(decision, expected)
assert not os.path.lexists(gate.RUN_DIR / 'CONSUMED.json')
(HERE / 'DECISION.json').write_text(json.dumps(decision, indent=2) + '\n')
gate.matches(gate.load(HERE / 'DECISION.json'), expected)
result = {'status': 'passed', 'freeze_files': len(freeze['files']), 'tests': len(cases), 'provenance': provenance, 'changed_existing_assets': changed, 'lifecycle_snapshot': state, 'decision_predicate': 'passed', 'launch_authorized': False}
(HERE / 'final-check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
