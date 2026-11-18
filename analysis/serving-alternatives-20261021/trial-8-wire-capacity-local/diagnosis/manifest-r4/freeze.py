"""Explicit inherited immutable-input allowlist; never recursively freeze the live root."""
import datetime as dt
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
old = json.loads((HERE / 'freeze-r3.json').read_text())
snapshot = HERE / 'superseded-review-5d64f9b0'
assert sha(snapshot / 'freeze-r3.json') == '5d64f9b01fb69929b14c0eb22f309a1c3434fb64771b43b5dcfecd5d97afe882'
outputs = {'quota-prelaunch.json', 'sessions-prelaunch.json', 'predecessor-status-prelaunch.json',
           'failed-save-reconciliation-prelaunch.json'}
for name, expected in old['files'].items():
    base = HERE if '..' in Path(name).parts else snapshot
    assert sha(base / name) == expected, ('r3 preservation', name)
# Keep the reviewed immutable list. Explicitly remove live audit OUTPUTS. New
# evidence is added by name; no glob/rglob of the current operational namespace.
files = {name: sha(HERE / name) for name in old['files'] if name not in outputs}
for name in old['files']:
    if '..' not in Path(name).parts:
        archived = 'superseded-review-5d64f9b0/' + name
        files[archived] = sha(HERE / archived)
additions = ['freeze-r3.json', 'FREEZE-R3-RECEIPT.json',
             'superseded-review-5d64f9b0/freeze-r3.json',
             'superseded-review-5d64f9b0/FREEZE-R3-RECEIPT.json',
             'superseded-review-5d64f9b0/PRESERVATION.json',
             'tests/test_launch_sequence.py', 'diagnosis/manifest-r4/freeze.py',
             'diagnosis/manifest-r4/BOUNDARY.json', 'qualification/manifest-boundary-red.log',
             'qualification/manifest-boundary-green.log', 'qualification/tests-r4-final.log',
             'qualification/tests-r4-first-pass.log', 'qualification/manifest-r4-existing-header-test.log']
for name in additions:
    files[name] = sha(HERE / name)
forbidden = outputs | {'slot-confirmation.json', 'launch-attempt.json', 'launch-response.json', 'launch-error.json'}
assert not forbidden.intersection(files)
assert not any(Path(name).parts[0] in {'kernel', 'output', 'runtime', 'remote-source', 'collection'} for name in files)
freeze = dict(old, schema='local-wire-package-capacity-review-freeze-v4',
              status='LOCAL_REVIEW_READY_UNASSIGNED_R4', created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
              supersedes_freeze_sha256=sha(HERE / 'freeze-r3.json'), changed_payload_files=[],
              manifest_boundary='Immutable launch inputs and archived evidence only; live operational outputs excluded.',
              excluded_live_outputs=sorted(forbidden), files=dict(sorted(files.items())))
(HERE / 'freeze-r4.json').write_text(json.dumps(freeze, indent=2) + '\n')
receipt = dict(freeze_file='freeze-r4.json', freeze_sha256=sha(HERE / 'freeze-r4.json'),
               bindings=len(files), sources_sha256=freeze['sources_sha256'], config_sha256=freeze['config_sha256'],
               supersedes_freeze_sha256=freeze['supersedes_freeze_sha256'], status=freeze['status'])
(HERE / 'FREEZE-R4-RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
for name, expected in files.items():
    assert sha(HERE / name) == expected, name
print(json.dumps(receipt, indent=2))
