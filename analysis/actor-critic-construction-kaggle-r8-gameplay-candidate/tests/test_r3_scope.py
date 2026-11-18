"""R3 is a trusted bootstrap exec repair, not another application adaptation."""
from __future__ import annotations
import json
from test_runtime import ROOT

PACKAGE = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r3'


def test_r3_only_changes_trusted_bootstrap_and_identity_assets():
    prior = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r2'
    old = json.loads((prior / 'ASSET-HASHES.json').read_text())
    new = json.loads((PACKAGE / 'ASSET-HASHES.json').read_text())
    assert old.keys() == new.keys()
    changed = {name for name in old if old[name] != new[name]}
    assert changed == {'donor_bootstrap.py', 'EXPERIMENT-CARD.md', 'nooa-join/CHASSIS.json'}
    before = json.loads((prior / 'MANIFEST.json').read_text())
    after = json.loads((PACKAGE / 'MANIFEST.json').read_text())
    assert before['caps'] == after['caps']
    assert before['driver'] == after['driver']
    assert after['identity'] == 'actor-critic-construction-kaggle-r3'
    metadata = json.loads((PACKAGE / 'kernel-metadata.json').read_text())
    baseline = json.loads((prior / 'kernel-metadata.json').read_text())
    assert metadata.pop('id') == 'pranavagrawai/actor-critic-construction-kaggle-r3'
    assert metadata.pop('title') == 'actor-critic-construction-kaggle-r3'
    baseline.pop('id')
    baseline.pop('title')
    assert metadata == baseline
