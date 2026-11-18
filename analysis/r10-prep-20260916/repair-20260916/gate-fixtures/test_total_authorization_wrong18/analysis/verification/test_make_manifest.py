"""Freeze authority adversaries in disposable worlds. No real approval or provider call."""
import importlib.util
from pathlib import Path
import pytest
from test_launch_control import world,gate,save,invoke

SPEC=importlib.util.spec_from_file_location('freeze_under_test',Path(__file__).with_name('make_manifest.py'))
maker=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(maker)

@pytest.fixture
def unfrozen(world,monkeypatch):
    monkeypatch.setattr(maker,'gate',gate)
    (world.package/'EXPERIMENT-CARD.md').write_bytes((world.run/'EXPERIMENT-CARD.md').read_bytes())
    world.freeze()
    for name in ('manifest.json','authorization.json','launch-review.json','hermes-release.json'):
        (world.run/name).unlink()
    return world

def test_freeze_requires_package_approval_before_writing(unfrozen):
    save(unfrozen.review,{'status':'HOLD'})
    assert maker.main(['--run-dir',str(unfrozen.run)])==2
    assert not (unfrozen.run/'manifest.json').exists()
    assert not (unfrozen.run/'authorization.json').exists()
    assert unfrozen.calls==[]

def test_freeze_exclusive_then_still_requires_launch_review_release(unfrozen):
    assert maker.main(['--run-dir',str(unfrozen.run)])==0
    assert maker.main(['--run-dir',str(unfrozen.run)])==2
    assert invoke(unfrozen)==2 and unfrozen.calls==[]
    assert not (unfrozen.run/'CONSUMED.json').exists()

@pytest.mark.parametrize('kind',['missing_review','wrong_inventory','wrong_notebook','card_drift','bytecode','symlink','unreviewed_file'])
def test_freeze_input_adversaries(unfrozen,kind):
    w=unfrozen
    if kind=='missing_review':w.review.unlink()
    elif kind=='wrong_inventory':gate.INVENTORY.write_text('{}')
    elif kind=='wrong_notebook':(w.package/'construction.ipynb').write_text('drift')
    elif kind=='card_drift':(w.run/'EXPERIMENT-CARD.md').write_text('different')
    elif kind=='bytecode':(w.package/'stray.pyc').write_bytes(b'bad')
    elif kind=='symlink':(w.package/'alias').symlink_to(w.package/'construction.ipynb')
    else:(w.package/'new.py').write_text('unreviewed')
    assert maker.main(['--run-dir',str(w.run)])==2
    assert not (w.run/'manifest.json').exists() and w.calls==[]

def test_partial_freeze_failure_retains_no_retry_marker(unfrozen,monkeypatch):
    original=gate.exclusive
    def interrupt(path,value):
        if path.name=='authorization.json':raise OSError('fixture interruption')
        return original(path,value)
    monkeypatch.setattr(gate,'exclusive',interrupt)
    assert maker.main(['--run-dir',str(unfrozen.run)])==2
    assert (unfrozen.run/'manifest.json').exists()
    assert maker.main(['--run-dir',str(unfrozen.run)])==2
    assert unfrozen.calls==[]

def test_approval_rechecked_at_freeze_boundary(unfrozen,monkeypatch):
    original=gate.validate_package;calls=[]
    def revoke():
        calls.append(1)
        if len(calls)==2:save(unfrozen.review,{'status':'HOLD'})
        return original()
    monkeypatch.setattr(gate,'validate_package',revoke)
    assert maker.main(['--run-dir',str(unfrozen.run)])==2
    assert not (unfrozen.run/'manifest.json').exists() and unfrozen.calls==[]

@pytest.mark.parametrize('mutation',['missing','status','authority','manifest_sha256','notebook_sha256','target','identity','max_pushes','retries'])
def test_hermes_release_is_required_before_any_provider_read(world,mutation):
    path=world.run/'hermes-release.json'
    if mutation=='missing':path.unlink()
    else:
        release=gate.load(path);release[mutation]='wrong';save(path,release)
    assert invoke(world)==2 and world.calls==[]
    assert not (world.run/'CONSUMED.json').exists()

def test_hermes_release_rechecked_after_provider_reads(world):
    import subprocess
    def revoke(argv,kwargs):
        (world.run/'hermes-release.json').unlink()
        return subprocess.CompletedProcess(argv,0,'Not found','')
    world.responses[:]=[(0,'[{"resource":"GPU","remaining":"1.00h"}]'),revoke]
    assert invoke(world)==2
    assert not any('push' in x for x in world.calls)
    assert not (world.run/'CONSUMED.json').exists()
