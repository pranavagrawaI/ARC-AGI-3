"""Reviewer-only offline checks; production authority files are never created here."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

GATES=Path(__file__).resolve().parent.parent/'r10a-verification'
sys.path.insert(0,str(GATES))
from test_launch_control import world,gate,save,invoke

@pytest.mark.parametrize('field,value',[('manifest_sha256','0'*64),('notebook_sha256','0'*64),('launch_authorized',True),('launch_authorized',0)])
def test_package_approval_exact_bindings(world,field,value):
    record=gate.load(world.review);record[field]=value;save(world.review,record)
    with pytest.raises(gate.Hold):gate.validate_package()
    assert invoke(world)==2 and world.calls==[]
    assert not (world.run/'CONSUMED.json').exists()

def test_actual_verify_frozen_entry_is_read_only_and_cannot_release(world,monkeypatch,capsys):
    spec=importlib.util.spec_from_file_location('reviewer_verify_frozen',GATES/'verify_frozen.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    monkeypatch.setattr(mod,'gate',gate)
    for name in ('launch-review.json','hermes-release.json'):(world.run/name).unlink()
    before={str(p):p.read_bytes() for p in world.run.rglob('*') if p.is_file()}
    assert mod.main(['--run-dir',str(world.run)])==0
    assert 'FROZEN_BYTES_VERIFIED_NOT_LAUNCH_APPROVAL' in capsys.readouterr().out
    assert before=={str(p):p.read_bytes() for p in world.run.rglob('*') if p.is_file()}
    assert invoke(world)==2 and world.calls==[]
    (world.package/'construction.ipynb').write_text('drift')
    assert mod.main(['--run-dir',str(world.run)])==2
    assert not (world.run/'CONSUMED.json').exists()
