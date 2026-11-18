import importlib.util
import json
import shutil
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / 'raw'
SESSION = RAW / 'nooa-join-public3/solver-workspaces/ls20-8md_11v1'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv('TYCHO_PUBLIC_GAME_IDS', 'ls20')
    pin = tmp_path / 'pin'; pin.mkdir()
    shutil.copy2(RAW / 'nooa-join/CHASSIS.json', pin / 'CHASSIS.json')
    monkeypatch.setenv('NOOA_JOIN_ROOT', str(pin))
    session = tmp_path / 'sessions/ls20-test'; session.mkdir(parents=True)
    for name in ['worker-metadata.json', 'launch.json']:
        shutil.copy2(SESSION / name, session / name)
    # Recreate only the empty game marker omitted by Kaggle's download.
    (session / 'workspace/ls20').mkdir(parents=True)
    (session / 'ipc').symlink_to(SESSION / 'ipc', target_is_directory=True)
    return load(HERE / 'nooa_public_probe.py', 'fixed_probe'), session, pin


def rewrite(path, key, value):
    obj = json.loads(path.read_text()); obj[key] = value
    path.write_text(json.dumps(obj))


def test_original_rejects_the_actual_successor(case):
    _, session, _ = case
    old = load(RAW / 'nooa-join/nooa_public_probe.py', 'old_probe')
    with pytest.raises(RuntimeError, match='chassis pin differs'):
        old._summarize_session(session, 'ls20')


def test_corrected_validator_accepts_actual_completed_evidence(case):
    fixed, session, _ = case
    report = fixed.summarize_nooa_sessions(session.parent)
    assert report['total_actor_calls'] == 155
    assert report['games']['ls20']['episodes'][0]['quick_metrics']['executed_actions'] == 100


def test_wrong_identity_still_fails(case):
    fixed, session, _ = case
    rewrite(session / 'worker-metadata.json', 'chassis', 'another-candidate')
    with pytest.raises(RuntimeError, match='chassis pin differs'):
        fixed._summarize_session(session, 'ls20')


def test_mutually_matching_but_wrong_contracts_fail(case):
    fixed, session, _ = case
    for name in ['worker-metadata.json', 'launch.json']:
        rewrite(session / name, 'contract_sha256', '0' * 64)
    with pytest.raises(RuntimeError, match='frozen manifest'):
        fixed._summarize_session(session, 'ls20')


def test_missing_manifest_fails_closed(case):
    fixed, session, pin = case
    (pin / 'CHASSIS.json').unlink()
    with pytest.raises(FileNotFoundError):
        fixed._summarize_session(session, 'ls20')


def test_identity_comes_from_manifest_not_another_literal(case):
    fixed, session, pin = case
    for identity in ['q6-stall-authority', 'future-successor']:
        rewrite(pin / 'CHASSIS.json', 'identity', identity)
        rewrite(session / 'worker-metadata.json', 'chassis', identity)
        assert fixed._summarize_session(session, 'ls20')['model_evidence']['actor_call_count'] == 155
