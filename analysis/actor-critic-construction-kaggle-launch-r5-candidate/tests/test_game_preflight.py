"""Public-byte provenance and generated startup/application rejection boundaries."""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
import pytest
from game_mount import PUBLIC, copy_game
from test_runtime import HERE, ROOT, LOCAL_MODEL, STARTUP, load, materialize

MUTATIONS = {
    'missing': "(game / 'ls20.py').unlink()",
    'wrong_bytes': "p=game/'ls20.py'; raw=p.read_bytes(); p.write_bytes(b'#'+raw[1:])",
    'truncated': "(game/'ls20.py').write_bytes(b'')",
    'malformed_metadata': "p=game/'metadata.json'; raw=p.read_bytes(); p.write_bytes(b'!'+raw[1:])",
    'wrong_metadata_id': "p=game/'metadata.json'; p.write_text(p.read_text().replace('ls20-9607627b','ls20-00000000'))",
    'source_symlink': "p=game/'ls20.py'; p.unlink(); p.symlink_to('/does-not-exist')",
    'directory_symlink': "p=game.parent; p.rename(p.with_name('moved')); p.symlink_to(p.with_name('moved'), target_is_directory=True)",
    'ambiguous_version': "(game.parent/'00000000').mkdir()",
    'nested_metadata': "(game/'duplicate').mkdir(); (game/'duplicate/metadata.json').write_text('{}')",
    'fifo': "p=game/'ls20.py'; p.unlink(); __import__('os').mkfifo(p)",
}


def test_public_bytes_match_frozen_manifest_and_not_checkout(tmp_path):
    gate = load(HERE / 'game_preflight.py', 'public_gate')
    manifest = ROOT / 'analysis/deployment-probes/20260901T054322Z-public25-kaggle-v1/public-environment-manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == gate.PUBLIC_MANIFEST_SHA256
    rows = {r['path']: r for r in json.loads(manifest.read_text())['files']}
    for name, (size, digest) in gate.FILES.items():
        raw = (PUBLIC / name).read_bytes()
        assert len(raw) == size and hashlib.sha256(raw).hexdigest() == digest
        assert rows['environment_files/ls20/9607627b/' + name] == {'path': 'environment_files/ls20/9607627b/' + name, 'bytes': size, 'sha256': digest}
    game = copy_game(tmp_path)
    result = gate.validate(game.parents[1])
    assert result['game_id'] == gate.GAME_ID
    # Other games are neither discovered nor imported by this allowlisted gate.
    (game.parents[1] / 'unrelated').symlink_to('/does-not-exist')
    assert gate.validate(game.parents[1]) == result
    with pytest.raises((ValueError, OSError)):
        gate.validate(ROOT / 'environment_files')


@pytest.mark.parametrize('stage', ['bootstrap', 'application'])
@pytest.mark.parametrize('mutation', list(MUTATIONS))
def test_generated_gate_refuses_before_installer_or_model_admission(tmp_path, stage, mutation):
    ns, _ = materialize(tmp_path)
    capsule = ns['root']
    code = f'''import sys,json
from pathlib import Path
sys.path[:0] = [{str(capsule)!r}, {str(HERE / 'tests')!r}]
from game_mount import enter
def mutate(game):
    {MUTATIONS[mutation]}
enter(Path({str(tmp_path / 'jail')!r}), mutate)
import donor_bootstrap,kaggle_probe
seen=[]
kaggle_probe.inspect_gpu=lambda out: seen.append('hardware-fixture')
kaggle_probe.require_seal=lambda out: seen.append('seal-fixture')
def install(*a,**kw):
    seen.append('INSTALLER')
    raise AssertionError('invalid game entered installer')
donor_bootstrap.runpy.run_path=install
sys.argv=['kaggle_probe.py','--base-url','http://127.0.0.1:1/v1','--model',{LOCAL_MODEL!r},'--workspace-root',{str(tmp_path/'app')!r},'--output',{str(tmp_path/'output.json')!r},'--timeout-s','900','--startup-argv-json',{json.dumps(STARTUP)!r}]
try:
    {'donor_bootstrap.main()' if stage == 'bootstrap' else 'kaggle_probe.main()'}
except (ValueError,OSError):
    pass
else:
    raise AssertionError('invalid source admitted')
receipt=Path({str(capsule/'GAME-PREFLIGHT.json') if stage == 'bootstrap' else str(tmp_path/'app/raw/GAME-PREFLIGHT.json')!r})
result=json.loads(receipt.read_text())
assert result['status']=='failed' and result['model_request_admitted'] is False
assert 'INSTALLER' not in seen
assert seen=={['hardware-fixture','seal-fixture'] if stage=='bootstrap' else []!r}
assert 'arc_agi' not in sys.modules
assert not Path({str(tmp_path/'app/raw/requests.jsonl')!r}).exists()
print(json.dumps({{'stage':{stage!r},'mutation':{mutation!r},'receipt':result,'reached':seen}}))
'''
    proc = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code], capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    print(proc.stdout.strip())


def test_generated_bootstrap_valid_default_identity_reaches_installer(tmp_path):
    ns, _ = materialize(tmp_path)
    capsule = ns['root']
    code = f'''import sys,json
from pathlib import Path
sys.path[:0]=[{str(capsule)!r},{str(HERE/'tests')!r}]
from game_mount import enter
enter(Path({str(tmp_path/'jail')!r}))
import donor_bootstrap,kaggle_probe
kaggle_probe.inspect_gpu=lambda out: None
kaggle_probe.require_seal=lambda out: None
class InstallerReached(Exception): pass
def install(path,**kw):
    assert path==str(Path({str(capsule)!r})/'runtime_install.py')
    assert json.loads((Path({str(capsule)!r})/'GAME-PREFLIGHT.json').read_text())['status']=='passed'
    raise InstallerReached()
donor_bootstrap.runpy.run_path=install
try: donor_bootstrap.main()
except InstallerReached: print('exact-default-public-source-before-installer; model-not-started')
else: raise AssertionError('installer not reached')
'''
    proc = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code], capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stderr
    assert 'exact-default-public-source-before-installer' in proc.stdout


def test_r4_changes_only_discovery_gate_and_identity():
    PACKAGE = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r4'
    prior = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r3'
    old = json.loads((prior / 'ASSET-HASHES.json').read_text())
    new = json.loads((PACKAGE / 'ASSET-HASHES.json').read_text())
    assert new.keys() - old.keys() == {'game_preflight.py'}
    assert old.keys() - new.keys() == set()
    assert {n for n in old if old[n] != new[n]} == {'donor_bootstrap.py', 'kaggle_probe.py', 'EXPERIMENT-CARD.md', 'nooa-join/CHASSIS.json'}
    before = json.loads((prior / 'MANIFEST.json').read_text())
    after = json.loads((PACKAGE / 'MANIFEST.json').read_text())
    assert before['caps'] == after['caps']
    assert before['driver'] == after['driver']
    meta = json.loads((PACKAGE / 'kernel-metadata.json').read_text())
    baseline = json.loads((prior / 'kernel-metadata.json').read_text())
    assert meta.pop('id') == 'pranavagrawai/actor-critic-construction-kaggle-r4'
    assert meta.pop('title') == 'actor-critic-construction-kaggle-r4'
    baseline.pop('id')
    baseline.pop('title')
    assert meta == baseline
    assert not any(n.endswith('/ls20.py') or n == 'ls20.py' for n in new)
