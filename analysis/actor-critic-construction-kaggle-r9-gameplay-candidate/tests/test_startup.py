"""Generated startup containment and hardware-discovery regression fixtures."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import pytest
from test_runtime import materialize, load


@pytest.mark.parametrize('kind', ['missing', 'loader_failure', 'malformed', 'wrong_device', 'synthetic_positive'])
def test_gpu_discovery_executes_scrubbed_binary_and_preserves_failure(tmp_path, monkeypatch, kind):
    ns, _ = materialize(tmp_path)
    root = ns['root']
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.delitem(sys.modules, 'accelerator', raising=False)
    import accelerator
    probe = load(root / 'gameplay_play.py', 'gpu_probe_fixture')
    executable = tmp_path / 'synthetic-nvidia-smi'
    if kind != 'missing':
        stdout = 'malformed'
        if kind in ('wrong_device', 'synthetic_positive'):
            name = 'Tesla P100' if kind == 'wrong_device' else 'NVIDIA RTX PRO 6000 Blackwell Server Edition'
            stdout = name + ', GPU-01234567-0123-0123-0123-0123456789ab, 580.1, 98304 MiB'
        script = '#!/bin/sh\ntest -z "$SYNTHETIC_CREDENTIAL" || exit 90\n'
        if kind == 'loader_failure':
            script += 'printf "synthetic libnvidia-ml.so load failure" >&2\nexit 12\n'
        else:
            script += 'printf "%s\\n" ' + repr(stdout) + '\n'
        executable.write_text(script)
        executable.chmod(0o755)
    monkeypatch.setenv('SYNTHETIC_CREDENTIAL', 'must-not-reach-trusted-query')
    discover = accelerator.discover
    monkeypatch.setattr(accelerator, 'discover', lambda execute: discover(execute, paths=[str(executable)]))
    out = tmp_path / 'evidence'
    out.mkdir()
    if kind == 'synthetic_positive':
        assert probe.inspect_gpu(out)['device_count'] == 1
    else:
        with pytest.raises(ValueError, match='ACCELERATOR'):
            probe.inspect_gpu(out)
    record = json.loads((out / 'GPU.json').read_text())
    assert record['status'] == ('passed' if kind == 'synthetic_positive' else 'failed')
    assert 'SYNTHETIC_CREDENTIAL' not in record['discovery']['trusted_environment']
    if kind == 'loader_failure':
        assert record['discovery']['returncode'] == 12
        assert 'load failure' in record['discovery']['stderr']
    assert not (out / 'SEAL.json').exists()


def test_containment_inherited_stdin_fd_and_environment(tmp_path):
    ns, _ = materialize(tmp_path)
    root = ns['root']
    canary = tmp_path / 'outside-future'
    canary.write_text('synthetic-future')
    fd = os.open(canary, os.O_RDONLY)
    os.set_inheritable(fd, True)
    code = f'''
import os,json,sys
sys.path.insert(0,{str(root)!r})
from integration_boundary import CandidateRPC,SealedBoundary
assert os.read({fd},100) == b'synthetic-future'
source = ''' + repr(f'''
import os
def probe():
    try:
        os.read({fd}, 100)
        fd_denied = False
    except OSError:
        fd_denied = True
    try:
        open({str(canary)!r}).read()
        future_denied = False
    except OSError:
        future_denied = True
    return {{'fd_denied':fd_denied, 'future_denied':future_denied, 'stdin':os.read(0,100).decode(), 'key':os.environ.get('SYNTHETIC_KEY')}}
def initialize(obs):return probe()
def predict(state,action):return []
def assimilate(state,obs,verdict):return state
''') + '''
rpc=CandidateRPC(SealedBoundary(), source).initialize({})
inspection=SealedBoundary().inspect(source+'\nimport json\nprint(json.dumps(probe()))', {'visible': 'prefix'})
print(json.dumps({'rpc':rpc, 'inspection':json.loads(inspection['stdout'])}))
'''.replace("source+'\nimport json\nprint(json.dumps(probe()))'", "source+'\\nimport json\\nprint(json.dumps(probe()))'")
    try:
        proc = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code], input=b'synthetic-stdin', pass_fds=(fd,), env={**os.environ, 'SYNTHETIC_KEY': 'synthetic-secret'}, capture_output=True, timeout=15)
        assert proc.returncode == 0, proc.stderr.decode()
        record = json.loads(proc.stdout)
        expected = {'fd_denied': True, 'future_denied': True, 'stdin': '', 'key': None}
        assert record == {'rpc': expected, 'inspection': expected}
    finally:
        os.close(fd)


def test_guard_startup_failure_is_durable_and_no_children(tmp_path):
    ns, _ = materialize(tmp_path)
    receipt = tmp_path / 'guard.json'
    proc = subprocess.run([sys.executable, '-I', '-S', '-B', str(ns['root'] / 'process_guard.py'), str(os.getpid()), '5', str(receipt), '/nonexistent/synthetic-command'], capture_output=True, timeout=15)
    assert proc.returncode == 1
    result = json.loads(receipt.read_text())
    assert result['status'] == 'failed'
    assert result['cleanup'] == 'retired'
    assert result['killed_remaining_descendants'] == []


def test_failed_seal_prevents_install_and_model_start(tmp_path):
    ns, _ = materialize(tmp_path)
    root = ns['root']
    code = f'''import sys,json
from pathlib import Path
sys.path.insert(0,{str(root)!r})
import donor_bootstrap, gameplay_play, containment_base
gameplay_play.inspect_gpu = lambda out: None  # explicit unavailable-hardware context
# Inject one sealing primitive failure, rather than returning a synthetic qualification.
def fail(*args, **kwargs): raise RuntimeError('synthetic rootfs acquisition failure')
containment_base.rootfs = fail
try:
    donor_bootstrap.main()
except RuntimeError as error:
    assert 'synthetic rootfs acquisition failure' in str(error)
else:
    raise AssertionError('failed sealing continued to installation/model')
record=json.loads((Path({str(root)!r})/'SEAL.json').read_text())
assert record['status']=='failed'
assert record['cases']=={{}}
print('actual-seal-failure-blocked-install')
'''
    proc = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code], capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == 'actual-seal-failure-blocked-install'
