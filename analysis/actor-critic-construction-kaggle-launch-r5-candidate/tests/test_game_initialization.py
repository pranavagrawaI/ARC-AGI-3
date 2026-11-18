"""Adjacent Arcade discovery/reset failures: actual app, zero admissions."""
from __future__ import annotations
import json
import subprocess
import sys
import pytest
from test_runtime import HERE, LOCAL_MODEL, STARTUP, materialize


@pytest.mark.parametrize('failure', ['duplicate_discovery', 'make_none', 'reset_none'])
def test_application_initialization_fails_closed(tmp_path, failure):
    ns, _ = materialize(tmp_path)
    capsule = ns['root']
    code = f'''import sys,json
from pathlib import Path
sys.path[:0]=[{str(capsule)!r},{str(HERE/'tests')!r}]
from game_mount import enter
enter(Path({str(tmp_path/'jail')!r}))
sys.dont_write_bytecode=True
import kaggle_probe
kaggle_probe.inspect_gpu=lambda out: None
# Real seal and imports; only the named post-identity initialization seam fails.
from arc_agi import Arcade
failure={failure!r}
if failure=='duplicate_discovery':
    original=Arcade._scan_for_environments
    def scan(self):
        original(self)
        self.available_environments.extend(self.available_environments[:1])
    Arcade._scan_for_environments=scan
else:
    original=Arcade.make
    def make(self,*a,**kw):
        env=original(self,*a,**kw)
        assert env is not None
        if failure=='make_none': return None
        env.reset=lambda: None
        return env
    Arcade.make=make
sys.argv=['kaggle_probe.py','--base-url','http://127.0.0.1:1/v1','--model',{LOCAL_MODEL!r},'--workspace-root',{str(tmp_path/'app')!r},'--output',{str(tmp_path/'result.json')!r},'--timeout-s','900','--startup-argv-json',{json.dumps(STARTUP)!r}]
try: kaggle_probe.main()
except Exception: pass
else: raise AssertionError('initialization defect accepted')
raw=Path({str(tmp_path/'app/raw')!r})
result=json.loads((raw/'STARTUP-FAILURE.json').read_text())
assert result['status']=='failed'
assert json.loads((raw/'GAME-PREFLIGHT.json').read_text())['status']=='passed'
assert json.loads((raw/'SEAL.json').read_text())['status']=='passed'
assert not (raw/'requests.jsonl').exists()
assert not (raw/'workspaces').exists()
assert not (raw/'cleanup.jsonl').exists(), 'relay retirement unresolved'
print(json.dumps(result))
'''
    proc = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    print(proc.stdout.strip())
