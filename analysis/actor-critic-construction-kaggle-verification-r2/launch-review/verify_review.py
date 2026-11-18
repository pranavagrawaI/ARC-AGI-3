"""Bounded offline DELTA review. Writes only review reports and explicit approval."""
import contextlib
import difflib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
NEW = REPORT.parent
OLD = ROOT / 'analysis/actor-critic-construction-kaggle-verification'
RUN = ROOT / 'analysis/actor-critic-construction-kaggle-20260914T183029Z'
EXPECTED = '1778641a826ed1d035eed700a3d63d87fa99a83e19faa6de4f44612eeafdd445'
APPROVAL = {'status': 'APPROVED', 'manifest_sha256': EXPECTED}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, value):
    (REPORT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def reconcile():
    assert sha(RUN / 'manifest.json') == EXPECTED
    m = json.loads((RUN / 'manifest.json').read_text())
    rows = [{'path': p, 'expected': h, 'actual': sha(ROOT / p)} for p, h in m['sources'].items()]
    assert len(rows) == 99
    assert all(r['expected'] == r['actual'] for r in rows)
    assert not os.path.lexists(RUN / 'CONSUMED.json')
    assert not os.path.lexists(RUN / 'launch.json')
    auth = json.loads((RUN / 'authorization.json').read_text())
    assert auth['used'] is False
    return m, rows, auth

def gate_load():
    spec = importlib.util.spec_from_file_location('independent_r2_gate', NEW / 'launch_control.py')
    assert spec is not None and spec.loader is not None
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate

def check(gate, prospective=False):
    stdout, stderr = io.StringIO(), io.StringIO()
    real_load = gate.load
    def load(p):
        if Path(p) == RUN / 'launch-review.json':
            return APPROVAL.copy()
        return real_load(p)
    with patch.object(gate.subprocess, 'run', side_effect=AssertionError('provider subprocess forbidden')) as provider, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        with patch.object(gate, 'load', side_effect=load) if prospective else contextlib.nullcontext():
            code = gate.main(['check', '--run-dir', str(RUN)])
    result = {'exit_code': code, 'stdout': stdout.getvalue(), 'stderr': stderr.getvalue(), 'provider_calls': provider.call_args_list, 'prospective_review_load_only': prospective}
    assert code == 0, result
    assert json.loads(result['stdout']) == {'status': 'GREEN_LOCAL_GATE', 'provider_checks': 'NOT_RUN'}
    assert not provider.call_args_list
    return result

if __name__ == '__main__':
    gate = gate_load()
    if sys.argv[1:] == ['finalize']:
        tests = json.loads((REPORT / 'gate-tests.json').read_text())
        assert tests['exit_code'] == 0 and '119 passed' in tests['stdout']
        m, rows, auth = reconcile()
        assert not os.path.lexists(RUN / 'launch-review.json')
        prospective = check(gate, prospective=True)
        # Re-read mutable controls and every source immediately before approval.
        m, rows, auth = reconcile()
        gate.matches(auth, dict(schema=1, identity=gate.IDENTITY, target=gate.TARGET,
            manifest_sha256=EXPECTED, card_sha256=m['card_sha256'], notebook_sha256=gate.NOTEBOOK_HASH,
            max_pushes=1, retries=0, timeout_seconds=3600, used=False))
        gate.exclusive(RUN / 'launch-review.json', APPROVAL)
        actual = check(gate)
        assert gate.load(RUN / 'launch-review.json') == APPROVAL
        reconcile()
        save('exact-real-gate.json', {'approval': APPROVAL, 'prospective': prospective, 'actual': actual,
             'consumed_before_approval': False, 'consumed_after_check': False,
             'launch_receipt_present': False, 'authorization_used': False, 'sources_rechecked': len(rows)})
        print(json.dumps({'status': 'APPROVED', 'actual_gate': actual, 'sources': len(rows)}))
    else:
        m, rows, auth = reconcile()
        assert not os.path.lexists(RUN / 'launch-review.json')
        assert sha(OLD / 'launch_control.py') == '3432379ab7cfa74c738edaa8613f7b6ce758b3f11a10308165fe81bb2ed0010b'
        assert sha(OLD / 'test_launch_control.py') == '6cc2650fffc1ea2123a391dc2c06fa464b588420272b8c3d33988b5648cb55c2'
        old = (OLD / 'launch_control.py').read_text()
        new = (NEW / 'launch_control.py').read_text()
        allowed = [
            ('actor-critic-construction-kaggle-20260914T174335Z', 'actor-critic-construction-kaggle-20260914T183029Z'),
            ('actor-critic-construction-kaggle-smoke', 'actor-critic-construction-kaggle-r2'),
            ('actor-critic-construction-kaggle-verification/repaired-review', 'actor-critic-construction-kaggle-verification-r2/repaired-review'),
            ('0fc78e34ff45d82e95d815bcd8b39ddef7c6c8067cd2775446df50dea3053962', gate.NOTEBOOK_HASH),
            ('234a62ec3727d6593043324f3f471877755da0fc146dfaef6d754375531213b3', gate.LOCAL_MANIFEST_HASH),
            ('warmup_calls=1, application_calls=23, actor_calls=20, actions=8,',
             'warmup_calls=1, supervisor_request_timeout_seconds=180, application_http_seconds=30,\n              application_calls=23, actor_calls=20, actions=8,')]
        normalized = old
        for a, b in allowed:
            assert a in normalized
            normalized = normalized.replace(a, b)
        assert normalized == new, 'unexpected gate drift'
        old_tests = (OLD / 'test_launch_control.py').read_text()
        new_tests = (NEW / 'test_launch_control.py').read_text()
        assert old_tests.replace('actor-critic-construction-kaggle-smoke', 'actor-critic-construction-kaggle-r2') == new_tests
        diff = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=str(OLD / 'launch_control.py'), tofile=str(NEW / 'launch_control.py')))
        (REPORT / 'gate-delta.diff').write_text(diff)
        result = check(gate, prospective=True)
        save('source-verification.json', {'manifest_sha256': EXPECTED, 'count': len(rows), 'mismatches': [],
            'sources': rows, 'prior_gate_and_tests_reviewed_hashes_match': True, 'allowed_delta_only': True,
            'prospective_gate': result, 'authorization_sha256': sha(RUN / 'authorization.json'),
            'consumed': False, 'launch_receipt': False})
        cmd = [str(ROOT / 'deployment/kaggle/.venv/bin/python'), '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               '--basetemp', str(REPORT / 'pytest-tmp'), str(NEW / 'test_launch_control.py')]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
        save('gate-tests.json', {'command': cmd, 'exit_code': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr})
        assert proc.returncode == 0 and '119 passed' in proc.stdout, proc.stdout + proc.stderr
        reconcile()
        print(json.dumps({'sources': len(rows), 'allowed_delta_only': True, 'prospective_gate': result, 'tests': proc.stdout}))
