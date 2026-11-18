"""Read-only bounded package review; writes evidence only beside this file."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PREP = ROOT / 'analysis/actor-critic-construction-kaggle-launch-r3'
PACKAGE = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r3'
PRIOR = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r2'
MANIFEST = '29f552e70dada5a70396cf2d701ab2e8faae050d83fab3f68e0f575a17823fc0'
NOTEBOOK = 'fefbf8a4a22781df03590280f9d3ec3703d4d435ed5569d2580374d46252dfb5'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def reconcile():
    checks = load_module('review_verify', PREP / 'verify.py').verify()
    assert checks['manifest_sha256'] == MANIFEST
    assert checks['notebook_sha256'] == NOTEBOOK
    freeze = json.loads((PREP / 'FREEZE.json').read_text())
    assert freeze['checks'] == checks
    for key, root in [('prep_files', PREP), ('package_files', PACKAGE)]:
        expected = freeze[key]
        actual = {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file() and p.name != 'FREEZE.json'}
        assert actual == expected, key
    old = json.loads((PRIOR / 'ASSET-HASHES.json').read_text())
    new = json.loads((PACKAGE / 'ASSET-HASHES.json').read_text())
    assert old.keys() == new.keys()
    for name, digest in old.items():
        assert sha(PRIOR / name) == digest
    changed = sorted(name for name in old if old[name] != new[name])
    assert changed == ['EXPERIMENT-CARD.md', 'donor_bootstrap.py', 'nooa-join/CHASSIS.json']
    before = json.loads((PRIOR / 'MANIFEST.json').read_text())
    after = json.loads((PACKAGE / 'MANIFEST.json').read_text())
    assert before['caps'] == after['caps']
    assert before['driver'] == after['driver']
    return dict(reconciliation=checks, freeze_exact=True, changed_assets=changed,
                unchanged_caps=after['caps'], unchanged_driver=after['driver'], freeze_sha256=sha(PREP / 'FREEZE.json'))

if __name__ == '__main__':
    decision = dict(status='HOLD', manifest_sha256=MANIFEST, notebook_sha256=NOTEBOOK,
                    launch_authorized=False, provider_calls=0, launches=0)
    try:
        evidence = reconcile()
        command = ['unshare', '--user', '--map-auto', '--map-root-user',
                   str(ROOT / 'deployment/kaggle/.venv/bin/python'), '-B', '-m', 'pytest', '-q', '-s',
                   '-p', 'no:cacheprovider', str(PREP / 'tests')]
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                                capture_output=True, text=True, timeout=180)
        (OUT / 'full-suite.stdout').write_text(result.stdout)
        (OUT / 'full-suite.stderr').write_text(result.stderr)
        evidence['command'] = command
        evidence['test_returncode'] = result.returncode
        assert result.returncode == 0 and '36 passed' in result.stdout
        assert '"revision": "r2"' in result.stdout and '"returncode": 12' in result.stdout
        assert '"revision": "r3"' in result.stdout and 'SYNTHETIC_SERVER_LOADER_OK' in result.stdout
        evidence['post_test_reconciliation'] = reconcile()
        gate = load_module('review_gate', OUT.parent / 'launch_control.py')
        assert not os.path.lexists(gate.RUN_DIR / 'CONSUMED.json'), 'identity consumed during review'
        evidence['mutable_state'] = {name: {'exists': (gate.RUN_DIR / name).exists(),
            'sha256': sha(gate.RUN_DIR / name) if (gate.RUN_DIR / name).is_file() else None}
            for name in ['CONSUMED.json', 'manifest.json', 'authorization.json', 'launch-review.json']}
        decision.update(status='APPROVED_LOCAL_PACKAGE', evidence='evidence.json',
            tests={'passed': 36, 'failed': 0, 'loader_regression_included': True},
            scope='Independent bounded R2-to-R3 trusted loader propagation delta; local package approval only.',
            findings=[], limitations=[
                'Synthetic compiled dlopen fixture, not NVIDIA/CUDA/SGLang or target hardware qualification.',
                'Real generated bootstrap invocation and supervisor CLI/GPU query exercised; R3 CLI still stops at lifecycle_observer_preflight in fixture.',
                'Exact generated server environment/Popen statements exercised separately in inherited environment; not full end-to-end startup.',
                'Generated CandidateRPC and inspection reject loader environment, synthetic driver file access and loading.',
                'Actual remote startup, inference, teardown and launch authorization remain coordinator-owned.'])
        # Execute the exact local review predicate from validate(), without running
        # provider code or the separate full launch authorization predicate.
        tree = ast.parse((OUT.parent / 'launch_control.py').read_text())
        validate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate')
        predicate = next(n for n in validate.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                         and isinstance(n.value.func, ast.Name) and n.value.func.id == 'matches'
                         and ast.unparse(n.value.args[0]) == 'load(LOCAL_REVIEW)')
        scope = dict(vars(gate))
        scope['load'] = lambda path: decision if path == gate.LOCAL_REVIEW else gate.load(path)
        exec(compile(ast.Module(body=[predicate], type_ignores=[]), 'exact-local-review-predicate', 'exec'), scope)
        evidence['exact_local_review_predicate'] = 'passed (in-memory decision; no launch gate bypass or provider call)'
        evidence['launch_gate_sha256'] = sha(OUT.parent / 'launch_control.py')
        (OUT / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
        decision['evidence_sha256'] = sha(OUT / 'evidence.json')
    except Exception as error:
        decision.update(status='HOLD', reason=repr(error), reproducer=str(Path(__file__).relative_to(ROOT)))
    (OUT / 'DECISION.json').write_text(json.dumps(decision, indent=2) + '\n')
    print(json.dumps(decision, indent=2))
    raise SystemExit(0 if decision['status'] == 'APPROVED_LOCAL_PACKAGE' else 1)
