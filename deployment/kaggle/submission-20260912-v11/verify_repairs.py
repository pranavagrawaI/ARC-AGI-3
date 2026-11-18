"""Reproduce local repair qualification. No remote model, upload, or GPU launch."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNTIME = HERE.parent/'submission-runtime-20260912-v8'
EVIDENCE = ROOT/'analysis/deployment-probes/20260912-epistemic-questioner-local-v1'
BASELINE_EVIDENCE = ROOT/'analysis/deployment-probes/20260910-endurance-bugfix'


def main():
    EVIDENCE.mkdir(exist_ok=True)
    baseline = json.loads((BASELINE_EVIDENCE/'BASELINE.json').read_text())
    original = ROOT/baseline['runtime']
    manifest = json.loads((original/'CHASSIS.json').read_text())
    assert manifest['contract_sha256'] == baseline['contract_sha256']
    for name, digest in {**manifest['files'], **manifest['sources']}.items():
        assert hashlib.sha256((original/name).read_bytes()).hexdigest() == digest, name
    assert hashlib.sha256((ROOT/baseline['notebook']/'submission.ipynb').read_bytes()).hexdigest() == baseline['notebook_sha256']
    commands = [
        ('freeze-check', [str(RUNTIME/'freeze_candidate.py'), '--check']),
        ('unit', ['-m', 'pytest', *[str(RUNTIME/name) for name in (
            'test_baseline_repairs.py', 'test_prediction_feedback.py', 'test_latent_planner.py',
            'test_epistemic_critic.py',
            'test_evidence_interface.py', 'test_typed_experiments.py',
            'test_decision_packet.py', 'test_step_check.py', 'test_contingent_controller.py')],
            '-q']),
        ('entrypoints', ['-m', 'pytest', *[str(HERE/name) for name in (
            'test_submission.py', 'test_budget_accounting.py', 'test_deadlines.py')], '-q']),
        ('contingent-worker', ['-m', 'pytest', str(RUNTIME/'test_contingent_worker.py'), '-q']),
    ]
    results = []
    for name, argv in commands:
        with (EVIDENCE/f'verify-{name}.txt').open('w') as output:
            result = subprocess.run([sys.executable, *argv], cwd=ROOT,
                                    stdout=output, stderr=subprocess.STDOUT)
        results.append({'check': name, 'returncode': result.returncode})
        print(name, result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {EVIDENCE}/verify-{name}.txt')
    (EVIDENCE/'VERIFIED.json').write_text(json.dumps({
        'status': 'passed', 'baseline_unchanged': True, 'checks': results,
        'runtime_contract': json.loads((RUNTIME/'CHASSIS.json').read_text())['contract_sha256'],
        'scope': 'local deterministic and loopback HTTP checks; no live model qualification',
    }, indent=2)+'\n')
    print('All repair checks passed', flush=True)


if __name__ == '__main__':
    main()
