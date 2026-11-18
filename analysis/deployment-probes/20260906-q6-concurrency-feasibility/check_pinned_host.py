"""Exercise the shipped host's admission gate without GPU/model requests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = ROOT / 'deployment/kaggle/q6-summary-boundary-20260906'
sys.path.insert(0, str(ROOT / 'deployment/kaggle'))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


source = CANDIDATE / 'shared_local_solver.py'
manifest = json.loads((CANDIDATE / 'CHASSIS.json').read_text())
digest = hashlib.sha256(source.read_bytes()).hexdigest()
assert digest == manifest['sources']['shared_local_solver.py']
load(source, 'runtime.local_solver')
support = load(ROOT / 'deployment/kaggle/tests/test_local_solver.py', 'gate_test')
with tempfile.TemporaryDirectory(prefix='q6-gate-check-') as td:
    support.test_two_games_share_one_generation_admission_slot(Path(td))
    events = support.read_events(Path(td) / 'shared-events.jsonl')
    sequence = [r['event'] for r in events if r['event'].startswith('generation_')]
result = {
    'status': 'CONFIRMED_SERIALIZATION',
    'pinned_host_sha256': digest,
    'worker_generation_events': sequence,
    'gate_scope': 'LocalSolver.decide: send turn through receive decision',
    'network_model_calls': 0,
    'scope': 'Two concurrent Solver instances with real local worker subprocesses; fake worker computation, no GPU batching or real-game qualification.',
}
(HERE / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
