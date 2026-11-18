"""Audit retained evidence with corrected pin validation; never rewrite raw status."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / 'raw'
RUNTIME = RAW / 'nooa-join-public3'
SESSION = RUNTIME / 'solver-workspaces/ls20-8md_11v1'
ROOT = HERE.parents[3]
CANDIDATE = ROOT / 'deployment/kaggle/q6-summary-boundary-20260906'
os.environ['TYCHO_PUBLIC_GAME_IDS'] = 'ls20'
os.environ['NOOA_JOIN_ROOT'] = str(RAW / 'nooa-join')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


manifest = read(RAW / 'nooa-join/CHASSIS.json')
for name, digest in manifest['files'].items():
    assert hashlib.sha256((RAW / 'nooa-join' / name).read_bytes()).hexdigest() == digest
assert hashlib.sha256((CANDIDATE / 'exact_request_outer_runtime.py').read_bytes()).hexdigest() == manifest['sources']['exact_request_outer_runtime.py']
old = load(RAW / 'nooa-join/nooa_public_probe.py', 'old_probe')
try:
    old._summarize_session(SESSION, 'ls20')
except RuntimeError as exc:
    assert 'chassis pin differs' in str(exc)
else:
    raise AssertionError('Original failure did not reproduce')
fixed = load(HERE / 'nooa_public_probe.py', 'fixed_probe')
with tempfile.TemporaryDirectory(prefix='q6-pin-replay-') as td:
    root = Path(td)
    session = root / SESSION.name
    session.mkdir()
    for name in ['worker-metadata.json', 'launch.json']:
        shutil.copy2(SESSION / name, session / name)
    (session / 'ipc').symlink_to(SESSION / 'ipc', target_is_directory=True)
    # Empty directories are not retained in Kaggle output downloads.
    (session / 'workspace/ls20').mkdir(parents=True)
    sessions = fixed.summarize_nooa_sessions(root)

launcher = read(RUNTIME / 'official-launcher-result.json')
assert launcher['launcher_returncode'] == 0 and launcher['returncode'] == 74
assert not launcher['timed_out'] and not launcher['residual_group_term_required']
assert launcher['public_progress_error'] is None
runtime = load(CANDIDATE / 'exact_request_outer_runtime.py', 'outer_runtime')
reconciliation = runtime.reconcile_launcher_request_count(
    {**launcher, 'solver_sessions': sessions},
    read(RUNTIME / 'warmup-facts.json'),
    runtime.server_request_facts((RUNTIME / 'server.log').read_text()),
)
stop = read(SESSION / 'stop.json')
assert stop['reason'] == 'competition_agent_cleanup' and stop['steps'] == 100
assert not (SESSION / 'ipc/ABORT').exists()
integrity = rows(SESSION / 'ipc/integrity.jsonl')
assert not any(r['event'] == 'INTEGRITY_FAILED' for r in integrity)
calls = rows(SESSION / 'ipc/model_calls.jsonl')
assert all(r.get('reasoning_tokens', 0) == 0 for r in calls)
summaries = [r for r in calls if r.get('summarizer')]
report = {
    'status': 'RETAINED_EVIDENCE_CHECKS_PASS',
    'original_run_status': 'ERROR; unchanged',
    'original_launcher_returncode': launcher['launcher_returncode'],
    'original_probe_returncode': launcher['returncode'],
    'cause': 'Hardcoded q6-stall-authority rejects manifest identity q6-summary-boundary-20260906',
    'source_manifest_verified': manifest['contract_sha256'],
    'session_validator': 'PASS with corrected manifest-based pin check',
    'request_reconciliation': reconciliation,
    'completed_actions': stop['steps'],
    'summary_calls': len(summaries),
    'summary_wall_seconds': sum(r['wall_seconds'] for r in summaries),
    'scope': 'Post-hoc session, request, stop, and integrity checks; not a new kernel run or a rewritten original qualification.',
}
(HERE / 'REPLAY.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
