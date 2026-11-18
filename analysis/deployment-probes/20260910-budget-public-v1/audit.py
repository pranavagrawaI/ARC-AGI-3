"""Read-only verification of the downloaded, pinned public run."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'
Q = RAW / 'qualification'
load = lambda p: json.loads(p.read_text())
rows = lambda p: [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
build = load(HERE/'source/BUILD.json')
assert all(digest(HERE/'source'/n) == h for n, h in build['files'].items())
manifest = load(RAW/'nooa-join/CHASSIS.json')
assert manifest['contract_sha256'] == build['contract_sha256']
assert all(digest(RAW/'nooa-join'/n) == h for n, h in manifest['files'].items())
for name, path in [('adapter.py', 'agents/templates/my_agent.py'),
                   ('shared_local_solver.py', 'runtime/local_solver.py')]:
    assert digest(RAW/'ARC-AGI-3-Agents'/path) == manifest['sources'][name]
source = load(HERE/'source/submission.ipynb')
remote = load(HERE/'remote-source/arc3-first-submission-20260910.ipynb')
assert [''.join(c['source']) for c in source['cells']] == [''.join(c['source']) for c in remote['cells']]
result = load(Q/'result.json')
assert result['status'] == 'passed' and result['model_verification'] == 'matched'
assert result['lifecycle']['status'] == 'passed' and result['lifecycle']['containment_passed']
assert result['lifecycle']['root_returncode'] == 0 and result['lifecycle']['term_only']
progress = rows(Q/'game-progress.jsonl')
allocations = rows(Q/'budget-allocations.jsonl')
assert [r['game_id'].split('-')[0] for r in progress] == ['ls20', 's5i5', 'cd82']
assert all(0 <= r['solver_actions'] <= 10 and r['stop_reason'] == 'model_admission_budget' for r in progress)
assert len(allocations) == 3 and all(a['admission_seconds'] == 75 for a in allocations)
sessions = sorted((Q/'solver-workspaces').iterdir())
assert len(sessions) == 3
counts = {}
for session in sessions:
    events = rows(session/'ipc/request_lifecycle.jsonl')
    calls = rows(session/'ipc/model_calls.jsonl')
    stop = load(session/'ipc/budget_exhausted.json')
    assert stop['reason'] == 'model_admission_budget' and stop['active_requests'] == 0
    assert stop['at_monotonic'] >= stop['admission_deadline']
    assert len(events) == 2*len(calls)
    assert [r['event'] for r in events] == ['admitted','completed']*len(calls)
    for begin, end in zip(events[::2], events[1::2]):
        assert begin['request_id'] == end['request_id']
        assert begin['at_monotonic'] < begin['admission_deadline']
        assert begin['at_monotonic'] <= end['at_monotonic'] <= stop['at_monotonic']
    assert not any(c.get('summarizer') or c.get('reasoning_present') or c.get('reasoning_tokens', 0) for c in calls)
    assert load(session/'worker-metadata.json')['contract_sha256'] == build['contract_sha256']
    assert not (session/'ipc/ABORT').exists()
    counts[session.name] = len(calls)
facts = result['request_count_facts']
assert sum(counts.values()) == 34 == facts['observed_successful_launcher_solver_requests']
assert facts['observed_total_chat_completion_accesses'] == facts['observed_total_successful_chat_completion_accesses'] == 35
assert facts['whole_model_request_count_verified']
log = (Q/'server.log').read_text()
failures = re.findall(r'out of memory|OutOfMemoryError|\bOOM\b|Request state was deleted|Traceback \(most recent call last\)|HTTP/1\.1" [45]\d\d', log, re.I)
assert not failures, failures
assert 'Remaining number of requests 0' in log
terminal = load(RAW/'arc3-first-submission-20260910.log')
assert max(r['time'] for r in terminal) < 3600
print(json.dumps(dict(status='passed', scope='public operational budget/lifecycle qualification only',
    contract_sha256=build['contract_sha256'], game_progress=progress, completed_model_calls=counts,
    outer_elapsed_seconds=result['elapsed_s'], notebook_last_log_seconds=max(r['time'] for r in terminal),
    server_error_markers=failures), indent=2))
