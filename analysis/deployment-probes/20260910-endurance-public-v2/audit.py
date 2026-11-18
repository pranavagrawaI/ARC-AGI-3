"""Read-only post-run checks and descriptive timing; assertions are validity gates."""
import hashlib
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
RAW = HERE/'raw'
Q = RAW/'qualification'
load = lambda p: json.loads(p.read_text())
rows = lambda p: [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return {'n': 0}
    index = (len(ordered)-1)*.9
    lo = int(index)
    p90 = ordered[lo] + (ordered[min(lo+1, len(ordered)-1)]-ordered[lo])*(index-lo)
    return dict(n=len(values), total=sum(values), mean=statistics.mean(values),
                median=statistics.median(values), p90_linear=p90, maximum=max(values))


def audit():
    build = load(HERE/'source/BUILD.json')
    assert all(digest(HERE/'source'/n) == h for n, h in build['files'].items())
    manifest = load(RAW/'nooa-join/CHASSIS.json')
    assert manifest['contract_sha256'] == build['contract_sha256']
    assert all(digest(RAW/'nooa-join'/n) == h for n, h in manifest['files'].items())
    for name, target in [('adapter.py', 'agents/templates/my_agent.py'),
                         ('shared_local_solver.py', 'runtime/local_solver.py')]:
        assert digest(RAW/'ARC-AGI-3-Agents'/target) == manifest['sources'][name]
    source = load(HERE/'source/submission.ipynb')
    remote = load(HERE/'remote-source/arc3-first-submission-20260910.ipynb')
    assert [''.join(c['source']) for c in source['cells']] == [''.join(c['source']) for c in remote['cells']]
    result = load(Q/'result.json')
    assert result['status'] == 'passed' and result['model_verification'] == 'matched'
    lifecycle = result['lifecycle']
    assert lifecycle['status'] == 'passed' and lifecycle['containment_passed']
    assert lifecycle['root_returncode'] == 0 and lifecycle['term_only']
    progress = rows(Q/'game-progress.jsonl')
    allocations = rows(Q/'budget-allocations.jsonl')
    assert [r['game_id'].split('-')[0] for r in progress] == ['ls20', 's5i5', 'cd82']
    assert [r['game_id'] for r in allocations] == [r['game_id'] for r in progress]
    assert all(a['admission_seconds'] == 600 for a in allocations)
    assert all(0 <= r['solver_actions'] <= 200 for r in progress)
    launcher = load(Q/'launcher-budget.json')
    assert 60 < launcher['effective_seconds'] <= 2700
    assert launcher['at_monotonic'] + launcher['effective_seconds'] <= float(launcher['notebook_work_deadline'])-29
    sessions = [p for p in (Q/'solver-workspaces').iterdir() if p.is_dir()]
    assert len(sessions) == 3
    observations = []
    all_calls = []
    for game, allocation in zip(progress, allocations):
        matches = [s for s in sessions if s.name.startswith(game['game_id']+'-')]
        assert len(matches) == 1
        session = matches[0]
        ipc = session/'ipc'
        events = rows(ipc/'request_lifecycle.jsonl')
        calls = rows(ipc/'model_calls.jsonl')
        timings = rows(ipc/'execution_timing.jsonl') if (ipc/'execution_timing.jsonl').exists() else []
        assert len(timings) == game['solver_actions']
        assert len(events) == 2*len(calls)
        assert [e['event'] for e in events] == ['admitted', 'completed']*len(calls)
        assert len({c['request_id'] for c in calls}) == len(calls)
        assert load(session/'worker-metadata.json')['contract_sha256'] == build['contract_sha256']
        assert not (ipc/'ABORT').exists() and not (ipc/'request_failure.json').exists()
        start = allocation['admission_deadline']-600
        end = start+game['game_elapsed_seconds']
        assert game['stop_reason'] in ('model_admission_budget', 'wall_budget', 'step_budget', 'win')
        if game['stop_reason'] == 'model_admission_budget':
            stop = load(ipc/'budget_exhausted.json')
            assert stop['active_requests'] == 0 and stop['reason'] == 'model_admission_budget'
            assert allocation['admission_deadline'] <= stop['at_monotonic'] <= end
        if game['stop_reason'] == 'wall_budget':
            assert end >= allocation['admission_deadline']
        if game['stop_reason'] == 'step_budget':
            assert game['solver_actions'] == 200
        if game['stop_reason'] == 'win':
            assert game['state'] == 'WIN'
        for begin, finish, call in zip(events[::2], events[1::2], calls):
            assert begin['request_id'] == finish['request_id'] == call['request_id']
            assert start <= begin['at_monotonic'] < allocation['admission_deadline']
            assert begin['at_monotonic'] <= finish['at_monotonic'] <= end
            assert begin['worker_peak_rss_kib'] > 0
            assert call['completion_tokens'] <= 16384
            assert not call.get('summarizer') and not call.get('reasoning_present') and not call.get('reasoning_tokens')
        windows = []
        for lower, upper in [(0, 300), (300, 600), (600, float('inf'))]:
            selected = [c for e, c in zip(events[::2], calls) if lower <= e['at_monotonic']-start < upper]
            actions = [t for t in timings if lower <= t['observed_at_monotonic']-start < upper]
            windows.append(dict(start_seconds=lower, end_seconds=upper if upper != float('inf') else None,
                admitted_calls=len(selected), host_received_actions=len(actions),
                call_wall_seconds=distribution([c['wall_seconds'] for c in selected]),
                prompt_tokens=distribution([c['prompt_tokens'] for c in selected])))
        observations.append(dict(**game, session=session.name, completed_calls=len(calls),
            call_wall_seconds=distribution([c['wall_seconds'] for c in calls]),
            prompt_tokens=distribution([c['prompt_tokens'] for c in calls]),
            windows=windows, peak_worker_rss_kib=max([e['worker_peak_rss_kib'] for e in events[::2]], default=None),
            last_call_completion_after_admission_seconds=max([e['at_monotonic']-allocation['admission_deadline'] for e in events[1::2]], default=None)))
        all_calls.extend(calls)
    facts = result['request_count_facts']
    assert len(all_calls) == facts['observed_successful_launcher_solver_requests']
    assert facts['observed_total_chat_completion_accesses'] == facts['observed_total_successful_chat_completion_accesses'] == len(all_calls)+1
    assert facts['whole_model_request_count_verified']
    log = (Q/'server.log').read_text()
    markers = re.findall(r'out of memory|OutOfMemoryError|\bOOM\b|Request state was deleted|Traceback \(most recent call last\)|HTTP/1\.1" [45]\d\d', log, re.I)
    assert not markers, markers
    assert 'Remaining number of requests 0' in log
    terminal = load(RAW/'arc3-first-submission-20260910.log')
    assert max(r['time'] for r in terminal) < 3600
    return dict(status='passed', scope='public operational endurance only; not comparative efficacy',
                contract_sha256=build['contract_sha256'], games=observations,
                total_model_calls=len(all_calls), call_wall_seconds=distribution([c['wall_seconds'] for c in all_calls]),
                outer_elapsed_seconds=result['elapsed_s'], notebook_last_log_seconds=max(r['time'] for r in terminal))


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
