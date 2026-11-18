"""Offline census and readable excerpts from three preserved sb26 lives."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBES = HERE.parent
SOURCES = {
    'duck_failed': ('20260923-v12-sb26-duck-batch-retry-v2', 'v12-sb26-duck-batch', 'sb26-c7_k1dh4'),
    'duck_success': ('20260923-v12-sb26-duck-batch-retry-v2', 'v12-sb26-duck-batch', 'sb26-wov3ulk5'),
    'critic_off_success': ('20260921-v12-sb26-critic-off', 'v12-sb26-critic-off', 'sb26-ef2qddnf'),
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def slots(grid):
    return [grid[29][col] for col in (22, 28, 34, 40)]


def totals(calls):
    return dict(calls=len(calls), model_wall_s=sum(x['wall_seconds'] for x in calls),
                prompt_tokens=sum(x.get('prompt_tokens') or 0 for x in calls),
                completion_tokens=sum(x.get('completion_tokens') or 0 for x in calls))


def main():
    summary = {}
    ledger = ['# Preserved model code and action ledger', '',
              'Call numbers and transition numbers are one-based JSONL line numbers. '
              'A call at step N occurs after N executed actions. Code comments are model claims, not ground truth.', '']
    initial_grids = []
    for label, (run, arm, session) in SOURCES.items():
        root = PROBES / run / 'kernel-output' / arm / 'nooa-join-public3' / 'solver-workspaces' / session
        ipc = root / 'ipc'
        calls = read_rows(ipc / 'model_calls.jsonl')
        transitions = read_rows(ipc / 'transitions.jsonl')
        batches = read_rows(ipc / 'actions.jsonl')
        initial_grids.append(transitions[0]['before']['grid'])
        colors = {9, 14, 11, 15}
        placement = next(t['step'] for t in transitions
                         if any(c in colors for c in slots(t['after']['grid'])))
        filled = next((t['step'] for t in transitions
                       if all(c in colors for c in slots(t['after']['grid']))), None)
        l1 = next((t['step'] for t in transitions if t['after']['levels_completed'] >= 1), None)
        phases = [('through_first_placement', 0, placement)]
        if filled is not None:
            phases += [('after_placement_through_all_filled', placement, filled),
                       ('all_filled_through_L1', filled, l1)]
        else:
            phases += [('after_placement_through_stop', placement, len(transitions))]
        metrics = {name: totals([c for c in calls if lo <= c['step'] < hi])
                   for name, lo, hi in phases}
        by_step = {str(step): totals([c for c in calls if c['step'] == step])
                   for step in sorted({c['step'] for c in calls})}
        files = {str(p.relative_to(PROBES)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (root / 'launch.json', ipc / 'model_calls.jsonl',
                           ipc / 'actions.jsonl', ipc / 'transitions.jsonl')}
        summary[label] = dict(session=session, first_placement_step=placement,
                              all_filled_step=filled, first_L1_step=l1,
                              all_calls=totals(calls), through_first_16=totals([c for c in calls if c['step'] < 16]),
                              phases=metrics, by_step=by_step, source_sha256=files)
        ledger += [f'## {label}', '', f'Source session: `{root}`', '',
                   '| Step | Turn | Action | Changed pixels | Slot colors after | Levels |',
                   '|---:|---:|---|---:|---|---:|']
        for t in transitions:
            changed = sum(a != b for ar, br in zip(t['before']['grid'], t['after']['grid'])
                          for a, b in zip(ar, br))
            ledger.append(f"| {t['step']} | {t['turn']} | {t['action']} | {changed} | {slots(t['after']['grid'])} | {t['after']['levels_completed']} |")
        ledger += ['', '### Submitted rationales', '']
        for b in batches:
            ledger += [f"- After step {b['step']}: `{b['actions']}` — {b.get('rationale', '')}"]
        ledger += ['', '### Model calls', '']
        for number, c in enumerate(calls, 1):
            if label == 'duck_failed':
                include = c['step'] <= 16 or c['step'] >= 38
            else:
                include = c['step'] < l1
            if not include:
                continue
            ledger += [f"#### {label} call {number}, after step {c['step']}, {c['wall_seconds']:.3f} s", '']
            if c.get('response_content'):
                ledger += [c['response_content'], '']
            for tool in c.get('tool_calls', []):
                args = tool['arguments']
                args = json.loads(args) if isinstance(args, str) else args
                ledger += [f"Tool: `{tool['name']}`", '', '```python',
                           args.get('code', json.dumps(args)), '```', '']
    summary['all_three_initial_grids_equal'] = all(g == initial_grids[0] for g in initial_grids)
    assert summary['all_three_initial_grids_equal']
    (HERE / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    (HERE / 'CALL-LEDGER.md').write_text('\n'.join(ledger) + '\n')
    print(json.dumps({k: {n: v[n] for n in ('first_placement_step', 'all_filled_step', 'first_L1_step', 'phases')}
                      for k, v in summary.items() if isinstance(v, dict)}, indent=2))


if __name__ == '__main__':
    main()
