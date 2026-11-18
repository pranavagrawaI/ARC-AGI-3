"""Offline census of preserved Duck traces; no model or game calls.

Run from any directory: python3 /absolute/path/to/audit.py
Counters describe logged events, not causal effects. Analyzer attempts include yields.
"""
import ast
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
DUCK = Path('/home/pranav/projects/arc-agi-3-reference/duck-qwen38')
ARMS = {
    'K': 'runs/flash-next/out-k2',
    'KF': 'runs/flash-next/out-kf2',
    'F': 'runs/qwen38-fixes/f/live/kaggle-output',
    'F2': 'runs/qwen38-fixes/f2x/live/kaggle-output',
    'Fc': 'runs/qwen38-fixes/fc/live/kaggle-output',
    'Q': 'runs/qwen38-fixes/q/live/kaggle-output',
    'Q3': 'runs/qwen38-fixes/q2x/live/kaggle-output',
    'Sa': 'runs/qwen38-fixes/sa/live/kaggle-output',
    'Sb': 'runs/qwen38-fixes/sb/live/kaggle-output',
    'Sc': 'runs/qwen38-fixes/sc/live/kaggle-output',
}
LABEL = re.compile(r'^\[(SYSTEM PROMPT|USER PROMPT|MODEL RESPONSE META|THINKING|TOOL CALL: python|TOOL RESULT: python|ANALYZER STATUS|ASSISTANT RESPONSE)\]\n', re.M)


def sections(text):
    matches = list(LABEL.finditer(text))
    return [(m[1], text[m.end():matches[i+1].start() if i+1 < len(matches) else len(text)]) for i, m in enumerate(matches)]


def codes(meta):
    if 'raw_tool_calls:\n' not in meta:
        return []
    try:
        calls = json.JSONDecoder().raw_decode(meta.split('raw_tool_calls:\n', 1)[1])[0]
        return [json.loads(c['function']['arguments']).get('code', '') for c in calls if c['function']['name'] == 'python']
    except (ValueError, KeyError, TypeError):
        return []


def main():
    inventory, result, errors = [], {}, []
    for arm, relative in ARMS.items():
        per_game = {}
        for path in sorted((DUCK / relative / 'artifacts').glob('*events.jsonl')):
            data = path.read_bytes()
            inventory.append({'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
            evs = [json.loads(line) for line in data.splitlines()]
            counter = collections.Counter()
            game = path.name[:4]
            digest = []
            for line, ev in enumerate(evs, 1):
                if ev['type'] == 'action':
                    counter['actions'] += 1
                    counter['levels'] += str(ev.get('level_completed')) == 'True'
                    counter['game_overs'] += str(ev.get('game_over')) == 'True'
                    counter['unchanged_actions'] += str(ev.get('board_changed')) == 'False'
                    if str(ev.get('level_completed')) == 'True':
                        digest.append(f"COMPLETION event line {line}, action {ev['action_num']}, level {ev['level']}")
                if ev['type'] != 'analysis':
                    continue
                counter['analyzer_attempts'] += 1
                parts = sections(ev['transcript'])
                tail = ev['transcript'][ev['transcript'].rfind('[ANALYZER STATUS]'):]
                counter['acted_attempts'] += 'step_executed: True' in tail
                counter['yielded_attempts'] += 'turn_time_budget' in tail and 'step_executed: True' not in tail
                latest_codes = []
                for label, value in parts:
                    if label == 'MODEL RESPONSE META':
                        counter['model_responses'] += 1
                        latest_codes = codes(value)
                        for code in latest_codes:
                            counter['python_calls'] += 1
                            counter['python_code_chars'] += len(code)
                            try:
                                tree = ast.parse(code)
                                counter['calls_defining_functions'] += any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
                            except SyntaxError:
                                counter['syntactically_invalid_calls'] += 1
                    if label == 'TOOL RESULT: python':
                        # Restrict to the tool's actual error footer; ordinary printed 'error' is not enough.
                        is_error = value.startswith(('Traceback (most recent call last):', 'Tool timed out', 'Python syntax error:', 'Sandbox process')) or '\nerror:\n' in value
                        if is_error:
                            counter['tool_errors'] += 1
                            timeout = 'Tool timed out after' in value
                            counter['tool_timeouts'] += timeout
                            if arm == 'KF':
                                errors.append({'game': game, 'event_line': line, 'analysis_step': ev.get('analysis_step'), 'action_num': ev.get('action_num'), 'error': value, 'codes': latest_codes})
                                if game == 'ls20' and timeout:
                                    (HERE / 'ls20-timeout-original.py').write_text(latest_codes[-1] + '\n')
                        if arm == 'KF' and 'output truncated' in value.lower():
                            counter['tool_truncation_markers'] += 1
                    if label == 'THINKING' and arm == 'KF':
                        digest.append(f"event line {line}, step {ev.get('analysis_step')}, action {ev.get('action_num')}: {value.strip()}")
            per_game[game] = dict(counter)
            if arm == 'KF':
                (HERE / 'digests').mkdir(exist_ok=True)
                (HERE / 'digests' / f'{game}.txt').write_text('\n\n'.join(digest))
        totals = collections.Counter()
        for counts in per_game.values():
            totals.update(counts)
        result[arm] = {'games': len(per_game), 'totals': dict(totals), 'per_game': per_game}
    (HERE / 'trace-census.json').write_text(json.dumps(result, indent=2) + '\n')
    (HERE / 'trace-source-hashes.json').write_text(json.dumps(inventory, indent=2) + '\n')
    (HERE / 'kf-tool-errors.json').write_text(json.dumps(errors, indent=2) + '\n')
    serving = {}
    for arm in ['K', 'KF']:
        folder = DUCK / ARMS[arm]
        metric_path = folder / 'vllm-metrics-final.prom'
        log_path = folder / 'vllm-openai-server.log'
        metric_text = metric_path.read_text()
        vals = {}
        for name in ['e2e_request_latency_seconds', 'request_queue_time_seconds',
                     'request_prefill_time_seconds', 'request_decode_time_seconds']:
            vals[name] = {suffix: float(re.search(r'^vllm:' + name + '_' + suffix + r'\{[^\n]*\} ([\d.e+-]+)', metric_text, re.M)[1]) for suffix in ['sum', 'count']}
            vals[name]['mean'] = vals[name]['sum'] / vals[name]['count']
        vals['queue_fraction_of_summed_e2e'] = vals['request_queue_time_seconds']['sum'] / vals['e2e_request_latency_seconds']['sum']
        rows = [tuple(map(float, m)) for m in re.findall(r'Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%', log_path.read_text())]
        active = [r for r in rows if r[0]+r[1] >= 20]
        vals['active_ge20_log_samples'] = len(active)
        vals['active_ge20_medians'] = {k: statistics.median(r[i] for r in active) for i, k in enumerate(['running','waiting','kv_percent'])}
        serving[arm] = vals
        assert vals['e2e_request_latency_seconds']['count'] == result[arm]['totals']['model_responses']
        for path in [metric_path, log_path, folder/'vllm-server-identity.json', folder/'score.json']:
            data = path.read_bytes()
            inventory.append({'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    (HERE/'serving-metrics.json').write_text(json.dumps(serving, indent=2)+'\n')
    (HERE / 'trace-source-hashes.json').write_text(json.dumps(inventory, indent=2) + '\n')
    path = DUCK / ARMS['KF'] / 'artifacts/sb26-7fbdac44_p0_events.jsonl'
    evs = [json.loads(line) for line in path.open()]
    actions = [e for e in evs if e['type'] == 'action']
    winner = next(e for e in actions if e['level_completed'])
    snapshots = [e for e in evs if e['type'] in ('initial', 'action')]
    before = snapshots[snapshots.index(winner)-1]
    boundary = {'source': str(path), 'winning_action': {k: winner[k] for k in ['action_num','action_display','level','score','level_completed']},
                'before_level': before['level'], 'after_level': winner['level'],
                'filter_after_level_eq_1_actions': [e['action_num'] for e in actions if e['level']==1],
                'winning_action_excluded': winner['level'] != 1}
    for line, e in enumerate(evs, 1):
        if e['type'] == 'analysis' and 'L1 completed automatically on the final correct placement (no SPACE)' in e['transcript']:
            boundary['false_claim_event_line'] = line
            for label, value in sections(e['transcript']):
                if label == 'MODEL RESPONSE META':
                    for code in codes(value):
                        if 'h1=[t for t in transitions if t.after_frame.level==1]' in code:
                            boundary['actual_query'] = code
    assert boundary['winning_action_excluded'] and boundary['winning_action']['action_display'] == 'SPACE'
    (HERE/'sb26-level-boundary.json').write_text(json.dumps(boundary, indent=2)+'\n')
    for arm, record in result.items():
        print(arm, record['games'], json.dumps(record['totals']))


if __name__ == '__main__':
    main()
