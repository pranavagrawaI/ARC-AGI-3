"""Bounded, paired real-environment continuations; no generated code executes."""
import ast
import copy
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'analysis/deployment-probes/20260912-objection-closure-public-v1/raw/qualification/solver-workspaces/ls20-9607627b-bsa1cz1e'
MODEL = 'qwen/qwen3.8-27b'
ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'
ACTIONS = {'UP': 'ACTION1', 'DOWN': 'ACTION2', 'LEFT': 'ACTION3', 'RIGHT': 'ACTION4', 'RESET': 'RESET'}
CHECKPOINT = '''DECISION CHECKPOINT: The retained Actor proposed stopping. Before accepting that conclusion, identify one consequential assumption not established by its observations. Specify a short probe and different observable outcomes that would retain versus revise the assumption. Execute the probe now, or identify a concrete budget/safety/input blocker. A failed search inside your model does not establish impossibility in the environment. The host does not know the missing mechanic. Do not merely acknowledge this review. Explain the assumption and outcome distinction in your rationale; update it after observing the result.'''
SYSTEM = '''You are continuing an interactive grid game from an archived premature stop. Discover its unknown goal/mechanics and try to clear a level. The retained notes are fallible Actor interpretations, not host facts. This is a fresh small continuation with six NEW actions and four model calls available, irrespective of any budget claim in old notes. No old Python variables survive. Both observation and action coordinates are zero-based. Grid encoding is lossless: rows_rle contains [repeat_count, row_runs], and row_runs contains [cell_count, integer_color]. Changes are [row,column,old,new]. You may act in batches of one or two, or explicitly stop. Use the decide tool. State your expected observation in prediction, and your reason in rationale. On subsequent calls respond to actual observations, not invented results. RESET also consumes an action. There are no tools beyond decide. These are observations of a public game, not a source-code exercise.'''
TOOLS = [{'type': 'function', 'function': {'name': 'decide', 'description': 'Execute zero to two valid game actions. Empty actions means voluntary stop.', 'parameters': {'type': 'object', 'properties': {'actions': {'type': 'array', 'items': {'type': 'string', 'enum': list(ACTIONS)}, 'maxItems': 2}, 'rationale': {'type': 'string'}, 'prediction': {'type': 'string'}, 'stop_reason': {'type': 'string'}}, 'required': ['actions', 'rationale', 'prediction', 'stop_reason'], 'additionalProperties': False}}}]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, sort_keys=True) + '\n')
        f.flush()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changes(before, after):
    return [[r, c, a, b] for r, (ar, br) in enumerate(zip(before, after, strict=True))
            for c, (a, b) in enumerate(zip(ar, br, strict=True)) if a != b]


def encode(grid):
    rows = [[[len(list(items)), color] for color, items in itertools.groupby(row)] for row in grid]
    return {'rows_rle': [[len(list(items)), row] for row, items in itertools.groupby(rows)]}


def evidence():
    transitions = [json.loads(l) for l in (SOURCE / 'ipc/transitions.jsonl').read_text().splitlines()]
    assert len(transitions) == 7
    reflection = (SOURCE / 'team_nemo/shared/knowledge/final_reflection.md').read_text()
    packet = {'initial_grid': encode(transitions[0]['before']['grid']), 'history': [
        {'step': x['step'], 'action': x['action'], 'changes': changes(x['before']['grid'], x['after']['grid'])}
        for x in transitions], 'retained_actor_reflection': reflection}
    return transitions, packet


def environment(transitions):
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction
    env = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT / 'environment_files')).make('ls20-9607627b', seed=0)
    assert env is not None
    raw = env.reset()
    assert raw is not None
    for row in transitions:
        assert raw.frame[-1].tolist() == row['before']['grid'], ('before', row['step'])
        raw = env.step(GameAction[ACTIONS[row['action']]])
        assert raw is not None
        assert raw.frame[-1].tolist() == row['after']['grid'], ('after', row['step'])
        assert raw.levels_completed == row['after']['levels_completed']
        assert raw.state.name == row['after']['state']
    return env, raw


def body_for(messages, seed):
    return {'model': MODEL, 'messages': copy.deepcopy(messages), 'tools': TOOLS,
            'tool_choice': {'type': 'function', 'function': {'name': 'decide'}},
            'max_tokens': 1024, 'temperature': 0.7, 'seed': seed,
            'reasoning': {'enabled': False},
            'provider': {'allow_fallbacks': False, 'require_parameters': True,
                         'max_price': {'prompt': 1, 'completion': 10}}}


def receipt(payload, prompt_bound, reservation):
    if payload.get('error') or payload.get('model') != MODEL:
        raise ValueError('provider error or unexpected model')
    usage = payload.get('usage')
    if not isinstance(usage, dict):
        raise ValueError('missing usage')
    for k, cap in [('prompt_tokens', prompt_bound), ('completion_tokens', 1024)]:
        value = usage.get(k)
        if type(value) is not int or not 0 <= value <= cap:
            raise ValueError('invalid or over-reservation usage')
    cost = usage.get('cost')
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not math.isfinite(cost) or not 0 <= cost <= reservation:
        raise ValueError('missing/invalid cost')
    choices = payload.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError('missing choices')
    message = choices[0]['message']
    if message.get('reasoning') or message.get('reasoning_details') or (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0):
        raise ValueError('unexpected reasoning')
    return cost


class Transport:
    def __init__(self, key, out=HERE, opener=urllib.request.urlopen):
        self.key, self.out, self.opener = key, out, opener
        self.started, self.spent, self.index = time.monotonic(), 0., 0

    def call(self, body, label):
        encoded = json.dumps(body, separators=(',', ':')).encode()
        prompt_bound = len(encoded) + 2048
        reservation = prompt_bound / 1e6 + 1024 * 10 / 1e6
        if len(encoded) > 24000 or self.index >= 24 or self.spent + reservation > 1 or time.monotonic() - self.started >= 900:
            raise RuntimeError('admission cap')
        index = self.index
        self.index += 1
        path = self.out / 'raw' / f'{index:02d}'
        dump(path.with_name(path.name + '-request.json'), body)
        event = {'index': index, 'label': label, 'reservation': reservation, 'prompt_bound': prompt_bound,
                 'request_sha256': sha(path.with_name(path.name + '-request.json'))}
        append(self.out / 'attempts.jsonl', dict(event, status='started'))
        called = time.monotonic()
        try:
            request = urllib.request.Request(ENDPOINT, data=encoded, headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
            with self.opener(request, timeout=min(60, 900 - (called - self.started))) as response:
                raw = response.read()
            path.with_name(path.name + '-response.body').write_bytes(raw)
            payload = json.loads(raw)
            dump(path.with_name(path.name + '-response.json'), payload)
            cost = receipt(payload, prompt_bound, reservation)
            self.spent += cost
        except Exception as error:
            if isinstance(error, urllib.error.HTTPError):
                path.with_name(path.name + '-error.body').write_bytes(error.read())
            append(self.out / 'attempts.jsonl', dict(event, status='failed', latency_seconds=time.monotonic()-called,
                   error_type=type(error).__name__, http_status=getattr(error, 'code', None), actual_cost=None))
            raise
        append(self.out / 'attempts.jsonl', dict(event, status='completed', latency_seconds=time.monotonic()-called,
               actual_cost=cost, usage=payload['usage']))
        return payload


def decision(payload):
    msg = payload['choices'][0]['message']
    calls = msg.get('tool_calls') or []
    if len(calls) != 1 or calls[0]['function']['name'] != 'decide':
        raise ValueError('exactly one decide call required')
    value = json.loads(calls[0]['function']['arguments'])
    if set(value) != {'actions', 'rationale', 'prediction', 'stop_reason'}:
        raise ValueError('decision fields')
    if not isinstance(value['actions'], list) or len(value['actions']) > 2 or any(type(x) is not str or x not in ACTIONS for x in value['actions']):
        raise ValueError('invalid actions')
    if any(type(value[k]) is not str for k in ('rationale', 'prediction', 'stop_reason')):
        raise ValueError('invalid prose fields')
    return msg, calls[0]['id'], value


def arm(transport, transitions, packet, name, seed):
    from arcengine import GameAction
    env, raw = environment(transitions)
    start_level = raw.levels_completed
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': json.dumps(packet)}]
    if name == 'checkpoint':
        messages.append({'role': 'user', 'content': CHECKPOINT})
    result = {'arm': name, 'seed': seed, 'actions': 0, 'calls': 0, 'level_gain': 0, 'stop': 'call_cap'}
    for turn in range(4):
        messages.append({'role': 'user', 'content': json.dumps({'new_actions_remaining': 6-result['actions'], 'calls_remaining': 4-turn})})
        payload = transport.call(body_for(messages, seed), f'{seed}-{name}-{turn}')
        result['calls'] += 1
        try:
            msg, call_id, value = decision(payload)
        except (ValueError, KeyError, TypeError) as error:
            result['stop'] = 'invalid_decision'
            break
        append(HERE / 'trajectories.jsonl', {'seed': seed, 'arm': name, 'turn': turn, 'event': 'decision', 'decision': value})
        if not value['actions']:
            result['stop'] = 'voluntary_stop'
            break
        observed = []
        for action in value['actions'][:6-result['actions']]:
            before = raw.frame[-1].tolist()
            previous_level = raw.levels_completed
            raw = env.step(GameAction[ACTIONS[action]])
            assert raw is not None
            result['actions'] += 1
            result['level_gain'] = raw.levels_completed - start_level
            row = {'action': action, 'before': before, 'after': raw.frame[-1].tolist(), 'levels_completed': raw.levels_completed, 'state': raw.state.name}
            append(HERE / 'trajectories.jsonl', {'seed': seed, 'arm': name, 'turn': turn, 'event': 'action', **row})
            observed.append({'action': action, 'changes': changes(before, row['after']), 'levels_completed': raw.levels_completed, 'state': raw.state.name})
            if raw.levels_completed != previous_level or raw.state.name in ('WIN', 'GAME_OVER'):
                break
        messages.append(msg)
        messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps(observed)})
        if raw.state.name in ('WIN', 'GAME_OVER'):
            result['stop'] = raw.state.name
            break
        if result['actions'] >= 6:
            result['stop'] = 'action_cap'
            break
    return result


def audit(out=HERE):
    ledger = out/'attempts.jsonl'
    rows = [json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []
    starts = {r['index']: r for r in rows if r['status']=='started'}
    ends = {r['index']: r for r in rows if r['status']!='started'}
    checks = {'paired_lifecycle': set(starts)==set(ends) and len(rows)==2*len(starts),
              'attempt_cap': len(starts)<=24, 'positive_latency': all(r['latency_seconds']>=0 for r in ends.values()),
              'no_requests_after_failure': all(i==max(starts) for i,r in ends.items() if r['status']=='failed')}
    completed = [r for r in ends.values() if r['status']=='completed']
    for i,r in starts.items():
        p = out/'raw'/f'{i:02d}-request.json'
        body = json.loads(p.read_text())
        checks[f'request_{i}_hash'] = sha(p)==r['request_sha256']
        checks[f'request_{i}_routing'] = body['model']==MODEL and body['reasoning']=={'enabled':False} and body['max_tokens']==1024 and body['provider']['allow_fallbacks'] is False
        if ends[i]['status']=='completed':
            payload = json.loads((out/'raw'/f'{i:02d}-response.json').read_text())
            checks[f'receipt_{i}'] = receipt(payload, r['prompt_bound'], r['reservation'])==ends[i]['actual_cost']
    cost = sum(r['actual_cost'] for r in completed)
    checks['spend_bound'] = cost + sum(r['reservation'] for r in ends.values() if r['status']=='failed') <= 1
    failed = any(r['status']=='failed' for r in ends.values())
    run_result = json.loads((out/'RESULT.json').read_text()) if (out/'RESULT.json').exists() else {}
    complete = run_result.get('classification')=='COMPLETE_PILOT' and len(run_result.get('arms',[]))==6
    result = {'checks':checks, 'passed':all(checks.values()), 'completed':len(completed), 'attempts':len(starts), 'receipted_cost':cost, 'total_cost':None if failed else cost, 'performance_valid':complete and not failed and all(checks.values())}
    dump(out/'AUDIT.json', result)
    return result


def subjects():
    return [HERE/'pilot.py', HERE/'test_pilot.py', HERE/'EXPERIMENT-CARD.md',
            SOURCE/'ipc/transitions.jsonl', SOURCE/'team_nemo/shared/knowledge/final_reflection.md',
            ROOT/'environment_files/ls20/9607627b/ls20.py', ROOT/'uv.lock']


def prepare():
    transitions, packet = evidence()
    a, first = environment(transitions)
    b, second = environment(transitions)
    assert first.frame[-1].tolist() == second.frame[-1].tolist()
    assert len(json.dumps(body_for([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': json.dumps(packet)}, {'role': 'user', 'content': CHECKPOINT}], 41), separators=(',', ':')).encode()) < 24000
    dump(HERE/'PRELAUNCH.json', {'passed': True, 'prefix_length': len(transitions), 'matching_replays': 2, 'source_sha256': {str(p): sha(p) for p in subjects()}})
    print('Preflight: two complete seven-action replays match; request fits cap.')


def run():
    frozen = json.loads((HERE/'PRELAUNCH.json').read_text())
    assert frozen['passed'] and all(sha(Path(p)) == h for p,h in frozen['source_sha256'].items())
    if (HERE/'STARTED.json').exists() or (HERE/'raw').exists():
        raise RuntimeError('already started; no retry')
    key = None
    with Path('/home/pranav/projects/arc-agi-3/.env').open() as stream:
        for line in stream:
            name, separator, value = line.strip().removeprefix('export ').partition('=')
            if separator and name.strip() == 'OPENROUTER_API_KEY':
                if key is not None:
                    raise RuntimeError('duplicate key')
                value = value.strip()
                key = ast.literal_eval(value) if value.startswith(('"', "\x27")) else value
    if not isinstance(key, str) or not key:
        raise RuntimeError('authorized key unavailable')
    dump(HERE/'STARTED.json', {'time_unix': time.time(), 'prelaunch_sha256': sha(HERE/'PRELAUNCH.json'), 'source_sha256': frozen['source_sha256']})
    transport = Transport(key)
    result = {'classification': 'COMPLETE_PILOT', 'arms': []}
    try:
        request = urllib.request.Request('https://openrouter.ai/api/v1/credits', headers={'Authorization':'Bearer '+key})
        with urllib.request.urlopen(request, timeout=30) as response:
            credit = json.load(response)['data']
        available = credit['total_credits'] - credit['total_usage']
        dump(HERE/'QUOTA.json', {'available_at_least_authorized_cap':available>=1})
        if available < 1:
            raise RuntimeError('insufficient provider credit for authorized cap')
        transitions, packet = evidence()
        for seed, order in [(41, ('baseline', 'checkpoint')), (42, ('checkpoint', 'baseline')), (43, ('baseline', 'checkpoint'))]:
            for name in order:
                result['arms'].append(arm(transport, transitions, packet, name, seed))
                dump(HERE/'PROGRESS.json', result)
    except Exception as error:
        result.update(classification='INTEGRATION_INVALID', error_type=type(error).__name__, error=str(error))
    result.update(request_attempts=transport.index, receipted_cost=transport.spent,
                  total_cost=transport.spent if result['classification']=='COMPLETE_PILOT' else None,
                  wall_seconds=time.monotonic()-transport.started)
    dump(HERE/'RESULT.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import sys
    {'prepare': prepare, 'run': run, 'audit': lambda: print(json.dumps(audit(),indent=2))}[sys.argv[1]]()
