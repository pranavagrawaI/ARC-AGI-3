"""Reproduce a generated historical checker and audit its actual comparisons.

Run with solvers/tycho_nooa/.venv/bin/python. Uses only local recorded artifacts.
The adapter and case selection are handmade; generated functions are unchanged.
"""
import ast
import hashlib
import json
import pickle
import re
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'analysis/m2-winners-20261001/runs/out-control'
LETTERS = 'WwgGcBMPRbSYOrNp'
FUNCTIONS = {'extract2', 'rc', 'blocked', 'ov', 'adjc', 'd2t', 't2d', 'blkm', 'sim4', 'sim5', 'sim6'}
ACTION_MAP = {12: 'O+', 6: 'O-', 27: 'S+', 21: 'S-', 42: 'Y+', 36: 'Y-', 57: 'N+', 51: 'N-'}


class Record:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.__dict__.update(kwargs)


class ArtifactReader(pickle.Unpickler):
    # Stored taaf wrappers need no runtime behavior; arcengine/numpy restore frames.
    def find_class(self, module, name):
        if module.startswith('taaf.'):
            return Record
        return super().find_class(module, name)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(value):
    return json.loads(value) if isinstance(value, str) else value


def main():
    started = time.perf_counter()
    requests_path = next(SOURCE.glob('s5i5*requests.jsonl'))
    snippets, seen = [], set()
    for line_number, line in enumerate(requests_path.open(), 1):
        request = json.loads(line)
        if line_number > 119:
            break
        for index, message in enumerate(request['messages']):
            for call in message.get('tool_calls') or []:
                code = parse(call['function']['arguments']).get('code', '')
                if code in seen:
                    continue
                seen.add(code)
                tree = ast.parse(code)
                selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
                if selected:
                    snippets.append(dict(request_line=line_number, action_count=request['action'],
                                         code=code, functions=[n.name for n in selected],
                                         result=message_result(request['messages'], index)))
    namespace = {}
    for snippet in snippets:
        tree = ast.parse(snippet['code'])
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
        exec(compile(ast.Module(body=selected, type_ignores=[]), '<recorded-functions>', 'exec'), namespace)
    with (SOURCE / 'intermediate_states.pkl').open('rb') as stream:
        all_states = ArtifactReader(stream).load()
    states = next(s for s in all_states if s[0].raw.game_id.startswith('s5i5'))
    benchmark = json.loads((SOURCE / 'benchmark.json').read_text())
    run = next(r for r in benchmark['game_runs'] if r['game_id'].startswith('s5i5'))
    history = parse(run['history'])
    assert len(states) == len(history) + 1
    frames = [SimpleNamespace(ascii='\n'.join(''.join(LETTERS[int(x)] for x in row)
                                            for row in state.raw.frame[-1])) for state in states]
    transitions = []
    for i, receipt in enumerate(history):
        action = receipt['action']
        assert dict(states[i+1].raw.action_input.data or {}) == (action.get('data') or {})
        if action['id'] == 'ACTION6':
            text = f"MOUSE(row={action['data']['y']}, col={action['data']['x']})"
        else:
            text = action['id']
        transitions.append(SimpleNamespace(action=text, before_frame=frames[i], after_frame=frames[i+1]))

    # Execute the exact original validation snippet, including its faulty selector.
    original = next(s for s in snippets if 'sim6' in s['functions'])
    calls, original_output = [], []
    original_ns = dict(namespace, current_frame=frames[-1], transitions=transitions,
                       print=lambda *args, **kw: original_output.append(' '.join(map(str, args))))
    original_tree = ast.parse(original['code'])
    # Install unchanged sim6, wrap only to count invocations, then execute the rest.
    definition = [n for n in original_tree.body if isinstance(n, ast.FunctionDef)]
    remainder = [n for n in original_tree.body if not isinstance(n, ast.FunctionDef)]
    exec(compile(ast.Module(body=definition, type_ignores=[]), '<original-definition>', 'exec'), original_ns)
    original_sim = original_ns['sim6']
    def count_calls(*args, **kwargs):
        calls.append(1)
        return original_sim(*args, **kwargs)
    original_ns['sim6'] = count_calls
    exec(compile(ast.Module(body=remainder, type_ignores=[]), '<original-checker>', 'exec'), original_ns)

    # Independently anchor the adapter to the trace's own printed observations.
    req115 = json.loads(requests_path.read_text().splitlines()[114])
    outputs115 = [parse(m['content']).get('stdout', '') for m in req115['messages'] if m['role']=='tool']
    logged_mismatch = next(s for s in outputs115 if 'MISMATCH at 0 N+' in s)
    expected_logged = ast.literal_eval(re.search(r' exp (\([^\n]+\))', logged_mismatch).group(1))
    observed_logged = ast.literal_eval(re.search(r' got (\([^\n]+\))', logged_mismatch).group(1))
    assert namespace['d2t'](namespace['extract2'](frames[35])) == observed_logged
    assert namespace['sim5'](namespace['d2t'](namespace['extract2'](frames[34])),
                             'N+', namespace['blkm'](frames[34])) == expected_logged
    req117 = json.loads(requests_path.read_text().splitlines()[116])
    printed = next(parse(m['content'])['stdout'] for m in req117['messages']
                   if m['role']=='tool' and '19 57, col=12)' in parse(m['content']).get('stdout',''))
    verified_trace_rows = 0
    for line in printed.splitlines():
        match = re.match(r'(\d+) ([0-9]+, col=[0-9]+\)) (.+)', line)
        if not match or int(match[1]) < 19:
            continue
        i = int(match[1])
        assert str(transitions[i].action).split('row=')[-1] == match[2]
        before = namespace['d2t'](namespace['extract2'](frames[i]))
        after = namespace['d2t'](namespace['extract2'](frames[i+1]))
        changed = [(color, before[3*j:3*j+3], after[3*j:3*j+3])
                   for j, color in enumerate(('O','Y','S','N')) if before[3*j:3*j+3] != after[3*j:3*j+3]]
        assert changed == ([] if match[3]=='NOOP' else ast.literal_eval(match[3]))
        verified_trace_rows += 1
    assert verified_trace_rows == 16

    rows = []
    for candidate, frontier in [('sim4', 34), ('sim5', 34), ('sim6', 35)]:
        for i, receipt in enumerate(history):
            # Original checker declares transitions 19 onward; require same level.
            if i < 19 or states[i].raw.levels_completed != 1 or states[i+1].raw.levels_completed != 1:
                continue
            action = receipt['action']
            data = action.get('data') or {}
            if action['id'] != 'ACTION6' or data.get('y') != 57 or data.get('x') not in ACTION_MAP:
                continue
            symbol = ACTION_MAP[data['x']]
            before = namespace['d2t'](namespace['extract2'](frames[i]))
            after = namespace['d2t'](namespace['extract2'](frames[i+1]))
            try:
                predicted = namespace[candidate](before, symbol, namespace['blkm'](frames[i]))
                if predicted is None:
                    verdict = 'rejected_observed_change' if before != after else 'abstained_on_noop'
                else:
                    verdict = 'match' if predicted == after else 'mismatch'
                error = None
            except Exception as exc:
                predicted, verdict, error = None, 'execution_error', repr(exc)
            rows.append(dict(candidate=candidate, transition_index=i, action=symbol,
                             evidence='historical' if i < frontier else 'later',
                             before=before, observed=after, predicted=predicted, verdict=verdict, error=error))
    report = dict(game=run['game_id'], source_run='out-control', levels_completed=run['levels_completed'],
                  source_hashes={str(p.relative_to(ROOT)): sha(p) for p in
                                 [requests_path, SOURCE/'benchmark.json', SOURCE/'intermediate_states.pkl']},
                  original_check=dict(request_line=original['request_line'],
                                      transitions_in_declared_range=len(transitions[19:]),
                                      simulator_calls=len(calls), stdout=original_output,
                                      validation_status='INVALID_EMPTY' if not calls else 'EXECUTED',
                                      logged_result=original['result']),
                  comparison_scope='Four extracted object bounding states; not full frames, goals, or terminal flags',
                  independent_checks=dict(trace_rows_verified=verified_trace_rows,
                                          logged_sim5_prediction_and_observation_reproduced=True),
                  elapsed_seconds=time.perf_counter()-started, comparisons=rows)
    (HERE/'generated-snippets.json').write_text(json.dumps(snippets, indent=2)+'\n')
    (HERE/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    # Freeze a possible next matched continuation at the first historical failure.
    # Keep future observations and later repairs OUT of this model-facing packet.
    historical = next(r for r in rows if r['candidate']=='sim4'
                      and r['evidence']=='historical' and r['verdict']=='rejected_observed_change')
    feedback = ('Historical replay counterexample for your current sim4. '
                f"Recorded transition {historical['transition_index']}: action {historical['action']}. "
                f"Before state: {historical['before']}. "
                f"sim4 returned None (the planner rejects this transition), but the recorded after state is {historical['observed']}. "
                'This action already occurred in your available history. Resolve this specific contradiction; '
                'a historical match does not establish correctness on unobserved transitions. '
                'Report how many transitions were selected and actually checked, alongside mismatches and skips.')
    control_request = json.loads(requests_path.read_text().splitlines()[106])
    assert control_request['action'] == 35
    (HERE/'continuation-control-request.json').write_text(json.dumps(control_request, indent=2)+'\n')
    (HERE/'repair-packet.json').write_text(json.dumps(dict(
        status='PREPARED_NOT_SENT', source_request_line=107, candidate='sim4',
        source_request_sha256=sha(HERE/'continuation-control-request.json'),
        historical_counterexample=historical, feedback=feedback,
        scope='Append feedback to the unchanged control request; later simulator revisions and transition 34 excluded.'
    ), indent=2)+'\n')
    (HERE/'frames.json').write_text(json.dumps([dict(index=i, ascii=frame.ascii,
                                                    levels_completed=states[i].raw.levels_completed)
                                              for i, frame in enumerate(frames)], indent=2)+'\n')
    print('Original check:',report['original_check'])
    for candidate in ('sim4', 'sim5', 'sim6'):
        for scope in ('historical', 'later'):
            subset=[r for r in rows if r['candidate']==candidate and r['evidence']==scope]
            counts={v:sum(r['verdict']==v for r in subset) for v in sorted({r['verdict'] for r in subset})}
            print(candidate,scope,len(subset),counts)


def message_result(messages, index):
    return messages[index+1].get('content') if index+1<len(messages) and messages[index+1]['role']=='tool' else None


if __name__ == '__main__':
    main()
