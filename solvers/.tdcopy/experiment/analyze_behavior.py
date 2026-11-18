"""Summarize frozen role/tool traces without modifying them or calling a model."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def arguments(event):
    value = event['arguments']
    return json.loads(value) if isinstance(value, str) else value


def analyze(directory):
    directory = Path(directory)
    root = directory / 'output/tycho-dfranzen'
    index = {r['path']: r['sha256'] for r in json.loads((root / 'evidence-index.json').read_text())['files']}
    report = {'run': directory.name, 'evidence': [], 'games': []}
    for game in sorted((root / 'hybrid-games').iterdir()):
        llm = jsonl(game / 'llm-trace.jsonl')
        tools = jsonl(game / 'tool-trace.jsonl')
        builder_tools = jsonl(game / 'builder-tool-trace.jsonl')
        triggers = jsonl(game / 'trigger-builder-trace.jsonl')
        for name in ('llm-trace.jsonl', 'tool-trace.jsonl', 'builder-tool-trace.jsonl', 'trigger-builder-trace.jsonl', 'host-actions.jsonl'):
            path = game / name
            if path.exists():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                relative = path.relative_to(directory / 'output').as_posix()
                assert digest == index[relative]
                report['evidence'].append({'path': relative, 'sha256': digest})
        roles = {}
        for r in llm:
            role = roles.setdefault(r['source'], dict(calls=0, completion_tokens=0, reasoning_tokens=0,
                                                     prompt_tokens=0, cached_tokens=0))
            role['calls'] += 1
            for key in ('completion_tokens', 'reasoning_tokens', 'prompt_tokens', 'cached_tokens'):
                role[key] += r.get(key) or 0
        writes = [{'source': t['source'], 'sequence': t['sequence'], 'level': t['level'],
                   'name': t['name'], 'path': arguments(t).get('path'), 'status': t['execution_status']}
                  for t in tools if t['name'] in ('write_file', 'edit_file', 'edit_function')]
        passes = []
        for sequence in sorted({r['sequence'] for r in llm if r['source'] == 'builder'}):
            calls = [r for r in llm if r['source'] == 'builder' and r['sequence'] == sequence]
            events = [t for t in builder_tools if t['sequence'] == sequence]
            passes.append(dict(sequence=sequence, level=calls[0]['level'], turn_in_level=calls[0]['turn_in_level'],
                first_call=calls[0]['call_index'], last_call=calls[-1]['call_index'], calls=len(calls),
                completion_tokens=sum(c['completion_tokens'] for c in calls),
                initial_prompt_tokens=calls[0]['prompt_tokens'], final_prompt_tokens=calls[-1]['prompt_tokens'],
                tools=dict(Counter(t['name'] for t in events)),
                failures=[{'call_index': t['call_index'], 'name': t['name'], 'stage': t['error_stage'],
                           'result': t['result']} for t in events if t['execution_status'] != 'complete'],
                writes=[{'path': arguments(t).get('path'), 'status': t['execution_status']}
                        for t in events if t['name'] in ('write_file', 'edit_file', 'edit_function')],
                maximum_tool_output_chars=max(len(str(t['result'])) for t in events)))
        samples = []
        builder_indices = {r['call_index'] for r in llm if r['source'] == 'builder'}
        for path in sorted((game / 'decision-checkpoints/calls').glob('*/request.json')):
            if int(path.parent.name) not in builder_indices:
                continue
            body = json.loads(path.read_text())['request_body']
            samples.append(dict(call_index=int(path.parent.name), messages=len(body['messages']),
                budget_notices=sum('BUILDER BUDGET' in str(m.get('content', '')) for m in body['messages']),
                offered_tools=[t['name'] for t in body['tools']], tool_choice=body.get('tool_choice')))
        report['games'].append(dict(game=game.name, roles=roles,
            tool_counts=dict(Counter(t['name'] for t in tools)), writes=writes,
            tools_calling_donor_segment=sum('dfr.segment(' in arguments(t).get('code', '') for t in tools + builder_tools),
            builder_passes=passes, builder_request_samples=samples, builder_events=triggers,
            actor_longest_decisions=Counter(r['sequence'] for r in llm if r['source'] == 'actor').most_common(5)))
    (directory / 'BEHAVIOR.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'run': report['run'], 'games': len(report['games']), 'hash_checked_traces': len(report['evidence'])}))


if __name__ == '__main__':
    analyze(sys.argv[1])
