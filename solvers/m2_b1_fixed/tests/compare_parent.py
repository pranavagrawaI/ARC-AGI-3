"""Compare retained messages/cache-eviction flags with the preserved B1 parent."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]


def fixtures():
    for pin, turns, budget, drain, levels in itertools.product(
            (0, 500, 12288), (0, 4, 8), (2500, 9000, 10**6), (0, 2), (1, 2)):
        m = [{'role': 'system', 'content': 'sys'}]
        for i in range(18):
            level = 1 if levels == 1 or i < 4 else 2
            m.extend([{'role': 'user', 'content': f'Current state: step {i+1}, level {level}.\nTURN{i} ' + 'u'*800},
                      {'role': 'assistant', 'content': f'TURN{i} ' + 'a'*800,
                       'tool_calls': [{'id': f'call{i}', 'type': 'function',
                                       'function': {'name': 'python', 'arguments': '{"code":"print(1)"}'}}]},
                      {'role': 'tool', 'tool_call_id': f'call{i}', 'content': '1'}])
        m.append({'role': 'user', 'content': f'Current state: step 19, level {levels}.\nNOW'})
        yield (pin, turns, budget, drain), m

    # Reuse historical B1 requests, preserving their exact message payloads.
    archived = ROOT.parents[1] / 'analysis/counterexample-continuations-v3-20261013/inputs/b1.json'
    found = []
    def walk(value):
        if isinstance(value, dict):
            m = value.get('messages')
            if isinstance(m, list) and m and isinstance(m[0], dict) and m[0].get('role') == 'system':
                found.append(m)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(json.loads(archived.read_text()))
    assert found, 'No historical requests selected'
    for m in found:
        yield (12288, 8, 9000, 0), m


def collect(source):
    sys.path.insert(0, str(source / 'src'))
    pkg = types.ModuleType('inference.agent')
    pkg.__path__ = [str(source / 'src/inference/agent')]
    sys.modules['inference.agent'] = pkg
    os.environ.update({'ARC3_MEMORY_SECTIONS': 'off', 'LOCAL_ANALYZER_TOOL_TIMEOUT': '60'})
    from inference.agent import tool_agent as ta
    rows = []
    for settings, messages in fixtures():
        pin, turns, budget, drain = settings
        os.environ.update({'ARC3_PIN_LEVEL_OPENING_TOKENS': str(pin),
                           'ARC3_HISTORY_ASSISTANT_TURNS': str(turns),
                           'ARC3_HISTORY_TURN_DRAIN': str(drain)})
        for phase in ('request', 'commit', 'overflow'):
            a = ta.ToolAgent(model='comparison', base_url='http://127.0.0.1:9', api_key='unused')
            a._context_budget_tokens = budget
            entry = {'request': a._trim_messages_for_context,
                     'commit': a._persistent_history_messages,
                     'overflow': a._force_reduce_messages}[phase]
            out = entry(messages)
            payload = json.dumps(out, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            rows.append({'phase': phase, 'settings': settings,
                         'messages_sha256': hashlib.sha256(payload.encode()).hexdigest(),
                         'retained': len(out),
                         'context_was_trimmed': a._context_was_trimmed,
                         'has_evicted': a._has_evicted})
    return rows


if len(sys.argv) == 2:
    print(json.dumps(collect(Path(sys.argv[1]).resolve())))
else:
    outputs = []
    for source in (ROOT.parent / 'm2_b1', ROOT):
        result = subprocess.run([sys.executable, '-B', __file__, str(source)],
                                check=True, capture_output=True, text=True)
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1], 'Retained messages or cache-eviction flags changed'
    report = {'cases': len(outputs[0]), 'identical_retained_messages_and_flags': True,
              'scope': 'Synthetic boundaries and archived B1 requests; no model calls',
              'rows': outputs[1]}
    (ROOT / 'evidence/parent-comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Parent comparison PASS:', len(outputs[0]), 'identical outcomes')
