"""CPU-only checks of archived winner sources; never launches a server or edits them.

Run with the submission's CPython 3.12.12. Modes run separately to avoid mixing
the two different `inference` packages in one interpreter.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
mode = sys.argv[1]
sources = {}


def read(path):
    path = Path(path)
    text = path.read_text()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return text


def method(path, name, namespace):
    """Compile one unchanged method; dependencies supplied explicitly below."""
    tree = ast.parse(read(path))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ToolAgent')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


if mode == 'siri':
    notebook = json.loads(read(ROOT / 'sirikilohit/arc-agi-3-duck-18-1gc-submit.ipynb'))
    wanted = {'_W90_SEG_SRC': 'm90_wang_segmentation.py', '_W90_SANDBOX_SRC': 'm90_wang_sandbox.py'}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        try:
            tree = ast.parse(''.join(cell['source']))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in wanted:
                    (HERE / wanted[name]).write_text(ast.literal_eval(node.value))
    sys.path.insert(0, str(HERE))
    # This supplies only the common grid alphabet imported by the embedded runner.
    sys.path.insert(0, str(ROOT / 'audit/dfranzen-repo/ARC3-Inference'))
    import m90_wang_sandbox as sandbox
elif mode in ('dfr', 'lord'):
    base = ROOT / ('audit/dfranzen-repo/ARC3-Inference' if mode == 'dfr' else 'audit/lord-bundle/src/ARC3-Inference')
    sys.path.insert(0, str(base))
    # Skip agent/__init__.py, whose eager ToolAgent import pulls in HTTP/vision
    # dependencies. Sandbox and all its own dependencies remain unmodified.
    agent_package = ModuleType('inference.agent')
    agent_package.__path__ = [str(base / 'inference/agent')]
    sys.modules['inference.agent'] = agent_package
    from inference.agent import python_tool_sandbox as sandbox
else:
    raise ValueError(mode)

read(sandbox.__file__)
state = {'current_frame': {'grid': [[0, 1], [2, 3]], 'ascii': 'Ww\ngG', 'shape': [2, 2], 'level': 1, 'step': 0}, 'history': [], 'valid_actions': ['UP']}
events = []


def handler(actions, **kwargs):
    receipt = {'executed': True, 'executed_count': len(actions), 'action_num': len(events) + 1}
    events.append(receipt)
    return {'action_result': receipt, 'state': state}


def run(code, timeout=3, **kwargs):
    start = time.monotonic()
    result = sandbox.run_sandboxed_python(code=code, timeout_seconds=timeout,
                                         initial_state=state, action_handler=handler, **kwargs)
    return {'seconds': round(time.monotonic() - start, 4), **result}


checks = {}
checks['ordinary_execution'] = run('print(current_frame.ascii)')
assert checks['ordinary_execution']['error'] == ''
before = len(events)
checks['mutable_action_receipt'] = run('receipt = action("UP")\nreceipt.clear()')
checks['mutable_action_receipt']['actual_host_receipts'] = events[before:]
assert len(events) == before + 1 and events[-1]['executed']
assert checks['mutable_action_receipt']['action_results'] == [{}]
checks['mutable_action_receipt']['caller_step_executed'] = any(
    bool(item.get('executed')) for item in checks['mutable_action_receipt']['action_results'])
assert not checks['mutable_action_receipt']['caller_step_executed']
checks['timeout_receipt'] = run('action("UP")\nwhile True: pass', timeout=1)
assert len(checks['timeout_receipt']['action_results']) == 1
assert 'timed out' in checks['timeout_receipt']['error']

if mode == 'dfr':
    from inference.utils.retained_functions import function_signature
    read(base / 'inference/utils/retained_functions.py')
    opts = {'kept_functions': [], 'retain_imports': True, 'repair_hints': False}
    first = run('sum = 7\ndef value():\n    return sum\nprint(value())', **opts)
    second = run('print(value())', kept_functions=[i['source'] for i in first['keepable_functions']],
                 retain_imports=True, repair_hints=False)
    checks['shadowed_builtin'] = {'first': first, 'second': second}
    assert first['stdout'].strip() == '7'
    assert len(first['keepable_functions']) == 1 and not first['retention_rejected']
    assert 'built-in function sum' in second['stdout']
    control = run('offset = 7\ndef value():\n    return offset', **opts)
    assert not control['keepable_functions'] and control['retention_rejected']
    checks['ordinary_dependency_rejected_control'] = control

    good = run('def inc(x):\n    return x + 1\nprint(inc(3))', **opts)
    saved = {i['name']: i['source'] for i in good['keepable_functions']}
    timeout = run('while True: pass', timeout=1, kept_functions=list(saved.values()), retain_imports=True, repair_hints=False)
    namespace = {'_persistent_functions': lambda: True, '_persistent_functions_scope': lambda: 'game',
                 '_get_env_bool': lambda *args: False, 'function_signature': function_signature}
    record = method(base / 'inference/agent/tool_agent.py', '_record_retained_functions', namespace)
    host = SimpleNamespace(_kept_functions=saved.copy())
    payload = {}
    record(host, timeout, payload)
    checks['timeout_erases_preexisting_helpers'] = {'before': saved, 'timeout': timeout, 'after': host._kept_functions, 'notice': payload}
    assert saved and host._kept_functions == {} and 'keepable_functions' not in timeout
    # An ordinary error delivers a complete retention snapshot and keeps inc.
    ordinary = run('1 / 0', kept_functions=list(saved.values()), retain_imports=True, repair_hints=False)
    host._kept_functions = saved.copy()
    record(host, ordinary, {})
    assert host._kept_functions == saved
    checks['ordinary_error_retains_control'] = ordinary
    # The terminal latch must stop a second action even under except Exception.
    def terminal_handler(actions, **kwargs):
        reply = handler(actions, **kwargs)
        reply['action_result']['level_completed'] = True
        return reply
    before = len(events)
    terminal = sandbox.run_sandboxed_python(code='action("UP")\ntry:\n    action("UP")\nexcept Exception:\n    pass',
        timeout_seconds=3, initial_state=state, action_handler=terminal_handler)
    assert len(events) == before + 1 and 'TerminalStateActionError' in terminal['error']
    checks['terminal_guard_control'] = terminal
    # Invoke the complete unchanged tool method with deterministic observation
    # adapters. The actual subprocess runs; no game engine or HTTP is needed.
    ns = {'Path': Path, 'Any': Any, 'json': json, '_ToolDispatchResult': SimpleNamespace,
          'load_runtime_state': lambda path: (None, []),
          '_normalize_valid_actions': list,
          '_ascii_frame_view_payload': lambda frame: state['current_frame'],
          '_ascii_history_view_payload': lambda history: [],
          '_get_env_bool': lambda *args: False, '_animation_enabled': lambda: False,
          'run_sandboxed_python': sandbox.run_sandboxed_python}
    run_tool = method(base / 'inference/agent/tool_agent.py', '_run_python_tool', ns)
    host = SimpleNamespace(_ensure_session=lambda path: None, _current_valid_actions=['UP'],
        _last_action_call_result={}, _step_env_callback=None, _noop_repeat_guard=None,
        _repeat_state_seen=set(), _python_timeout=3, _retained_sources=lambda: None,
        _record_retained_functions=lambda result, payload: None,
        _render_tool_payload=lambda payload, **kwargs: json.dumps(payload),
        _last_step_summary={'stale': True, 'stale_reason': 'no_tool_call', 'executed_actions': ['UP']})
    # SimpleNamespace is keyword-only, whereas dispatch is constructed positionally.
    ns['_ToolDispatchResult'] = lambda content, **kwargs: SimpleNamespace(content=content, **kwargs)
    dispatch = run_tool(host, HERE / 'unused-state.json', {'code': 'print(2 + 2)'})
    assert json.loads(dispatch.content)['stdout'].strip() == '4'
    tree = ast.parse(read(base / 'inference/agent/tool_agent.py'))
    stale = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_stale_summary_lines')
    exec(compile(ast.Module(body=[stale], type_ignores=[]), str(base / 'inference/agent/tool_agent.py'), 'exec'), ns)
    lines = ns['_stale_summary_lines'](host._last_step_summary)
    assert 'produced no tool call' in lines[0]
    checks['stale_no_tool_reason_after_real_tool'] = {'real_tool_result': json.loads(dispatch.content),
        'summary_after_tool': host._last_step_summary, 'next_prompt_lines': lines}

if mode == 'lord':
    checks['workspace_save'] = run('save_module("helper", "def f(): return 1", "check(module.f(), 1)")\nprint(load_module("helper").f())')
    assert not checks['workspace_save']['error']
    saved = checks['workspace_save']['modules']
    checks['workspace_failed_save'] = run('save_module("helper", "def f(): return 2", "check(module.f(), 1)")', modules=saved)
    assert checks['workspace_failed_save']['error']
    checks['workspace_timeout'] = run('while True: pass', timeout=1, modules=saved)
    # Actual ToolAgent commit rule: only a successful call replaces modules.
    committed = saved
    for result in (checks['workspace_failed_save'], checks['workspace_timeout']):
        if not result.get('error') and isinstance(result.get('modules'), dict):
            committed = result['modules']
    assert committed == saved
    checks['workspace_error_commit_control'] = {'saved_module_names_after_errors': sorted(committed)}

result = {'mode': mode, 'python': sys.version, 'sources_sha256': sources, 'checks': checks}
(HERE / f'{mode}-results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
