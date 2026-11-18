"""Behavioral regressions; real sandbox subprocesses, no model or GPU.

M2_SOURCE_ROOT selects the archived dfranzen tree for the red baseline.
Only the four AuditRegression tests are expected to fail on that tree.
"""
import ast
import copy
from enum import Enum
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace, MethodType
from typing import Any
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('M2_SOURCE_ROOT', ROOT / 'src')).resolve()
sys.path.insert(0, str(SOURCE))
package = ModuleType('inference.agent')
package.__path__ = [str(SOURCE / 'inference/agent')]
sys.modules['inference.agent'] = package
from inference.agent import python_tool_sandbox as sandbox
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state, load_runtime_state
from inference.utils.retained_functions import function_signature

STATE = {'current_frame': {'grid': [[0, 1]], 'ascii': 'Ww', 'shape': [1, 2], 'step': 0, 'level': 1},
         'history': [], 'valid_actions': ['UP']}


def extracted(name, namespace, *, cls='ToolAgent', path=None):
    path = path or SOURCE / 'inference/agent/tool_agent.py'
    tree = ast.parse(path.read_text())
    nodes = next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls) if cls else tree.body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def commit(pool, result):
    ns = {'_persistent_functions': lambda: True, '_persistent_functions_scope': lambda: 'game',
          '_get_env_bool': lambda *args: False, 'function_signature': function_signature}
    host = SimpleNamespace(_kept_functions=pool.copy())
    payload = {}
    extracted('_record_retained_functions', ns)(host, result, payload)
    return host._kept_functions, payload


class SandboxCase(unittest.TestCase):
    def setUp(self):
        self.receipts = []

    def handler(self, actions, **kwargs):
        receipt = {'executed': True, 'executed_count': len(actions),
                   'action_num': len(self.receipts) + 1, 'executed_actions': [a['action'] for a in actions]}
        self.receipts.append(receipt.copy())
        return {'action_result': receipt, 'state': STATE}

    def run_code(self, code, **kwargs):
        return sandbox.run_sandboxed_python(code=code, timeout_seconds=kwargs.pop('timeout', 3),
            initial_state=kwargs.pop('state', STATE), action_handler=kwargs.pop('handler', self.handler), **kwargs)


class AuditRegression(SandboxCase):
    def test_receipts_are_host_owned(self):
        result = self.run_code('r = action("UP"); r["executed_actions"].clear(); r.clear()')
        self.assertEqual(len(self.receipts), 1)
        self.assertTrue(result['action_results'][0].get('executed'))
        self.assertEqual(result['action_results'][0]['executed_actions'], ['UP'])

    def test_timeout_retains_previous_code(self):
        first = self.run_code('def inc(x): return x + 1', kept_functions=[], retain_imports=True)
        saved, _ = commit({}, first)
        self.assertEqual(set(saved), {'inc'})
        result = self.run_code('while True: pass', timeout=1, kept_functions=list(saved.values()))
        self.assertIn('timed out', result['error'])
        after, notice = commit(saved, result)
        self.assertEqual(after, saved)
        resumed = self.run_code('print(inc(3))', kept_functions=list(after.values()))
        self.assertEqual(resumed['stdout'].strip(), '4')

    def test_shadowed_builtin_rejects_dependency_and_wrapper(self):
        result = self.run_code('sum = 7\ndef value(): return sum\ndef wrapper(): return value()\nprint(wrapper())',
                               kept_functions=[], retain_imports=True)
        self.assertEqual(result['stdout'].strip(), '7')
        self.assertEqual(result['keepable_functions'], [])
        self.assertEqual({r['name'] for r in result['retention_rejected']}, {'value', 'wrapper'})

    def test_real_inspection_clears_no_tool_reason(self):
        ns = {'Path': Path, 'Any': Any, 'json': json,
              '_ToolDispatchResult': lambda content, **kw: SimpleNamespace(content=content, **kw),
              'load_runtime_state': lambda path: (None, []), '_normalize_valid_actions': list,
              '_ascii_frame_view_payload': lambda frame: STATE['current_frame'],
              '_ascii_history_view_payload': lambda history: [], '_get_env_bool': lambda *args: False,
              '_animation_enabled': lambda: False, 'run_sandboxed_python': sandbox.run_sandboxed_python}
        host = SimpleNamespace(_ensure_session=lambda path: None, _current_valid_actions=['UP'],
            _last_action_call_result={}, _step_env_callback=None, _noop_repeat_guard=None,
            _repeat_state_seen=set(), _python_timeout=3, _retained_sources=lambda: None,
            _record_retained_functions=lambda result, payload: None,
            _render_tool_payload=lambda payload, **kw: json.dumps(payload),
            _last_step_summary={'stale': True, 'stale_reason': 'no_tool_call', 'executed_actions': ['UP']})
        result = extracted('_run_python_tool', ns)(host, ROOT / 'unused.json', {'code': 'print(2 + 2)'})
        self.assertEqual(json.loads(result.content)['stdout'].strip(), '4')
        lines = extracted('_stale_summary_lines', ns, cls=None)(host._last_step_summary)
        self.assertIn('previous Python tool call', lines[0])
        self.assertNotIn('produced no tool call', lines[0])

        for code in ('for', ''):
            with self.subTest(rejected_code=code):
                host._last_step_summary={'stale':True,'stale_reason':'no_tool_call','executed_actions':['UP']}
                rejected=extracted('_run_python_tool',ns)(host,ROOT/'unused.json',{'code':code})
                self.assertTrue(json.loads(rejected.content)['error'])
                lines=extracted('_stale_summary_lines',ns,cls=None)(host._last_step_summary)
                self.assertNotIn('produced no tool call',lines[0])


class RecoveryAndControls(SandboxCase):
    def test_missing_final_snapshot_is_not_an_empty_snapshot(self):
        program='import sys,json;sys.stdin.readline();print(json.dumps({"type":"final","stdout":"4"}))'
        saved={'inc':'def inc(x): return x+1'}
        with patch.object(sandbox,'_SANDBOX_BOOTSTRAP',program):
            result=self.run_code('pass',kept_functions=list(saved.values()))
        self.assertEqual(result['stdout'],'4')
        self.assertNotIn('keepable_functions',result)
        self.assertEqual(commit(saved,result)[0],saved)

    def test_timeout_preserves_action_and_rolls_back_code_only(self):
        saved = {'inc': 'def inc(x): return x + 1'}
        result = self.run_code('action("UP")\ndef inc(x): return 99\nwhile True: pass', timeout=1,
                               kept_functions=list(saved.values()))
        self.assertEqual(len(self.receipts), 1)
        self.assertTrue(result['action_results'][0]['executed'])
        after, _ = commit(saved, result)
        self.assertEqual(after, saved)

    def test_explicit_deletion_commits_empty_pool(self):
        saved = {'inc': 'def inc(x): return x + 1'}
        result = self.run_code('del inc', kept_functions=list(saved.values()))
        self.assertEqual(commit(saved, result)[0], {})

    def test_builtin_and_explicit_argument_helpers_still_work(self):
        first = self.run_code('def total(values): return sum(values)\ndef shifted(x, offset): return x+offset',
                              kept_functions=[], retain_imports=True)
        self.assertFalse(first['retention_rejected'])
        again = self.run_code('print(total([2,3]), shifted(4,7))',
                              kept_functions=[i['source'] for i in first['keepable_functions']])
        self.assertEqual(again['stdout'].strip(), '5 11')

    def test_ordinary_exception_preserves_completed_definitions(self):
        result = self.run_code('def inc(x): return x+1\n1/0', kept_functions=[])
        self.assertIn('ZeroDivisionError', result['error'])
        self.assertIn('inc', commit({}, result)[0])

    def test_terminal_latch_stops_followup_action(self):
        def terminal(actions, **kw):
            result = self.handler(actions, **kw)
            result['action_result']['level_completed'] = True
            return result
        result = self.run_code('action("UP")\ntry:\n action("UP")\nexcept Exception:\n pass', handler=terminal)
        self.assertEqual(len(self.receipts), 1)
        self.assertIn('TerminalStateActionError', result['error'])

    def test_malformed_or_eof_child_is_reaped(self):
        for program in ('import sys,time;sys.stdin.readline();print("invalid",flush=True);time.sleep(10)',
                        'import os,sys,time;sys.stdin.readline();os.close(1);time.sleep(10)'):
            with self.subTest(program=program):
                children = []
                real_popen = sandbox.subprocess.Popen
                def spawn(*a, **kw):
                    child = real_popen(*a, **kw); children.append(child); return child
                start = time.monotonic()
                with patch.object(sandbox, '_SANDBOX_BOOTSTRAP', program), patch.object(sandbox.subprocess, 'Popen', spawn):
                    result = self.run_code('pass', timeout=1)
                self.assertTrue(result['error'])
                self.assertLess(time.monotonic()-start, 3)
                self.assertIsNotNone(children[0].poll())
                self.assertTrue(children[0].stdin.closed)

    def test_initial_pipe_write_has_deadline(self):
        with patch.object(sandbox, '_SANDBOX_BOOTSTRAP', 'import time;time.sleep(10)'):
            start = time.monotonic()
            result = self.run_code('pass', timeout=0.15, state={'large': 'x' * 2_000_000})
        self.assertTrue(result['timed_out'])
        self.assertLess(time.monotonic()-start, 2)


class AnimationHistory(SandboxCase):
    def test_actual_execute_action_records_animation_before_auto_reset(self):
        from inference.agent.runtime_state import append_history_entry
        from inference.utils.animation import history_animation, collapse_frames, normalize_frames, summarize_animation
        from inference.agent.action_names import to_model_actions
        path=SOURCE/'inference/framework/solver.py'
        actions=Enum('Actions', 'ACTION1 ACTION6 RESET')
        statuses=Enum('Statuses', 'PLAYING GAME_OVER WIN')
        engine=SimpleNamespace(GameState=statuses, GameAction=actions,
                               ActionInput=lambda **kw: SimpleNamespace(**kw))
        ns={'Any':Any,'arcengine':engine,'taaf':SimpleNamespace(game=SimpleNamespace(GameState=object)),
            'Frame':Frame,'HistoryEntry':HistoryEntry,'copy':copy,'ANIMATION_ENABLED':True,
            'append_history_entry':append_history_entry,'history_animation':history_animation,
            'collapse_frames':collapse_frames,'normalize_frames':normalize_frames,
            'summarize_animation':summarize_animation,'to_model_actions':to_model_actions,
            '_format_action_display':lambda name,data:name,'_level_number':lambda game:1,
            '_engine_action_names':lambda game:['ACTION1'],
            '_interior_gameplay_changed':lambda a,b:a!=b}
        for name in ['_grid_from_state','_animation_chain']:
            extracted(name,ns,cls=None,path=path)
        def state(status,frames):
            return SimpleNamespace(frame=SimpleNamespace(data=frames[-1]), levels_completed=0,
                                   just_won_level=False,raw=SimpleNamespace(state=status,frame=frames))
        host=SimpleNamespace(action_count=0,history_entries=[HistoryEntry('',Frame(((0,),),0,1))],
            timing_payload=lambda:{},_append_action_viewer_event=lambda *args:None,
            write_viewer_payload=lambda:None)
        game=SimpleNamespace(current_state=state(statuses.PLAYING,[[[0]]]),number_of_levels=3)
        host.game=game
        def execute(action,**kwargs):
            host.action_count+=1
            game.current_state=(state(statuses.PLAYING,[[[0]]]) if action.id==actions.RESET else
                                state(statuses.GAME_OVER,[[[1]],[[0]]]))
            return game.current_state
        game.execute_action=execute
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'state.json'
            host.write_runtime_state=lambda:write_runtime_state(target,current_frame=host.history_entries[-1].frame,history=host.history_entries)
            host._execute_action=MethodType(extracted('_execute_action',ns,cls='_HarnessGameSession',path=path),host)
            host._execute_action(SimpleNamespace(id=actions.ACTION1,data={}),batch_index=1,batch_size=1,generated_tokens=0)
            extracted('_execute_auto_reset',ns,cls='_HarnessGameSession',path=path)(host)
            _,history=load_runtime_state(target)
        self.assertTrue(history[-2].result['game_over'])
        self.assertEqual(history[-2].animation['frames'],['1'])
        self.assertTrue(history[-1].result['automatic'])
        self.assertEqual(history[-1].animation,{})

    def test_sampling_is_explicit_and_retains_endpoints(self):
        from inference.utils.animation import history_animation
        chain = [((i % 16,),) for i in range(102)]
        record = history_animation(chain)
        self.assertEqual(len(record['frames']), 32)
        self.assertEqual((record['indices'][0],record['indices'][-1]), (1,100))
        self.assertTrue(record['sampled'])
        self.assertEqual(record['intermediate_count'], 100)

    def test_eviction_and_reset_keep_correct_action_evidence(self):
        from inference.agent.runtime_state import append_history_entry, history_entry_to_payload
        from inference.utils.animation import history_animation
        history = [HistoryEntry('', Frame(((0,),),0,1))]
        for i in range(1,10):
            append_history_entry(history, HistoryEntry('UP', Frame(((0,),),i,1),
                {'game_over': i==9}, history_animation([((0,),),((i,),),((0,),)])))
        append_history_entry(history, HistoryEntry('RESET', Frame(((0,),),10,1), {'automatic': True}))
        self.assertFalse(history[1].animation['retained'])
        self.assertEqual(history[1].animation['intermediate_count'], 1)
        self.assertEqual(sum(bool(e.animation.get('frames')) for e in history), 8)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.json'
            write_runtime_state(path, current_frame=history[-1].frame, history=history)
            current, restored = load_runtime_state(path)
        state={'current_frame': {'grid': [[0]], 'ascii':'W', 'shape':[1,1], 'level':1,'step':10},
               'history':[history_entry_to_payload(e) for e in restored]}
        result=self.run_code('print(transitions[-2].action, transitions[-2].frames[0].ascii, transitions[-1].frames)\nprint(transitions[0].animation_info["retained"])',state=state)
        self.assertFalse(result['error'], result)
        self.assertEqual(result['stdout'].splitlines(), ['UP b []', 'False'])


if __name__ == '__main__':
    unittest.main()
