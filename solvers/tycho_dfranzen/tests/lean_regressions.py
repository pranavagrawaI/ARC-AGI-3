"""Actor-only lean treatments: per-decision call cap, closed-decision pruning, lean prompt.

A loopback OpenAI-compatible server drives the real worker process; every request
it receives is the exact provider request. Writes nothing under evidence/.
"""
from __future__ import annotations
import asyncio
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE/'runtime'))
for _key in list(os.environ):
    if _key.startswith(('ARC_LLM_', 'OPENAI_', 'OPENROUTER_', 'ANTHROPIC_', 'NOOA_LLM_', 'TYCHO_NOOA_SCRIPTED')):
        os.environ.pop(_key)
os.environ.update(LITELLM_LOCAL_MODEL_COST_MAP='True', TYCHO_SANDBOX_RUNTIME='host',
    MULTIMODAL_CONTEXT='current_grid', MULTIMODAL_UPSCALE='10',
    ARC3_MAX_ACTIVE_STREAMS='0', ARC3_PRIORITY_TAIL_FADE='0')

import arcengine
from taaf.benchmark import Benchmark
from taaf.game_examples import ExampleGame
from hybrid.solver import HybridSolver
from hybrid import lean, policy

OBSERVATION = 'Investigate the following new public observation'
REASONING = 'fixture reasoning for this call'
BULK = 'x' * 5000  # one run_python result, far above the closed-decision bound


def text(message):
    content = message.get('content')
    if isinstance(content, list):
        return ''.join(part.get('text', '') for part in content)
    return content or ''


def images(message):
    content = message.get('content')
    return sum(part.get('type') == 'image_url' for part in content) if isinstance(content, list) else 0


def newest_observation(messages):
    return max(i for i, m in enumerate(messages) if m['role'] == 'user' and OBSERVATION in text(m))


def decision_of(messages):
    return int(re.findall(r"'sequence': (\d+)", text(messages[newest_observation(messages)]))[0])


class Provider:
    """plan[d] = tool-using calls before acting voluntarily (None: never);
    refuse[d] = action-only calls answered without take_actions; burst[d] = actions per batch."""

    def __init__(self, plan, refuse=None, burst=None):
        self.plan, self.refuse, self.burst, self.calls = plan, refuse or {}, burst or {}, []
        self.explored, self.demanded = {}, {}

    def reply(self, body):
        self.calls.append(body)
        d = decision_of(body['messages'])
        if [tool['function']['name'] for tool in body['tools']] == ['take_actions']:
            self.demanded[d] = self.demanded.get(d, 0) + 1
            act = self.demanded[d] > self.refuse.get(d, 0)
        else:
            self.explored[d] = self.explored.get(d, 0) + 1
            act = self.plan[d] is not None and self.explored[d] > self.plan[d]
        if act:
            name, arguments = 'take_actions', {'actions': [{'action': 'UP'}]*self.burst.get(d, 1),
                                               'rationale': 'lean fixture'}
        else:
            name, arguments = 'run_python', {'code': f"print('{BULK}')"}
        return {'role': 'assistant', 'content': '', 'reasoning_content': REASONING,
                'tool_calls': [{'id': f'call-{len(self.calls)}', 'type': 'function',
                                'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


def run(treatment, provider):
    """Play a three-action ExampleGame through the real worker; return game state and requests."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'healthy')

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            message = provider.reply(body)
            wire = json.dumps({'id': 'lean', 'object': 'chat.completion', 'created': 0, 'model': 'flashnext',
                'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls'}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}}).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(wire))); self.end_headers(); self.wfile.write(wire)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    os.environ.update(TYCHO_MODE='no_world_model', TYCHO_NOOA_TREATMENT=treatment,
        ARC_LLM_MODEL='flashnext', ARC_LLM_API_KEY='loopback-fixture',
        ARC_LLM_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',
        ARC_LLM_CONTEXT_WINDOW=str(128*1024), ARC_LLM_RETRIES='0', ARC_LLM_USE_RESPONSES='0',
        ARC_LLM_TIMEOUT_SECONDS='10', ARC3_HTTP_RETRY_INITIAL_SECONDS='10')
    try:
        with tempfile.TemporaryDirectory() as directory:
            game = ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1]*3], label='lean-fixture')
            bm = Benchmark(games=[game], job_dir=Path(directory), solver=HybridSolver(
                worker_python=sys.executable, runtime_root=str(HERE/'runtime'),
                max_runtime_s_per_game=90, max_actions_per_game=3, analyzer_timeout=60))
            asyncio.run(bm.run())
            game_dir = next((Path(directory)/'hybrid-games').iterdir())
            compute = json.loads((game_dir/'compute.json').read_text())
            windows = {json.loads(line)['window'] for line in (game_dir/'context-projection.jsonl').read_text().splitlines()}
            return bm.game_runs[0], compute, provider.calls, windows
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
        for key in [k for k in os.environ if k.startswith('ARC_LLM_')] + ['ARC3_HTTP_RETRY_INITIAL_SECONDS']:
            os.environ.pop(key, None)


def per_decision(calls):
    counts = {}
    for call in calls:
        d = decision_of(call['messages'])
        counts[d] = counts.get(d, 0) + 1
    return counts


class LeanSelection(unittest.TestCase):
    def test_lean_requires_actor_only_mode(self):
        self.assertIsNone(lean.lean({'TYCHO_NOOA_TREATMENT': 'control', 'TYCHO_MODE': 'orchestrator'}))
        self.assertEqual(lean.lean({'TYCHO_NOOA_TREATMENT': 'lean_cap4', 'TYCHO_MODE': 'no_world_model'}).tool_calls, 4)
        with self.assertRaises(ValueError):
            lean.lean({'TYCHO_NOOA_TREATMENT': 'lean_cap4', 'TYCHO_MODE': 'orchestrator'})

    def test_canonical_table_knows_every_lean_name_without_a_scribe(self):
        from tycho_nooa.treatments import TREATMENTS, DETERMINISTIC_LEVEL_CHECKPOINT
        for name in lean.TREATMENTS:
            self.assertEqual(TREATMENTS[name].level_boundary_policy, DETERMINISTIC_LEVEL_CHECKPOINT)
            self.assertFalse(TREATMENTS[name].decision_discipline)

    def test_only_a_capped_arm_announces_a_budget(self):
        os.environ.update(TYCHO_MODE='no_world_model', TYCHO_NOOA_TREATMENT='lean_prune')
        self.assertEqual(policy.system_guidance(), policy.SYSTEM_GUIDANCE)
        os.environ['TYCHO_NOOA_TREATMENT'] = 'lean_cap4'
        self.assertIn('at most 4 model calls', policy.system_guidance())
        os.environ.update(TYCHO_MODE='orchestrator', TYCHO_NOOA_TREATMENT='control')
        self.assertEqual(policy.system_guidance(), policy.SYSTEM_GUIDANCE)


class Pruning(unittest.TestCase):
    def test_prunes_closed_decisions_once_and_keeps_tool_results(self):
        from nooa.runtime.event_manager import EventManager
        from nooa.events import LLMOutput, Task
        from nooa.context_blocks.events import ToolCallEvent, ToolResult
        from tycho_nooa.history import OBSERVATION_REF
        events = EventManager()
        self.assertEqual(lean.prune_closed_decisions(events, 1536), 0)  # nothing observed yet
        old_task = events.add(Task(prompt='old', images=[{'type': 'image_url', 'image_url': {'url': 'data:'}}],
                                   metadata={OBSERVATION_REF: {'level': 0, 'turn': 0}}))
        old_call = events.add(ToolCallEvent(tool_call_id='a', name='run_python', arguments={'code': BULK},
                                            reasoning=REASONING, result=ToolResult(tool_call_id='a', content=BULK)))
        old_prose = events.add(LLMOutput(content=BULK, reasoning=REASONING))
        events.add(Task(prompt='new', images=[{'type': 'image_url', 'image_url': {'url': 'data:'}}],
                        metadata={OBSERVATION_REF: {'level': 0, 'turn': 1}}))
        new_call = events.add(ToolCallEvent(tool_call_id='b', name='run_python', arguments={'code': BULK},
                                            reasoning=REASONING, result=ToolResult(tool_call_id='b', content=BULK)))
        self.assertEqual(lean.prune_closed_decisions(events, 1536), 3)
        self.assertEqual(lean.prune_closed_decisions(events, 1536), 0)  # idempotent: no cache churn
        pruned = events[old_call]
        self.assertIsNone(pruned.reasoning)
        self.assertLessEqual(len(pruned.result.content.encode()), 1536)
        self.assertLessEqual(len(pruned.arguments['code'].encode()), 1536)
        self.assertEqual(pruned.result.tool_call_id, 'a')
        self.assertEqual(events[old_task].images, [])
        self.assertIsNone(events[old_prose].reasoning)
        self.assertLessEqual(len(events[old_prose].content.encode()), 1536)
        self.assertEqual(events[new_call].reasoning, REASONING)  # the open decision is untouched
        self.assertEqual(events[new_call].result.content, BULK)


class LeanGameplay(unittest.TestCase):
    def assert_closed_decisions_pruned(self, calls):
        for call in calls:
            messages = call['messages']
            newest = newest_observation(messages)
            closed, open_ = messages[1:newest], messages[newest+1:]
            self.assertFalse(any(m.get('reasoning_content') for m in closed))
            self.assertTrue(all(len(text(m).encode()) <= 1536 for m in closed if m['role'] == 'tool'))
            self.assertEqual(sum(images(m) for m in closed), 0)
            self.assertGreater(images(messages[newest]), 0)
            self.assertEqual(sum('grid (row labels' in text(m) for m in messages), 1)
            if any(m['role'] == 'assistant' for m in open_):
                self.assertTrue(any(m.get('reasoning_content') == REASONING for m in open_))

    def assert_lean_prompt(self, calls):
        system = text(calls[0]['messages'][0])
        self.assertIn(policy.SYSTEM_GUIDANCE.splitlines()[0], system)
        for absent in ('symmetry.canon', 'dynamics.key_counterexample', 'An option is asymmetric', 'invoke_builder'):
            self.assertNotIn(absent, system)
        offered = {t['function']['name'] for c in calls for t in c['tools']}
        self.assertNotIn('invoke_builder', offered)

    def test_cap_forces_action_after_four_calls_and_retries_once(self):
        provider = Provider(plan={0: None, 1: 1, 2: None}, refuse={2: 1})
        run_, compute, calls, windows = run('lean_cap4', provider)
        self.assertEqual(windows, {64*1024})
        self.assertEqual(run_.state, 'won', run_.solver_note)
        self.assertEqual(per_decision(calls), {0: 5, 1: 2, 2: 6})
        demands = [c for c in calls if [t['function']['name'] for t in c['tools']] == ['take_actions']]
        self.assertEqual(len(demands), 3)
        for call in demands:
            self.assertFalse(call['chat_template_kwargs']['enable_thinking'])
            self.assertEqual(call['top_k'], 20)
            self.assertEqual(call['max_tokens'], 2048)
        self.assertIn('Decision budget: each decision allows at most 4', text(calls[0]['messages'][0]))
        self.assert_lean_prompt(calls)
        self.assert_closed_decisions_pruned(calls)
        self.assertEqual(set(compute['roles']), {'actor'})

    def test_burst_continuations_keep_one_inlined_grid(self):
        provider = Provider(plan={0: 1, 2: 0}, burst={0: 2})
        run_, compute, calls, windows = run('lean_prune', provider)
        self.assertEqual(run_.state, 'won', run_.solver_note)
        self.assertEqual(per_decision(calls), {0: 2, 2: 1})  # the second UP needed no model call
        last = calls[-1]['messages']
        self.assertTrue(any('BATCH CONTINUATION RECEIPT' in text(m) for m in last))
        self.assert_closed_decisions_pruned(calls)

    def test_uncapped_lean_keeps_the_old_ceiling(self):
        provider = Provider(plan={0: 6, 1: 0, 2: 0})
        run_, compute, calls, windows = run('lean_prune', provider)
        self.assertEqual(windows, {64*1024})
        self.assertEqual(run_.state, 'won', run_.solver_note)
        self.assertEqual(per_decision(calls), {0: 7, 1: 1, 2: 1})
        self.assertNotIn('Decision budget', text(calls[0]['messages'][0]))
        self.assert_lean_prompt(calls)
        self.assert_closed_decisions_pruned(calls)
        self.assertEqual(set(compute['roles']), {'actor'})


if __name__ == '__main__':
    unittest.main()
