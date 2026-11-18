"""The contingent interface is the isolated treatment: off must remove it everywhere.

Off and on differ only in ARC_CONTINGENT_ARM. Each case runs the actual adapter,
transport, host, dispatcher and CodeAct against a deterministic local model and
inspects the real request the worker sent, so the assertions are about the
prompt and tool surface the model actually received, not about a rendered copy.
"""
import importlib.util
import json
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from test_contingent_controller import relevant_proposal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE / 'chassis'))
TOOLS = ('compare_contingent_plans', 'submit_contingent_plan')


def test_arm_defaults_to_on_and_rejects_anything_else(monkeypatch):
    import contingent_controller as cc
    monkeypatch.delenv(cc.ARM_VARIABLE, raising=False)
    assert cc.arm() == 'on' and cc.enabled()
    monkeypatch.setenv(cc.ARM_VARIABLE, 'OFF')
    assert cc.arm() == 'off' and not cc.enabled()
    monkeypatch.setenv(cc.ARM_VARIABLE, 'a')
    with pytest.raises(ValueError):
        cc.arm()


@pytest.mark.parametrize('arm', ['off', 'on'])
def test_actual_worker_prompt_carries_the_interface_only_in_the_on_arm(monkeypatch, tmp_path, arm):
    sys.path[:0] = [str(ROOT / 'deployment/kaggle/tests'), str(ROOT / 'deployment/kaggle')]
    import test_my_agent, test_nooa_join, test_join_qualification as qual
    spec = importlib.util.spec_from_file_location('runtime.local_solver', HERE / 'shared_local_solver.py')
    runtime = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'runtime.local_solver', runtime)
    spec.loader.exec_module(runtime)
    monkeypatch.setattr(test_my_agent, 'AGENT_PATH', HERE / 'adapter.py')
    monkeypatch.setattr(test_nooa_join, 'CANDIDATE', HERE)
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(body)
            if len(seen) > 2:
                self.send_error(500, 'fixture call cap exceeded')
                return
            # Both arms attempt the contingent call. Off must refuse it and still
            # produce an ordinary action, so the arms stay budget-comparable.
            code = ('self.submit_contingent_plan(' + repr(relevant_proposal()) + ')' if len(seen) == 1
                    else 'self.submit_actions(["LEFT"], rationale="predict: fixture continuation")')
            reply = {'id': f'fixture-{len(seen)}', 'object': 'chat.completion', 'created': 1, 'model': 'fixture',
                     'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {'role': 'assistant', 'content': None,
                         'tool_calls': [{'id': f'c{len(seen)}', 'type': 'function', 'function': {
                             'name': 'execute_python', 'arguments': json.dumps({'code': code})}}]}}],
                     'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}}
            raw = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    settings = {'ARC_DISABLE_DOTENV': '1', 'ARC_LLM_MODEL': 'qwen/qwen3.8-27b', 'ARC_LLM_API_KEY': 'fixture',
        'ARC_LLM_BASE_URL': f'http://127.0.0.1:{server.server_port}/v1',
        'ARC_LLM_EXTRA_BODY': '{"reasoning":{"enabled":false}}', 'ARC_LLM_MAX_TOKENS': '1024',
        'ARC_SOLVER_ARM': 'loop', 'ARC_PROCEDURAL_DIAGNOSTIC': '0', 'ARC_DISCOVERY_ARM': 'a',
        'ARC_CONTINGENT_ARM': arm,
        'NOOA_JOIN_ROOT': str(HERE), 'NOOA_AGENT_COMMAND': json.dumps([sys.executable, str(HERE / 'headless.py'), '--run-dir', '.']),
        'TYCHO_NOOA_WORKER_TIMEOUT_SECONDS': '30', 'LITELLM_LOCAL_MODEL_COST_MAP': 'True',
        'TYCHO_KAGGLE_MAX_ACTIONS': '2', 'ARC_STALL_REVIEW': '0'}
    for k, v in settings.items():
        monkeypatch.setenv(k, v)
    frame = lambda grid: test_my_agent.frame(grid, actions=tuple(getattr(test_my_agent.GameAction, f'ACTION{i}') for i in range(1, 5)))
    first = frame([[0, 0], [0, 0]])
    agent = qual.build_agent(monkeypatch, tmp_path, first)
    try:
        if arm == 'on':
            # Root UP, then the dispatcher takes the matching LEFT continuation.
            assert agent.choose_action(agent.frames, first).name == 'ACTION1'
            agent.action_counter += 1
            agent.frames.append(frame([[1, 0], [0, 0]]))
            assert agent.choose_action(agent.frames, agent.frames[-1]).name == 'ACTION3'
        else:
            # The refused plan submits nothing, so the same decision costs a
            # second crossing and yields the ordinary LEFT instead.
            assert agent.choose_action(agent.frames, first).name == 'ACTION3'
    finally:
        try:
            agent.cleanup(scorecard=None)
        finally:
            server.shutdown()
            thread.join(timeout=2)

    prompt = json.dumps(seen[0])
    (tmp_path / 'first_request.json').write_text(json.dumps(seen[0], indent=2) + '\n')
    events = [json.loads(l) for p in tmp_path.rglob('contingent_events.jsonl') for l in p.read_text().splitlines()]
    if arm == 'on':
        # Tool surface, prompt instructions and the event log are all present.
        assert all(tool in prompt for tool in TOOLS)
        assert 'observation-contingent choices' in prompt
        assert events and events[0]['status'] == 'proposed'
        assert [e['status'] for e in events][-1] == 'continuation_submitted'
        assert len(seen) == 1  # two actions, one crossing
    else:
        # No tool, no instruction, no advertised packet key, no event, and the
        # refused call cost a second model crossing rather than an action.
        assert not any(tool in prompt for tool in TOOLS)
        assert 'observation-contingent choices' not in prompt
        assert 'contingent_controller' not in prompt
        assert not events
        assert len(seen) == 2
    assert not list(tmp_path.rglob('ABORT'))
    (tmp_path / 'RESULT.json').write_text(json.dumps({'arm': arm, 'model_calls': len(seen),
        'contingent_tools_in_prompt': [t for t in TOOLS if t in prompt], 'events': events,
        'claim': 'treatment isolation under a scripted model; no Qwen quality evidence'}, indent=2) + '\n')


@pytest.mark.parametrize('arm', ['off', 'on'])
def test_direct_calls_are_refused_in_the_off_arm(monkeypatch, arm):
    import contingent_controller as cc
    monkeypatch.setenv(cc.ARM_VARIABLE, arm)
    if arm == 'on':
        assert cc.compile_plan(relevant_proposal())['selected']['id'] == 'probe'
        return
    # The tools leave the documented surface, but a copied cell must still fail
    # closed rather than run the treatment inside the control arm.
    sys.path.insert(0, str(HERE))
    from solver_agent import MdArcSolverAgent
    with pytest.raises(RuntimeError, match='off in this arm'):
        MdArcSolverAgent.compare_contingent_plans(object(), relevant_proposal())
    assert MdArcSolverAgent.submit_contingent_plan(object(), relevant_proposal()).startswith('REJECTED')
