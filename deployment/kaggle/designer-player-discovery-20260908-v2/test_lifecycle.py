"""No paid calls: reproduce the real timeout and truncated-response boundaries."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from probe_lifecycle import AdmissionWindow, shutdown_bounded


def test_deadline_closes_admission_even_without_signal_and_cannot_be_renewed():
    now = [10.0]
    gate = AdmissionWindow(2, clock=lambda: now[0])
    with gate.admit():
        assert gate.active == 1
    now[0] = 12.0
    with pytest.raises(PermissionError, match='deadline'):
        with gate.admit():
            pytest.fail('expired admission')
    now[0] = 10.0
    with pytest.raises(PermissionError):
        with gate.admit():
            pytest.fail('closed admission renewed')


def test_closing_does_not_wait_for_provider_and_completion_remains_owned():
    gate = AdmissionWindow(60)
    entered, release, completed = (threading.Event() for _ in range(3))

    def provider():
        with gate.admit():
            entered.set()
            assert release.wait(5)
            completed.set()

    thread = threading.Thread(target=provider)
    thread.start()
    try:
        assert entered.wait(2)
        gate.close('cleanup')
        assert not gate.wait_idle(0)
        with pytest.raises(PermissionError, match='cleanup'):
            with gate.admit():
                pytest.fail('cleanup admitted replacement')
    finally:
        release.set()
        thread.join(2)
    assert completed.is_set() and gate.wait_idle(0)


def test_server_shutdown_cannot_undo_cleanup_bound():
    entered, release, closed = (threading.Event() for _ in range(3))

    class Server:
        def shutdown(self):
            entered.set()
            assert release.wait(3)

        def server_close(self):
            closed.set()

    try:
        assert not shutdown_bounded(Server(), .05)
        assert entered.is_set() and not closed.is_set()
    finally:
        release.set()
    assert closed.wait(2)


def test_cleanup_error_preserves_timeout_and_stops_subsequent_admission(monkeypatch):
    spec = importlib.util.spec_from_file_location('cleanup_failure_runtime', HERE/'shared_local_solver.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    solver = object.__new__(runtime.LocalSolver)
    solver._require_active = lambda: None
    solver._validate_turn_continuity = lambda _: None
    solver._expected_previous_decision = lambda: None
    solver._pending = None
    solver._next_sequence = 0
    solver._current_observation = None
    solver.timeout_seconds = 10
    monkeypatch.setattr(runtime, 'TurnMessage', lambda **kw: kw)
    sent = []
    solver._send_model = sent.append
    original = TimeoutError('original deadline')

    def receive(*_, **__):
        raise original

    def failed_close(**_):
        raise OSError('controlled retirement failure')

    solver._receive = receive
    solver.close = failed_close
    turn = SimpleNamespace(sequence=0, action_limit=10, actions_used=0,
        actions_remaining=10, final_permitted_action=False, level=0, turn_in_level=0)
    with pytest.raises(TimeoutError) as caught:
        solver.decide(turn)
    assert caught.value is original
    assert 'retirement failure' in original.__notes__[0]
    with pytest.raises(runtime.LocalSolverError, match='retirement unresolved'):
        solver.decide(turn)
    assert len(sent) == 1


@pytest.mark.parametrize('failure', ['length', 'outer_signal'])
def test_real_worker_failure_retires_before_cleanup(monkeypatch, tmp_path, failure):
    root = ROOT
    sys.path[:0] = [str(root/'deployment/kaggle/tests'), str(root/'deployment/kaggle')]
    import test_my_agent, test_nooa_join, test_join_qualification as qual
    spec = importlib.util.spec_from_file_location('runtime.local_solver', HERE/'shared_local_solver.py')
    runtime = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'runtime.local_solver', runtime)
    spec.loader.exec_module(runtime)
    monkeypatch.setattr(test_my_agent, 'AGENT_PATH', HERE/'adapter.py')
    monkeypatch.setattr(test_nooa_join, 'CANDIDATE', HERE)
    admitted, release, completed = (threading.Event() for _ in range(3))
    gate = AdmissionWindow(60)
    seen = []
    cutoff = TimeoutError('controlled outer deadline')

    def interrupt(*_):
        gate.close('outer deadline')
        raise cutoff

    old_handler = signal.signal(signal.SIGALRM, interrupt)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                with gate.admit():
                    seen.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                    admitted.set()
                    if failure == 'outer_signal':
                        # Signal only after actual HTTP admission, never by sleep.
                        os.kill(os.getpid(), signal.SIGALRM)
                        assert release.wait(15)
                    message = {'role':'assistant', 'content':None, 'reasoning_content':'Synthetic fixture.'}
                    if failure == 'outer_signal':
                        message['tool_calls'] = [{'id':'late','type':'function','function':{
                            'name':'execute_python', 'arguments':json.dumps({'code':'self.submit_actions(["UP"], rationale="predict: move")'})}}]
                    payload = {'id':'fixture','object':'chat.completion','created':1,'model':'qwen/qwen3.8-27b',
                        'choices':[{'index':0,'finish_reason':'length' if failure=='length' else 'tool_calls','message':message}],
                        'usage':{'prompt_tokens':100,'completion_tokens':16384,'total_tokens':16484,
                                 'completion_tokens_details':{'reasoning_tokens':9721}}}
                    raw = json.dumps(payload).encode()
                    try:
                        self.send_response(200); self.send_header('Content-Type','application/json')
                        self.end_headers(); self.wfile.write(raw)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    completed.set()
            except PermissionError:
                self.send_error(429)

    server = HTTPServer(('127.0.0.1',0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    settings = {'ARC_DISABLE_DOTENV':'1','ARC_LLM_MODEL':'qwen/qwen3.8-27b','ARC_LLM_API_KEY':'fixture',
        'ARC_LLM_BASE_URL':f'http://127.0.0.1:{server.server_port}/v1',
        'ARC_LLM_EXTRA_BODY':'{"reasoning":{"enabled":false}}','ARC_LLM_MAX_TOKENS':'16384',
        'ARC_SOLVER_ARM':'mindset','NOOA_JOIN_ROOT':str(HERE),
        'NOOA_AGENT_COMMAND':json.dumps([sys.executable,str(HERE/'headless.py'),'--run-dir','.']),
        'TYCHO_NOOA_WORKER_TIMEOUT_SECONDS':'10','LITELLM_LOCAL_MODEL_COST_MAP':'True'}
    for k,v in settings.items():
        monkeypatch.setenv(k,v)
    first = test_my_agent.frame([[0,1],[0,0]])
    agent = qual.build_agent(monkeypatch,tmp_path,first)
    try:
        expected = TimeoutError if failure=='outer_signal' else runtime.LocalSolverError
        with pytest.raises(expected) as caught:
            agent.choose_action(agent.frames,first)
        if failure=='outer_signal':
            assert caught.value is cutoff
        else:
            assert 'agent exited' in str(caught.value)
        assert admitted.is_set() and len(seen)==1
        assert agent._solver._closed
        assert agent._solver._process.poll() is not None
        pid = agent._solver.worker_metadata['agent_pid']
        proc = Path(f'/proc/{pid}/stat')
        def child_stopped():
            try:
                return proc.read_text().rsplit(')',1)[1].split()[0] == 'Z'
            except FileNotFoundError:
                return True
        deadline = time.monotonic() + 2
        while not child_stopped() and time.monotonic() < deadline:
            threading.Event().wait(.01)
        assert child_stopped(), 'Actor child remains executable after protocol worker exit'
        # Actual adapter cleanup cannot queue finish behind the dead turn.
        with pytest.raises(runtime.LocalSolverError, match='closed'):
            agent.cleanup(scorecard=None)
        assert all(not p.read_text().strip() for p in tmp_path.rglob('actions.jsonl'))
    finally:
        gate.close('test cleanup')
        if agent._solver is not None:
            agent._solver.close(abort=True)
        release.set()
        assert gate.wait_idle(3)
        server.shutdown(); thread.join(2)
        signal.signal(signal.SIGALRM, old_handler)
    assert completed.is_set()
    assert len(seen)==1
    assert all(not p.read_text().strip() for p in tmp_path.rglob('actions.jsonl'))
