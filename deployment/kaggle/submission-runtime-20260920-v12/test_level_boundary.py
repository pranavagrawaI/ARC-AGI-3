"""A level completes under the real worker path and play continues.

No tycho-nooa arm has ever completed a level live, so until now the level
boundary — host terminal evidence, the scribe consolidation gate in
``submit_actions``, the worker's turn accounting across ``levels_completed``
0→1 — had only ever run against in-process fakes. This drives the official
``MyAgent`` adapter → ``LocalSolver`` → ``nooa_worker`` → ``headless`` chain
with a scripted OpenAI-compatible model that behaves like a model would at the
boundary: it submits, gets the scribe rejection, reads the terminal evidence,
writes the insight note, then submits again.
"""
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

CONSOLIDATE = (
    "t = self.level_terminal(0)\n"
    "assert t['winning_action'] == 'UP', t\n"
    "assert t['level'] == 0 and 'pre_win_grid_rows' in t and 'terminal_grid_rows' in t, t\n"
    "self.write_notes('level_0_insights.md', 'Level 0: UP from the start state completed the level.')\n"
    "print(self.submit_actions(['UP'], rationale='predict: same opening on the new level'))"
)
SUBMIT = "print(self.submit_actions(['UP'], rationale='predict: a one-action exploratory move'))"


def _completion(code=None):
    message = {'role': 'assistant', 'content': None, 'tool_calls': [{
        'id': 'call-%d' % (abs(hash(code)) % 10_000), 'type': 'function',
        'function': {'name': 'execute_python', 'arguments': json.dumps({'code': code})}}]}
    if code is None:  # the critic wants prose, not a cell
        message = {'role': 'assistant', 'content': 'No question.'}
    return json.dumps({
        'id': 'level-fixture', 'object': 'chat.completion', 'created': 1, 'model': 'qwen/qwen3.8-27b',
        'choices': [{'index': 0, 'finish_reason': 'tool_calls' if code else 'stop', 'message': message}],
        'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150},
    }).encode()


def _user_texts(body):
    out = []
    for message in body.get('messages', []):
        if message.get('role') == 'user':
            content = message.get('content', '')
            out.append(content if isinstance(content, str) else json.dumps(content))
    return out


def _actor_reply(body):
    """What a model would do with this request: consolidate once after the
    scribe rejection, otherwise submit. The critic gets a prose answer."""
    texts = _user_texts(body)
    if texts and texts[-1].lstrip().startswith('{"review_trigger"'):
        return None
    if any('you just COMPLETED level 0' in t for t in texts) and not any('wrote level_0_insights.md' in t for t in texts):
        return CONSOLIDATE
    return SUBMIT


def test_level_completion_passes_the_scribe_gate_and_play_continues(monkeypatch, tmp_path):
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
            raw = _completion(_actor_reply(body))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    settings = {
        'ARC_DISABLE_DOTENV': '1', 'ARC_LLM_MODEL': 'qwen/qwen3.8-27b', 'ARC_LLM_API_KEY': 'fixture',
        'ARC_LLM_BASE_URL': f'http://127.0.0.1:{server.server_port}/v1',
        'ARC_LLM_EXTRA_BODY': json.dumps({'reasoning': {'enabled': False}}),
        'ARC_LLM_MAX_TOKENS': '16384', 'ARC_SOLVER_ARM': 'loop', 'NOOA_JOIN_ROOT': str(HERE),
        'NOOA_AGENT_COMMAND': json.dumps([sys.executable, str(HERE / 'headless.py'), '--run-dir', '.']),
        'TYCHO_NOOA_WORKER_TIMEOUT_SECONDS': '60', 'LITELLM_LOCAL_MODEL_COST_MAP': 'True',
        'TYCHO_KAGGLE_MAX_ACTIONS': '3',
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)

    start = test_my_agent.frame([[0, 1], [0, 0]])
    won = test_my_agent.frame([[1, 0], [0, 0]], levels_completed=1)          # the UP completed level 0
    next_level = test_my_agent.frame([[0, 0], [1, 0]], levels_completed=1)   # the new level's successor
    agent = qual.build_agent(monkeypatch, tmp_path, start)
    try:
        emitted = qual.drive(agent, [start, won, next_level])
    finally:
        agent.cleanup(scorecard=None)
        server.shutdown()
        thread.join(timeout=2)

    assert [a.name for a in emitted] == ['ACTION1', 'ACTION1', 'ACTION1'], emitted
    assert not list(tmp_path.rglob('ABORT'))
    workspaces = [p for p in (tmp_path / 'workspaces').iterdir() if p.is_dir()]
    assert len(workspaces) == 1, 'a level boundary must not open a new solver session'
    ipc = workspaces[0] / 'ipc'

    transitions = [json.loads(s) for s in (ipc / 'transitions.jsonl').read_text().splitlines()]
    assert [t['after']['levels_completed'] for t in transitions][:2] == [1, 1]
    terminal = json.loads((ipc / 'level_0_terminal.json').read_text())
    assert terminal['winning_action'] == 'UP' and terminal['level'] == 0

    rejections = [json.loads(s) for s in (ipc / 'rejections.jsonl').read_text().splitlines()]
    assert [r['reason'] for r in rejections] == ['scribe'], rejections
    notes = list(workspaces[0].rglob('level_0_insights.md'))
    assert notes and 'completed the level' in notes[0].read_text()

    # Exactly one extra Actor call was spent at the boundary (submit, rejected,
    # consolidate+submit); the worker kept its turn accounting.
    actor = [b for b in seen if _actor_reply(b) is not None or 'you just COMPLETED' in ''.join(_user_texts(b))]
    assert len([b for b in seen if not _user_texts(b)[-1].lstrip().startswith('{"review_trigger"')]) == 4, \
        [_user_texts(b)[-1][:60] for b in seen]
    assert not any('WorkerError' in p.read_text() for p in workspaces[0].glob('worker-stderr.log'))
