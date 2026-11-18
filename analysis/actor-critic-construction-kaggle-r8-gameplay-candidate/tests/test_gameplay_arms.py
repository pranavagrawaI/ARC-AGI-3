"""r8 paired-arm contract: arm equality, purse caps, budget arithmetic, baseline-activation guard."""
from __future__ import annotations
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
from test_runtime import CANDIDATE, HERE, LOCAL_MODEL, PACKAGE, STARTUP, WORK, load, materialize


@pytest.fixture
def play(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE))
    for name in ('game_preflight', 'local_wire', 'local_accounting'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return load(PACKAGE / 'gameplay_play.py', 'r8_gameplay_play')


def test_frozen_parameters_match_card(play):
    assert play.GAMES == ('ls20',) and play.ARMS == ('A', 'B')
    assert play.ADMISSION_SECONDS == 600 and play.MAX_ACTIONS == 200
    assert play.LLM_TIMEOUT_SECONDS == 90 and play.ACTOR_MAX_TOKENS == 16384 and play.CRITIC_MAX_TOKENS == 192
    assert play.ARM_CALL_CAP == {'A': 129, 'B': 129} and play.TOTAL_CALL_CAP == 258
    assert len(set(play.ARM_CALL_CAP.values())) == 1, 'arm purses must be symmetric'
    assert play.TOTAL_CALL_CAP == sum(play.ARM_CALL_CAP.values())
    assert play.HTTP_POST_SECONDS == {'actor': 90, 'summarizer': 90, 'critic': 90, 'constructor': 30}
    assert (play.CONSTRUCTOR_WALL, play.CONSTRUCTOR_HARD, play.CONSTRUCTOR_MAX_REQUESTS) == (45, 50, 2)
    assert '--launcher-timeout-s\', \'1800\'' in (PACKAGE / 'donor_bootstrap.py').read_text()


def test_arm_diff_accepts_only_worker_command_and_per_arm_roots(play, tmp_path):
    relay, python = 'http://127.0.0.1:1/v1', sys.executable
    base = play.base_environment(relay, python)
    envs = {arm: play.arm_environment(arm, base, tmp_path / f'arm-{arm}', relay, python) for arm in play.ARMS}
    diff = play.arm_diff(envs, relay, python)
    assert diff['equal'] is True and diff['forbidden_differences'] == []
    assert set(diff['differing_keys']) == {'NOOA_AGENT_COMMAND', 'ARC_CONSTRUCTION_ARM', 'TYCHO_KAGGLE_WORKSPACE_ROOT', 'RECORDINGS_DIR'}
    assert diff['worker_command']['A'] == [python, str(PACKAGE / 'nooa-join' / 'headless.py'), '--run-dir', '.']
    assert diff['worker_command']['B'][:2] == [python, str(PACKAGE / 'headless.py')]
    assert '--constructor-transport' in diff['worker_command']['B'] and '--constructor-transport' not in diff['worker_command']['A']
    assert envs['A']['ARC_CONSTRUCTION_ARM'] == '0' and envs['B']['ARC_CONSTRUCTION_ARM'] == '1'
    # Any hidden difference is refused: a budget, a prompt switch, a model binding.
    for key, value in (('TYCHO_KAGGLE_MAX_ACTIONS', '201'), ('ARC_STALL_REVIEW', '1'), ('ARC_LLM_MAX_TOKENS', '4096'), ('ARC_LLM_MODEL', 'other')):
        hidden = {arm: dict(env) for arm, env in envs.items()}
        hidden['B'][key] = value
        tampered = play.arm_diff(hidden, relay, python)
        assert tampered['equal'] is False and tampered['forbidden_differences'] == [key]


def test_admission_budget_requires_full_window(play):
    assert play.admission_budget(1740, 2, 90) == 600.0
    assert play.admission_budget(990, 1, 90) == 600.0
    with pytest.raises(RuntimeError, match='cannot fund'):
        play.admission_budget(1400, 2, 90)


def test_expected_requests_enforces_the_whole_run_purse(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE))
    monkeypatch.delitem(sys.modules, 'local_wire', raising=False)
    accounting = load(PACKAGE / 'local_accounting.py', 'r8_local_accounting')
    assert accounting.MAX_ADMITTED == 258

    def result(n, constructor=2, stop='step_budget', **overrides):
        roles = {'actor': n - constructor, 'summarizer': 0, 'constructor': constructor, 'critic': 0}
        record = {'schema': 'local-model-request-accounting-v1', 'model': LOCAL_MODEL, 'identity_status': 'ESTABLISHED',
                  'admitted': n, 'receipted': n, 'unreceipted_usage': 'none', 'roles': roles, 'complete': True,
                  'receipts': [{'call': i, 'purpose': 'actor'} for i in range(1, n + 1)]}
        base = {'status': 'passed', 'local_requests': record, 'games': [{'arm': 'A', 'stop_reason': stop}, {'arm': 'B', 'stop_reason': stop}]}
        base.update(overrides)
        return base

    assert accounting.expected_requests(result(258)) == 258
    assert accounting.expected_requests(result(3, constructor=0, stop='model_admission_budget')) == 3
    for bad in (result(259), result(0), result(10, constructor=5), result(10, stop='exception'),
                result(10, games=[]), result(10, error='boom'), result(10, fatal=True), result(10, status='failed')):
        with pytest.raises(ValueError):
            accounting.expected_requests(bad)


class _Fixture(BaseHTTPRequestHandler):
    """Scripted loopback upstream: three actions per arm, constructor handled."""
    ctor_calls = 0
    actor_calls: dict[str, int] = {}
    received: list = []

    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        type(self).received.append(body)
        names = {t['function']['name'] for t in body.get('tools') or []}
        arm = 'B' if any('request_construction' in json.dumps(m) for m in body['messages']) else 'A'
        if 'submit_candidate' in names:
            type(self).ctor_calls += 1
            if type(self).ctor_calls % 2 == 1:
                fn, args = 'execute_python', {'code': 'print(1)'}
            else:
                fn, args = 'submit_candidate', {'work_item': WORK, 'candidate_source': CANDIDATE}
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'ctor:{type(self).ctor_calls}', 'type': 'function', 'function': {'name': fn, 'arguments': json.dumps(args)}}]}
        elif body.get('max_tokens') == 192:
            message = {'role': 'assistant', 'content': '{"claim_ref":"","question":"","reason":""}'}
        else:
            n = type(self).actor_calls.get(arm, 0) + 1
            type(self).actor_calls[arm] = n
            codes = ["self.submit_actions(['UP'], rationale='r')", "print(self.submit_actions(['RIGHT'], rationale='r'))", "print(self.submit_actions(['DOWN'], rationale='r'))"]
            code = codes[n - 1] if n <= len(codes) else 'print(self.observations())'
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'actor:{arm}:{n}', 'type': 'function', 'function': {'name': 'execute_python', 'arguments': json.dumps({'code': code})}}]}
        response = {'id': 'fx', 'object': 'chat.completion', 'created': 1, 'model': LOCAL_MODEL,
                    'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop'}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13, 'completion_tokens_details': {'reasoning_tokens': 0}}}
        raw = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _run_driver(tmp_path, extra_wrapper_lines):
    ns, _ = materialize(tmp_path)
    package = ns['root']
    _Fixture.ctor_calls, _Fixture.actor_calls, _Fixture.received = 0, {}, []
    server = HTTPServer(('127.0.0.1', 0), _Fixture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    (tmp_path / 'working').mkdir()
    wrapper = tmp_path / 'wrapper.py'
    wrapper.write_text('\n'.join([
        'import sys, json',
        'sys.dont_write_bytecode = True',
        'from pathlib import Path',
        f'sys.path.insert(0, {str(package)!r})',
        'import gameplay_play',
        "gameplay_play.inspect_gpu = lambda out: None",
        'gameplay_play.ADMISSION_SECONDS = 30',
        'gameplay_play.MAX_ACTIONS = 3',
        *extra_wrapper_lines,
        f'sys.path.insert(0, {str(HERE / "tests")!r})',
        'from game_mount import enter',
        f'enter(Path({str(tmp_path / "mount")!r}))',
        'sys.dont_write_bytecode = True',
        'raise SystemExit(gameplay_play.main())',
        '',
    ]))
    out = tmp_path / 'ws'
    argv = [sys.executable, '-I', '-B', str(wrapper), '--base-url', f'http://127.0.0.1:{server.server_port}', '--model', LOCAL_MODEL,
            '--workspace-root', str(out), '--output', str(tmp_path / 'result.json'), '--timeout-s', '640',
            '--startup-argv-json', json.dumps(STARTUP), '--working', str(tmp_path / 'working')]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
    return proc, json.loads((out / 'raw' / 'RESULTS.json').read_text()), out / 'raw'


def test_construction_activates_only_in_arm_b(tmp_path):
    proc, result, raw = _run_driver(tmp_path, [])
    assert proc.returncode == 0, proc.stderr[-2000:]
    a, b = result['games']
    assert result['status'] == 'passed' and [g['arm'] for g in (a, b)] == ['A', 'B']
    assert a['construction']['frozen'] is False and a['roles']['constructor'] == 0
    assert b['construction']['frozen'] is True and b['construction']['queried_frozen_predict'] is True and b['roles']['constructor'] == 2
    assert not list((raw.parent / 'arm-A').rglob('FROZEN.json')) and list((raw.parent / 'arm-B').rglob('FROZEN.json'))
    assert json.loads((raw / 'ARM-DIFF.json').read_text())['equal'] is True


def test_baseline_arm_carrying_the_constructor_is_refused(tmp_path):
    """Discriminating control: give arm A the construction worker; the relay must refuse it."""
    proc, result, raw = _run_driver(tmp_path, [
        'original = gameplay_play.worker_command',
        "gameplay_play.worker_command = lambda arm, relay, python: original('B', relay, python)",
    ])
    assert proc.returncode != 0
    assert result['status'] == 'failed'
    assert 'constructor request in baseline arm' in (result['error'] or '') or any(
        'constructor request in baseline arm' in line for line in (raw / 'local-rejections.jsonl').read_text().splitlines())
    assert result['fatal'] is True
    assert not result['games'], 'the baseline arm must not be reported as a completed game'
