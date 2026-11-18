"""Generated notebook + actual production acceptance, with explicit unavailable-context fixtures.
No target GPU, model, provider, or remote launch claim. HTTP is loopback only.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
PLAY_GAME = os.environ.get('R9_PACKAGE_GAME', 's5i5')  # the suite runs once per built package
assert PLAY_GAME in ('s5i5', 'cd82')
PACKAGE = HERE / 'r10a-package'
GAME_IDS = {'ls20': 'ls20-9607627b', 's5i5': 's5i5-18d95033', 'cd82': 'cd82-fb555c5d'}
GAME_ID = GAME_IDS[PLAY_GAME]
# Game-aware fixture Actor scripts: s5i5 exposes only CLICK; cd82 and ls20 expose directions.
FIXTURE_ACTIONS = {'s5i5': ['CLICK 32 32', 'CLICK 10 10', 'CLICK 40 40'], 'cd82': ['UP', 'CLICK 31 41', 'DOWN'], 'ls20': ['UP', 'RIGHT', 'DOWN']}[PLAY_GAME]
LOCAL_MODEL = '/kaggle/input/datasets/jakobbrggen/qwen3-8-27b-fp8-hf-snapshot'
STARTUP = ['python', '-m', 'sglang.launch_server', '--model-path', LOCAL_MODEL]
CANDIDATE = '''
def initialize(observation): return {"grid": observation["grid"]}
def predict(state, action): return [{"state": {"grid": state["grid"], "last_action": action}, "commitments": {"marker": "join-fixture-alpha", "action": action}, "declared_dependencies": ["grid:observation"]}]
def assimilate(predicted_state, observation, verdict): return {"grid": observation["grid"], "verdict": verdict, "last_action": predicted_state.get("last_action")}
'''
WORK = {'decision': 'next move', 'obstacle': 'transition uncertainty', 'operation': 'test', 'claim': 'grid successor', 'evidence_refs': [1], 'dependencies': ['grid:observation'], 'completion_checks': [1], 'remaining_scope': 'unseen actions', 'reasoning_need': 'needs_computation'}


def materialize(tmp_path):
    nb = json.loads((PACKAGE / 'construction.ipynb').read_text())
    ns = {}
    source = ''.join(nb['cells'][1]['source']).replace("Path('/kaggle/working')", f'Path({str(tmp_path)!r})')
    exec(compile(source, 'generated-notebook-cell-1', 'exec'), ns)
    return ns, nb


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generated_notebook_clean_existing_and_cache_conflicts(tmp_path, monkeypatch):
    ns, nb = materialize(tmp_path)
    root = ns['root']
    again, _ = materialize(tmp_path)
    assert again['root'] == root
    expected = json.loads((root / 'ASSET-HASHES.json').read_text())
    assert not any(n.endswith(('case.json', '.zip')) or 'remote_entry' in n for n in expected)
    assert all(hashlib.sha256((root / n).read_bytes()).hexdigest() == h for n, h in expected.items())
    # Actual default notebook branch in the exact scrubbed fresh process. Missing GPU must
    # fail durably, not invoke cached success functions or continue to installation/models.
    for name in ('accelerator', 'audit', 'donor_bootstrap', 'containment_preflight'):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(discover=lambda *_: {'status': 'passed'}, main=lambda: 0))
    for name in ('CONSTRUCTION_FIXTURE', 'CONSTRUCTION_LOCAL_BASE_URL', 'KAGGLE_PROBE_SKIP_GAME'):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(subprocess.CalledProcessError):
        exec(compile(''.join(nb['cells'][2]['source']), 'generated-notebook-cell-2', 'exec'), ns)
    failure = json.loads((root / 'BOOTSTRAP-FAILURE.json').read_text())
    assert failure['status'] == 'failed'
    assert not (root / 'SEAL.json').exists(), 'local missing GPU should fail before containment'
    assert json.loads((root / 'PROCESS-GUARD.json').read_text())['cleanup'] == 'retired'
    (root / 'local_wire.py').write_text('drift')
    with pytest.raises(RuntimeError, match='existing asset differs'):
        materialize(tmp_path)


@pytest.mark.parametrize('failure,revise', [(None, False), (None, True), ('missing_usage', False), ('wrong_model', False), ('second_http', False), ('call_cap', False)])
def test_actual_supervisor_worker_http(tmp_path, monkeypatch, failure, revise):
    """Real official Agent.main loop, both arms, through the unchanged supervisor branch.

    Fixture caps (admission 30 s, 3 actions, 13/13 calls per arm) replace the frozen
    600 s / 200 / 129-129 so two arms finish in seconds; everything else is the packaged
    driver, adapter, worker, relay and public game.
    """
    ns, _ = materialize(tmp_path)
    package = ns['root']
    monkeypatch.syspath_prepend(str(package))
    for name in ('local_wire', 'local_accounting'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    received, ctor_calls = [], 0
    actor_calls = {'A': 0, 'B': 0}
    first, second, third = FIXTURE_ACTIONS
    if revise:
        second = {'s5i5': 'CLICK 20 20', 'cd82': 'LEFT', 'ls20': 'LEFT'}[PLAY_GAME]
    codes = ['print(self.observations())'] * 2 + [f"self.submit_actions([{first!r}], rationale='fixture initial probe')"]
    codes += [f"print(self.submit_actions([{second!r}], rationale='actor-selected post-freeze action; host queries then executes'))"]
    codes += [f"print(self.submit_actions([{third!r}], rationale='third action reaches the fixture step budget'))"]
    if failure == 'call_cap':
        codes = ['print(self.observations())'] * 40

    class Fixture(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            nonlocal ctor_calls
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            received.append(body)
            assert body['model'] == LOCAL_MODEL
            assert body['chat_template_kwargs'] == {'enable_thinking': False}
            assert not {'provider', 'parallel_tool_calls', 'reasoning', 'extra_body'} & set(body)
            for m in body['messages']:
                assert not {'reasoning_content', 'reasoning_details', 'refusal'} & set(m)
                for c in m.get('tool_calls') or []:
                    assert 'index' not in c
            if failure == 'second_http' and len(received) == 2:
                self.send_error(503, 'synthetic transport failure')
                return
            names = {t['function']['name'] for t in body.get('tools') or []}
            arm = 'B' if any('request_construction' in json.dumps(m) for m in body['messages']) else 'A'
            if 'request_evidence' in names:
                assert 'only successful ABI replay' in body['messages'][0]['content']
                ctor_calls += 1
                time.sleep(float(os.environ.get("R10_FIXTURE_CONSTRUCTOR_DELAY", "0")))
                if ctor_calls in (1, 3):
                    fn, args = 'execute_python', {'code': 'import json\nx=json.load(open("/evidence/case.json"))\nprint(type(x).__name__)'}
                elif ctor_calls == 2:
                    fn, args = 'submit_candidate', {'work_item': WORK, 'candidate_source': CANDIDATE.replace('\n', r'\n')}
                elif ctor_calls == 4:
                    fn, args = 'submit_candidate', {'work_item': WORK, 'candidate_source': 'def broken(:'}
                elif ctor_calls == 5:
                    fn, args = 'submit_candidate', {'work_item': WORK, 'candidate_source': ['def initialize(observation): return 1', 'def predict(s,a): return []', 'def assimilate(s,o,v): return {}']}
                else:
                    fn, args = 'submit_candidate', {'work_item': WORK, 'candidate_source': CANDIDATE.splitlines()}
                names = {t['function']['name'] for t in body['tools']}
                assert fn in names
                if ctor_calls == 2:
                    feedback = [json.loads(m['content']) for m in body['messages'] if m['role'] == 'tool']
                    assert feedback[-1]['status'] == 'completed' and 'dict' in feedback[-1]['stdout']
                if ctor_calls == 3:
                    feedback = [json.loads(m['content']) for m in body['messages'] if m['role'] == 'tool']
                    assert 'serialization' in feedback[-1]['error']

                calls = [{'id': f'opaque/ctor:{ctor_calls}', 'index': 0, 'type': 'function', 'function': {'name': fn, 'arguments': json.dumps(args)}}]
                message = {'role': 'assistant', 'content': None, 'tool_calls': calls}
            elif body.get('max_tokens') == 192:
                message = {'role': 'assistant', 'content': '{"claim_ref":"","question":"","reason":""}'}
            else:
                actor_calls[arm] += 1
                n = actor_calls[arm]
                code = codes[n - 1] if n <= len(codes) else 'print(self.observations())'
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'opaque/actor:{arm}:{n}', 'index': 0, 'type': 'function', 'function': {'name': 'execute_python', 'arguments': json.dumps({'code': code})}}]}
            message.update(reasoning_content=None, refusal=None)
            response = {'id': f'local-fixture-{len(received)}', 'object': 'chat.completion', 'created': 1,
                'model': LOCAL_MODEL, 'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop'}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 30, 'total_tokens': 130, 'completion_tokens_details': {'reasoning_tokens': 0}}}
            if failure == 'missing_usage':
                response.pop('usage')
            if failure == 'wrong_model':
                response['model'] = 'wrong-local-model'
            raw = json.dumps(response).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = HTTPServer(('127.0.0.1', 0), Fixture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Only unavailable hardware/mount context and the small fixture caps are supplied here.
    # The generated driver, containment, adapter, worker, relay and official loop are unchanged.
    wrapper = tmp_path / 'hardware_fixture.py'
    wrapper.write_text('\n'.join([
        'import sys, json, importlib.util, socket',
        'sys.dont_write_bytecode = True',
        'from pathlib import Path',
        f'sys.path.insert(0, {str(package)!r})',
        'def network(event, args):',
        "    if event == 'socket.connect' and args[1][0] not in ('127.0.0.1', '::1'):",
        "        raise RuntimeError('fixture forbids non-loopback network')",
        'sys.addaudithook(network)',
        'import gameplay_play',
        "gameplay_play.inspect_gpu = lambda out: (out / 'GPU-FIXTURE.json').write_text(json.dumps({'evidence_kind': 'synthetic hardware context, NOT GPU observation'}))",
        f'gameplay_play.ADMISSION_SECONDS = {120 if os.environ.get("R10_FIXTURE_CONSTRUCTOR_DELAY") else 30}',
        'gameplay_play.MAX_ACTIONS = 3',
        "gameplay_play.ARM_CALL_CAP = {'A': 13, 'B': 13}",
        f'sys.path.insert(0, {str(HERE / "tests")!r})',
        'from game_mount import enter',
        f'enter(Path({str(tmp_path / "default-game-mount")!r}))',
        "assert '--environments' not in sys.argv",
        'import os',
        "os.environ['OPERATION_MODE'] = 'ONLINE'  # explicit OFFLINE must win",
        f'sys.argv += ["--working", {str(tmp_path / "working")!r}]',
        'raise SystemExit(gameplay_play.main())',
        '',
    ]))
    (tmp_path / 'working').mkdir()
    supervisor = load(package / 'official_launcher_runtime.py', 'packaged_supervisor')
    # Exercise the COMPLETE unchanged production branch, not only a count helper.
    tree = ast.parse((package / 'official_launcher_runtime.py').read_text())
    branches = [n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == 'args.official_launcher_probe' and 'launcher_result_path' in ast.unparse(n)]
    branch = next(n for n in branches if any(isinstance(s, ast.Return) for s in n.body))
    fn = ast.FunctionDef(name='exercise', args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=branch.body, decorator_list=[])
    program = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    output_dir = tmp_path / 'runtime-output'
    output_dir.mkdir()
    fixture_process = SimpleNamespace(poll=lambda: None)  # unavailable model-server liveness only
    warmup = {'chat_completion_accesses': 1, 'successful_chat_completion_accesses': 1}
    globals_ = dict(vars(supervisor), args=SimpleNamespace(launcher_python=sys.executable, official_launcher_probe=wrapper, launcher_timeout_s=900 if os.environ.get("R10_FIXTURE_CONSTRUCTOR_DELAY") else 700),
        process=fixture_process, warmup_facts=warmup, deadline_raw=None, output_dir=output_dir,
        base_url=f'http://127.0.0.1:{server.server_port}', model_name=LOCAL_MODEL, command=STARTUP,
        started_at='explicit-fixture', started=time.monotonic(), event=lambda message: (output_dir / 'fixture-supervisor-event.txt').write_text(message), mode='explicit-fixture', model_verification={'status': 'synthetic-not-model-observation'}, runtime_verification='synthetic-context', config={'claim_ceiling': 'local fixture only'})
    # Counters are explicitly HTTP-fixture observations, never SGLang log claims.
    globals_['wait_for_chat_completion_count'] = lambda *_a, **_kw: {'chat_completion_accesses': 1 + len(received), 'successful_chat_completion_accesses': 1 + len(received), 'successful_freeze_gc_accesses': 1, 'ready_marker_count': 1}
    exec(compile(program, 'actual-generated-supervisor-acceptance', 'exec'), globals_)
    transport_failure = failure in ('missing_usage', 'wrong_model', 'second_http')
    try:
        if transport_failure:
            with pytest.raises(RuntimeError, match='official launcher command failed'):
                globals_['exercise']()
        else:
            result = globals_['exercise']()
            assert result['status'] == 'passed'
            counts = json.loads((output_dir / 'request-count-facts.json').read_text())
            assert counts['expected_launcher_solver_requests'] == len(received)
        raw_dir = output_dir / 'solver-workspaces/raw'
        identity = json.loads((raw_dir / 'GAME-PREFLIGHT.json').read_text())
        assert identity['game_id'] == GAME_ID
        assert identity['environments'] == '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files'
        assert identity['files'][PLAY_GAME + '.py'] == {'s5i5': '18d950333a006e984941ceaa16cdc4dc8dca8ca26cb82194198759089aac5965', 'cd82': 'fb555c5ddf0a3b742188646bb28d2f57cb98cf2d3263c594032b45ec8fce2a86'}[PLAY_GAME]
        assert identity['selected_directory'].endswith({'s5i5': 's5i5/18d95033', 'cd82': 'cd82/fb555c5d'}[PLAY_GAME])
        framework = json.loads((raw_dir / 'FRAMEWORK.json').read_text())
        assert framework['official']['main.py'] == '864254c750bbbd12a211f2d8aa1b1025d0609283f07dea4ede83722f2435301b'
        assert framework['official']['agents/agent.py'] == '85655f7a11a105e50bba76efd00bebf8056d02ddbae827a8f8a5b46e849fc087'
        result = json.loads((raw_dir / 'RESULTS.json').read_text())
        diff = json.loads((raw_dir / 'ARM-DIFF.json').read_text())
        assert diff['equal'] is True and diff['forbidden_differences'] == []
        assert set(diff['differing_keys']) == {'NOOA_AGENT_COMMAND', 'ARC_CONSTRUCTION_ARM', 'TYCHO_KAGGLE_WORKSPACE_ROOT', 'RECORDINGS_DIR'}
        assert '--constructor-transport' in diff['worker_command']['B'] and '--constructor-transport' not in diff['worker_command']['A']
        if transport_failure:
            assert result['status'] == 'failed'
            assert len(received) == (2 if failure == 'second_http' else 1)
            assert not result['local_requests']['complete']
            assert result['local_requests']['known_usage']['prompt_tokens'] == (100 if failure == 'second_http' else 0)
            assert result['local_requests']['unreceipted_usage'] == 'unknown'
        else:
            assert result['status'] == 'passed', result
            games = result['games']
            assert [g['arm'] for g in games] == ['A', 'B']
            progress = [json.loads(line) for line in (raw_dir / 'game-progress.jsonl').read_text().splitlines()]
            assert progress == games
            for g in games:
                assert g['game_id'] == GAME_ID
                assert g['stop_reason'] in {'win', 'step_budget', 'wall_budget', 'model_admission_budget', 'call_cap'}
                assert g['actions'] == g['solver_actions']
            arm_a, arm_b = games
            assert arm_a['construction']['frozen'] is False and arm_a['construction']['requested_construction'] is False
            assert arm_a['roles']['constructor'] == 0
            assert not list((output_dir / 'solver-workspaces/arm-A').rglob('FROZEN.json'))
            assert not list((output_dir / 'solver-workspaces/arm-A').rglob('host-events.jsonl'))
            import copy
            for key, value in [('model', 'wrong'), ('admitted', 999), ('complete', False)]:
                corrupted = copy.deepcopy(result)
                corrupted['local_requests'][key] = value
                with pytest.raises(ValueError):
                    supervisor.expected_launcher_solver_requests(corrupted)
            if failure == 'call_cap':
                assert [g['stop_reason'] for g in games] == ['call_cap', 'call_cap']
                assert [g['actions'] for g in games] == [0, 0]
                assert result['arm_calls'] == {'A': 13, 'B': 13} and result['calls'] == len(received) == 26
            else:
                assert [g['stop_reason'] for g in games] == ['step_budget', 'step_budget']
                assert [g['actions'] for g in games] == [3, 3]
                assert arm_b['construction']['frozen'] is True
                assert arm_b['construction']['requested_construction'] is True
                # Frozen R6 mechanism: the host queries the freeze only when the submitted action string is
                # itself in available_actions. A click game needs 'CLICK x y', which never matches 'CLICK',
                # so on s5i5 the freeze is produced but never queried/assimilated. Recorded, not repaired.
                uses_freeze = PLAY_GAME != 's5i5'
                assert arm_b['construction']['queried_frozen_predict'] is uses_freeze
                assert arm_b['construction']['assimilated_frozen'] is uses_freeze
                assert arm_b['construction']['action_step_digest_bound'] is uses_freeze
                assert arm_b['roles']['constructor'] == 6 and result['local_requests']['roles']['constructor'] == 6
                assert any(m.get('tool_call_id') == 'opaque/ctor:1' for b in received for m in b['messages'])
                assert result['local_requests']['roles']['actor'] == sum(actor_calls.values())
            assert not list(raw_dir.rglob('GPU.json'))
            # Session directories recorded and addressable by exact path (the r8 evidence gap).
            assert result['play_game'] == PLAY_GAME and result['game_id'] == GAME_ID
            for g in games:
                assert g['workspace'] == f"arm-{g['arm']}" and len(g['sessions']) == 1, g
                session = output_dir / 'solver-workspaces' / g['workspace'] / g['sessions'][0]['session_dir']
                assert session.is_dir() and (session / 'ipc').is_dir()
                for name in g['sessions'][0]['ipc_files']:
                    assert (session / 'ipc' / name).exists(), name
                assert 'model_calls.jsonl' in g['sessions'][0]['ipc_files'] and 'actions.jsonl' in g['sessions'][0]['ipc_files']
                for rel in g['sessions'][0]['host_events'] + g['sessions'][0]['constructor_results']:
                    assert (output_dir / 'solver-workspaces' / g['workspace'] / rel).is_file(), rel
            if failure != 'call_cap':
                assert arm_b['sessions'][0]['frozen'] is True and arm_b['sessions'][0]['host_events'], arm_b['sessions']
                assert arm_a['sessions'][0]['frozen'] is False and not arm_a['sessions'][0]['host_events']
            # Request receipts carry digests, not message bodies; full bodies live per call under raw-requests/.
            rows = [json.loads(line) for line in (raw_dir / 'requests.jsonl').read_text().splitlines()]
            assert len(rows) == len(received) and all('messages' not in r['request'] and r['request']['messages_sha256'] and r['request']['model'] == LOCAL_MODEL for r in rows)
            bodies = sorted((raw_dir / 'raw-requests').glob('*.wire.json'))
            assert len(bodies) == len(received)
            import hashlib as _h
            for r, body in zip(rows, bodies):
                wire = json.loads(body.read_text())
                assert _h.sha256(json.dumps(wire['messages'], sort_keys=True).encode()).hexdigest() == r['request']['messages_sha256']
            assert (raw_dir / 'requests.jsonl').stat().st_size < 16 * 1024 * 1024
        assert json.loads((raw_dir / 'SEAL.json').read_text())['status'] == 'passed'
        print(json.dumps({'evidence_kind': 'real packaged official Agent.main loop over both arms with scripted loopback upstream; NOT model/GPU observation',
                          'failure_fixture': failure, 'actor_revision_fixture': revise,
                          'game_identity': identity, 'result': {k: result[k] for k in ('status', 'games', 'calls', 'arm_calls') if k in result}}, sort_keys=True))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_guard_reaps_setsid_descendants_on_outer_death(tmp_path):
    guard = PACKAGE / 'process_guard.py'
    receipt = tmp_path / 'guard.json'
    pidfile = tmp_path / 'descendant.json'
    child = 'import os,time,json; from pathlib import Path; Path(' + repr(str(pidfile)) + ').write_text(json.dumps({"pid":os.getpid()})); time.sleep(30)'
    parent = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",' + repr(child) + '],start_new_session=True); time.sleep(30)'
    outer = 'import subprocess,sys,os,time; subprocess.Popen([sys.executable,"-I","-S",' + repr(str(guard)) + ',str(os.getpid()),"20",' + repr(str(receipt)) + ',sys.executable,"-c",' + repr(parent) + ']); time.sleep(30)'
    proc = subprocess.Popen([sys.executable, '-c', outer])
    try:
        deadline = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(.03)
        assert pidfile.exists()
        pid = json.loads(pidfile.read_text())['pid']
        proc.kill()
        proc.wait(3)
        deadline = time.monotonic() + 10
        while not receipt.exists() and time.monotonic() < deadline:
            time.sleep(.03)
        assert receipt.exists()
        record = json.loads(receipt.read_text())
        assert record['status'] == 'failed'
        assert record['cleanup'] == 'retired'
        assert not Path(f'/proc/{pid}').exists()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
