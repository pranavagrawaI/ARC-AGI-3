"""Generated notebook + actual production acceptance, with explicit unavailable-context fixtures.
No target GPU, model, provider, or remote launch claim. HTTP is loopback only.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
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
PACKAGE = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-smoke'
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


@pytest.mark.parametrize('failure,revise', [(None, False), (None, True), ('missing_usage', False), ('wrong_model', False), ('second_http', False), ('actor_cap', False)])
def test_actual_supervisor_worker_http(tmp_path, monkeypatch, failure, revise):
    ns, _ = materialize(tmp_path)
    package = ns['root']
    monkeypatch.syspath_prepend(str(package))
    for name in ('local_wire', 'local_accounting'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    received, actor_calls, ctor_calls = [], 0, 0
    action = 'LEFT' if revise else 'RIGHT'
    codes = ['print(self.observations())'] * 8 + ["self.submit_actions(['UP'], rationale='fixture initial probe')", "print(self.submit_actions(['RIGHT'], rationale='actor-selected fixture choice'))"]
    if revise:
        codes += ["print(self.submit_actions(['LEFT'], rationale='actor revises unexecuted choice'))"]
    codes += [f"print(self.submit_actions(['{action}'], rationale='confirm actor-selected fixture choice'))"]
    if failure == 'actor_cap':
        codes = ['print(self.observations())'] * 24

    class Fixture(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            nonlocal actor_calls, ctor_calls
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
            if 'submit_candidate' in names:
                ctor_calls += 1
                if ctor_calls == 1:
                    fn, args = 'execute_python', {'code': 'import json\nx=json.load(open("/evidence/case.json"))\nprint(type(x).__name__)'}
                else:
                    fn, args = 'submit_candidate', {'work_item': WORK, 'candidate_source': CANDIDATE}
                calls = [{'id': f'opaque/ctor:{ctor_calls}', 'index': 0, 'type': 'function', 'function': {'name': fn, 'arguments': json.dumps(args)}}]
                message = {'role': 'assistant', 'content': None, 'tool_calls': calls}
            elif body.get('max_tokens') == 192:
                message = {'role': 'assistant', 'content': '{"claim_ref":"","question":"","reason":""}'}
            else:
                actor_calls += 1
                code = codes[actor_calls - 1] if actor_calls <= len(codes) else 'print(self.observations())'
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'opaque/actor:{actor_calls}', 'index': 0, 'type': 'function', 'function': {'name': 'execute_python', 'arguments': json.dumps({'code': code})}}]}
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
    # Only unavailable hardware/mount context is supplied in this explicit test wrapper.
    # The actual generated probe, containment, driver, worker and local relay are unchanged.
    wrapper = tmp_path / 'hardware_fixture.py'
    wrapper.write_text(f'''import sys, json, importlib.util, socket
from pathlib import Path
sys.path.insert(0, {str(package)!r})
def network(event, args):
    if event == 'socket.connect' and args[1][0] not in ('127.0.0.1', '::1'):
        raise RuntimeError('fixture forbids non-loopback network')
sys.addaudithook(network)
import kaggle_probe
kaggle_probe.inspect_gpu = lambda out: (out / 'GPU-FIXTURE.json').write_text(json.dumps({{'evidence_kind': 'synthetic hardware context, NOT GPU observation'}}))
sys.argv += ['--environments', {str(ROOT / 'environment_files')!r}]
raise SystemExit(kaggle_probe.main())
''')
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
    globals_ = dict(vars(supervisor), args=SimpleNamespace(launcher_python=sys.executable, official_launcher_probe=wrapper, launcher_timeout_s=120),
        process=fixture_process, warmup_facts=warmup, deadline_raw=None, output_dir=output_dir,
        base_url=f'http://127.0.0.1:{server.server_port}', model_name=LOCAL_MODEL, command=STARTUP,
        started_at='explicit-fixture', started=time.monotonic(), event=lambda message: (output_dir / 'fixture-supervisor-event.txt').write_text(message), mode='explicit-fixture', model_verification={'status': 'synthetic-not-model-observation'}, runtime_verification='synthetic-context', config={'claim_ceiling': 'local fixture only'})
    # Counters are explicitly HTTP-fixture observations, never SGLang log claims.
    globals_['wait_for_chat_completion_count'] = lambda *_a, **_kw: {'chat_completion_accesses': 1 + len(received), 'successful_chat_completion_accesses': 1 + len(received), 'successful_freeze_gc_accesses': 1, 'ready_marker_count': 1}
    exec(compile(program, 'actual-generated-supervisor-acceptance', 'exec'), globals_)
    try:
        if failure:
            with pytest.raises(RuntimeError, match='official launcher command failed'):
                globals_['exercise']()
        else:
            result = globals_['exercise']()
            assert result['status'] == 'passed'
            counts = json.loads((output_dir / 'request-count-facts.json').read_text())
            assert counts['expected_launcher_solver_requests'] == len(received)
        raw_dir = output_dir / 'solver-workspaces/raw'
        result = json.loads((raw_dir / 'RESULTS.json').read_text())
        if failure:
            assert result['status'] == 'failed'
            if failure == 'actor_cap':
                assert result['actor_summarizer'] == 20
                assert result['actions'] == 0
                assert result['calls'] == len(received) <= 23
                assert result['stop'] == 'actor/summarizer reserve'
            else:
                assert len(received) == (2 if failure == 'second_http' else 1)
                assert not result['local_requests']['complete']
                assert result['local_requests']['known_usage']['prompt_tokens'] == (100 if failure == 'second_http' else 0)
                assert result['local_requests']['unreceipted_usage'] == 'unknown'
        else:
            assert result['status'] == 'passed', result
            import copy
            for key, value in [('model', 'wrong'), ('admitted', 999), ('complete', False)]:
                corrupted = copy.deepcopy(result)
                corrupted['local_requests'][key] = value
                with pytest.raises(ValueError):
                    supervisor.expected_launcher_solver_requests(corrupted)
            assert result['qualification_complete'] is True
            assert result['actions'] == 2
            assert result['local_requests']['roles']['constructor'] == 2
            assert result['audit']['action_step_digest_bound'] is True
            assert result['audit']['env_matches_transitions'] is True
            assert any(m.get('tool_call_id') == 'opaque/ctor:1' for b in received for m in b['messages'])
            assert not list(raw_dir.rglob('GPU.json'))
            assert any(json.loads(line)['status'] == 'abort_retired_after_qualification' for line in (raw_dir / 'cleanup.jsonl').read_text().splitlines())
        assert json.loads((raw_dir / 'SEAL.json').read_text())['status'] == 'passed'
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
