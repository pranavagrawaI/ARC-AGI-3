"""Local qualification of the exact notebook payload and real play entry point."""
import ast
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import zipfile

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DONOR = Path(os.environ.get('SUBMISSION_TEST_RUNTIME', str(HERE.parent / 'submission-runtime-20260910-v4'))).resolve()
OFFICIAL = Path('/home/pranav/projects/arc-agi-3-reference/kaggle/runs/nooa-upstream/ARC-AGI-3-Agents')


@pytest.mark.parametrize('count', [1, 3, 25, 55, 110, 135])
def test_competition_budget(count):
    spec = importlib.util.spec_from_file_location('submission_play', HERE/'play.py')
    play = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(play)
    budget = play.admission_budget(25140, count, 120)
    assert budget > 0
    assert count * (budget + 120 + 15 + 30) + 180 == pytest.approx(25140)
    if count == 110:
        assert budget == pytest.approx(61.9090909091)
    assert play.admission_budget(25130, max(1, count-1), 120) > budget or count == 1


@pytest.mark.parametrize('remaining,count,request_seconds', [(0,110,90), (100,0,90), (float('nan'),1,90), (1000,1,-1)])
def test_invalid_budget(remaining, count, request_seconds):
    spec = importlib.util.spec_from_file_location('submission_play', HERE/'play.py')
    play = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(play)
    with pytest.raises(ValueError):
        play.admission_budget(remaining, count, request_seconds)


def test_notebook_payload_matches_frozen_candidate():
    notebook = json.loads((HERE/'submission.ipynb').read_text())
    manifest = json.loads((DONOR/'CHASSIS.json').read_text())
    assert notebook['metadata']['kernelspec']['name'] == 'python3'
    assert len({c['id'] for c in notebook['cells']}) == len(notebook['cells'])
    payloads = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'b64decode' and node.args and isinstance(node.args[0], ast.Constant)):
                payloads.append(base64.b64decode(node.args[0].value))
    assert (HERE/'play.py').read_bytes() in payloads
    assert (DONOR/'adapter.py').read_bytes() in payloads
    assert (DONOR/'shared_local_solver.py').read_bytes() in payloads
    archives = [zipfile.ZipFile(io.BytesIO(p)) for p in payloads if p.startswith(b'PK')]
    archive = next(z for z in archives if 'CHASSIS.json' in z.namelist())
    assert json.loads(archive.read('CHASSIS.json')) == manifest
    for name, expected in manifest['files'].items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
    source = ''.join(notebook['cells'][-1]['source'])
    assert 'refuses competition rerun' not in source
    assert "'--official-launcher-probe','/tmp/submission_play.py'" in source


@pytest.mark.parametrize('gateway', [False, True])
@pytest.mark.parametrize('exhausted', [False, True, 'thinking', 'slow_http', 'http_error', 'stalled_tool'])
def test_real_entrypoint(tmp_path, gateway, exhausted):
    fault = exhausted in ('slow_http', 'http_error', 'stalled_tool')
    framework = tmp_path/'framework'
    (framework/'agents/templates').mkdir(parents=True)
    for name in ['agent.py', 'recorder.py', 'tracing.py', 'swarm.py']:
        shutil.copy2(OFFICIAL/'agents'/name, framework/'agents'/name)
    (framework/'agents/__init__.py').write_text('from .agent import Agent, Playback\nfrom .swarm import Swarm\n')
    shutil.copy2(DONOR/'adapter.py', framework/'agents/templates/my_agent.py')
    (framework/'runtime').mkdir()
    (framework/'runtime/__init__.py').write_text('')
    shutil.copy2(DONOR/'shared_local_solver.py', framework/'runtime/local_solver.py')
    shutil.copy2(DONOR/'local_solver.py', framework/'local_solver.py')
    requests = []
    actor_requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            if exhausted == 'http_error':
                self.send_error(503, 'injected qualification transport failure')
                return
            if exhausted == 'slow_http':
                time.sleep(5)
            if len(requests) > 40:
                self.send_error(429, 'qualification request cap')
                return
            is_actor = any(t.get('function', {}).get('name') == 'execute_python'
                           for t in requests[-1].get('tools', []))
            if is_actor:
                actor_requests.append(requests[-1])
            action = ['UP', 'DOWN', 'LEFT', 'RIGHT'][(len(actor_requests)-1) % 4]
            reply = dict(id=f'local-{len(requests)}', object='chat.completion', created=1,
                model='controlled-local', choices=[dict(index=0, finish_reason='tool_calls', message=dict(
                    role='assistant', content=None, tool_calls=[dict(id=f'call-{len(requests)}', type='function',
                    function=dict(name='execute_python', arguments=json.dumps(dict(code=
                    f'grid=self.model_grid(); self.submit_actions([{action!r}], rationale="predict: controlled movement")'))))]))],
                usage=dict(prompt_tokens=22000, completion_tokens=50, total_tokens=22050))
            if not is_actor:
                reply['choices'] = [dict(index=0, finish_reason='stop',
                    message=dict(role='assistant', content=json.dumps(dict(
                        value='Controlled summary: mechanics and goal remain unproven.'))))]
            data=json.dumps(reply).encode()
            if exhausted == 'thinking':
                time.sleep(1)
                reply['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = json.dumps(
                    dict(code='print(self.model_grid())'))
                data = json.dumps(reply).encode()
            if exhausted == 'stalled_tool':
                reply['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = json.dumps(
                    dict(code='while True: pass'))
                data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected for the intentionally timed-out HTTP request.

    server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    env = dict(os.environ, KAGGLE_IS_COMPETITION_RERUN='true' if gateway else 'false', SUBMISSION_FRAMEWORK=str(framework),
        SUBMISSION_ENVIRONMENTS=str(ROOT/'environment_files'), SUBMISSION_PUBLIC_GAMES='ls20',
        NOOA_JOIN_ROOT=str(DONOR), ARC_LLM_MAX_TOKENS='16384', ARC_LLM_RETRIES='0',
        ARC_LLM_TIMEOUT_SECONDS='30', TYCHO_NOOA_WORKER_TIMEOUT_SECONDS='40',
        ARC_STALL_REVIEW='0', ARC_DISCOVERY_ARM='a', ARC_SOLVER_ARM='loop', ARC_CONTINGENT_ARM='on', SUBMISSION_ACTION_LIMIT='10',
        ARC_LLM_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false},"reasoning":{"enabled":false}}')
    if exhausted:
        env['SUBMISSION_ADMISSION_CAP_SECONDS'] = '12' if exhausted == 'thinking' else '0'
    if fault:
        env['SUBMISSION_ADMISSION_CAP_SECONDS'] = '12'
        env['ARC_LLM_TIMEOUT_SECONDS'] = '2'
    expected_games = 2 if exhausted is True and not gateway else 1
    if expected_games == 2:
        # Two invocations prove the first exhausted game does not abort the loop.
        env['SUBMISSION_PUBLIC_GAMES'] = 'ls20,ls20'
    gateway_server = None
    if gateway:
        from arc_agi import Arcade, OperationMode
        from arc_agi.server import create_app
        from werkzeug.serving import make_server
        arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT/'environment_files'))
        app, _ = create_app(arcade, competition_mode=True)
        gateway_server = make_server('127.0.0.1', 0, app, threaded=True)
        gateway_thread = threading.Thread(target=gateway_server.serve_forever, daemon=True)
        gateway_thread.start()
        env['SUBMISSION_GATEWAY_BASE_URL'] = f'http://127.0.0.1:{gateway_server.server_port}'
    try:
        # Widen the observed finish/cleanup race without bypassing Agent.main.
        runner = ('import sys,time;sys.path.insert(0,' + repr(str(framework)) + ');'
                  'from agents.templates.my_agent import MyAgent;'
                  'original=MyAgent.cleanup;'
                  'MyAgent.cleanup=lambda self:(time.sleep(1),original(self))[-1];'
                  'import runpy;runpy.run_path(sys.argv.pop(1),run_name="__main__")')
        completed = subprocess.run([sys.executable,'-c',runner,str(HERE/'play.py'),
            '--base-url',f'http://127.0.0.1:{server.server_port}/v1','--model','qwen/qwen3.8-27b',
            '--workspace-root',str(tmp_path/'workspaces'),'--output',str(tmp_path/'RESULT.json'),
            '--timeout-s','1200'], env=env, capture_output=True, text=True, timeout=180)
        (tmp_path/'stdout.log').write_text(completed.stdout)
        (tmp_path/'stderr.log').write_text(completed.stderr)
        (tmp_path/'requests.json').write_text(json.dumps(requests,indent=2)+'\n')
        if fault:
            assert completed.returncode != 0, 'Injected failure was incorrectly accepted'
            assert not (tmp_path/'RESULT.json').exists()
            assert (tmp_path/'RESULT.failure.json').exists()
            assert requests, 'Fixture failed before reaching injected request'
            if exhausted == 'stalled_tool':
                assert 'timed out' in completed.stderr.lower()
                assert not list((tmp_path/'workspaces').rglob('budget_exhausted.json'))
            else:
                lifecycle = [json.loads(line) for p in (tmp_path/'workspaces').rglob('request_lifecycle.jsonl')
                             for line in p.read_text().splitlines()]
                assert any(row['event'] == 'failed' for row in lifecycle)
                assert len(requests) == 1, 'A failed call made another HTTP attempt'
                assert 'Model request failed' in completed.stderr
            assert not (tmp_path/'game-progress.jsonl').exists(), 'Fault was mistaken for normal game completion'
            # PIDs recorded by the real worker must be retired before returning.
            for p in (tmp_path/'workspaces').rglob('worker-metadata.json'):
                pid = json.loads(p.read_text())['agent_pid']
                with pytest.raises(ProcessLookupError):
                    os.kill(pid, 0)
            return
        assert completed.returncode == 0, completed.stdout + completed.stderr
        result=json.loads((tmp_path/'RESULT.json').read_text())
        assert result['status']=='passed'
        assert result['operation_mode']==('gateway' if gateway else 'offline')
        assert len(result['games'])==expected_games
        assert result['games'][0]['solver_actions']==(0 if exhausted else 10)
        assert result['solver_sessions']['total_actor_calls']==len(requests)
        if exhausted == 'thinking':
            assert len(requests) >= 2, 'Did not exercise a multi-call turn'
        else:
            assert len(requests) == (0 if exhausted else 10), 'Model work escaped the host action-admission boundary'
        if exhausted:
            assert result['games'][0]['stop_reason'] == 'model_admission_budget'
            assert len(list((tmp_path/'workspaces').rglob('budget_exhausted.json'))) == expected_games
        allocations = [json.loads(line) for line in (tmp_path/'budget-allocations.jsonl').read_text().splitlines()]
        assert len(allocations) == expected_games
        if expected_games == 2:
            assert allocations[1]['remaining_games'] == 1
            assert allocations[1]['remaining_seconds'] < allocations[0]['remaining_seconds']
        assert not list((tmp_path/'workspaces').rglob('ABORT'))
        for p in (tmp_path/'workspaces').rglob('request_lifecycle.jsonl'):
            events = [json.loads(line) for line in p.read_text().splitlines()]
            assert all(r['worker_peak_rss_kib'] > 0 for r in events if r['event'] == 'admitted')
        timing = [json.loads(line) for p in (tmp_path/'workspaces').rglob('execution_timing.jsonl')
                  for line in p.read_text().splitlines()]
        assert len(timing) == sum(g['solver_actions'] for g in result['games'])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if gateway_server:
            gateway_server.shutdown()
            gateway_server.server_close()
            gateway_thread.join(timeout=2)
