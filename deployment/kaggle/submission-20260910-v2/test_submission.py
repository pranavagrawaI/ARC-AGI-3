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
import zipfile

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DONOR = HERE.parent / 'contingent-controller-20260909'
OFFICIAL = Path('/home/pranav/projects/arc-agi-3-reference/kaggle/runs/nooa-upstream/ARC-AGI-3-Agents')


@pytest.mark.parametrize('count', [1, 3, 25, 55, 110, 135])
def test_competition_budget(count):
    spec = importlib.util.spec_from_file_location('submission_play', HERE/'play.py')
    play = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(play)
    budget = play.admission_budget(25140, count, 120)
    assert budget > 0
    assert count * (budget + 120 + 30) + 180 <= 25140
    if count == 110:
        assert budget == 75


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
def test_real_entrypoint(tmp_path, gateway):
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

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            if len(requests) > 20:
                self.send_error(429, 'qualification request cap')
                return
            action = ['UP', 'DOWN', 'LEFT', 'RIGHT'][(len(requests)-1) % 4]
            reply = dict(id=f'local-{len(requests)}', object='chat.completion', created=1,
                model='controlled-local', choices=[dict(index=0, finish_reason='tool_calls', message=dict(
                    role='assistant', content=None, tool_calls=[dict(id=f'call-{len(requests)}', type='function',
                    function=dict(name='execute_python', arguments=json.dumps(dict(code=
                    f'grid=self.model_grid(); self.submit_actions([{action!r}], rationale="predict: controlled movement")'))))]))],
                usage=dict(prompt_tokens=100, completion_tokens=50, total_tokens=150))
            data=json.dumps(reply).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    env = dict(os.environ, KAGGLE_IS_COMPETITION_RERUN='true' if gateway else 'false', SUBMISSION_FRAMEWORK=str(framework),
        SUBMISSION_ENVIRONMENTS=str(ROOT/'environment_files'), SUBMISSION_PUBLIC_GAMES='ls20',
        NOOA_JOIN_ROOT=str(DONOR), ARC_LLM_MAX_TOKENS='16384', ARC_LLM_RETRIES='0',
        ARC_LLM_TIMEOUT_SECONDS='30', TYCHO_NOOA_WORKER_TIMEOUT_SECONDS='40',
        ARC_STALL_REVIEW='0', ARC_DISCOVERY_ARM='a', ARC_SOLVER_ARM='loop', ARC_CONTINGENT_ARM='on', SUBMISSION_ACTION_LIMIT='10',
        ARC_LLM_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false},"reasoning":{"enabled":false}}')
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
        completed = subprocess.run([sys.executable,str(HERE/'play.py'),
            '--base-url',f'http://127.0.0.1:{server.server_port}/v1','--model','qwen/qwen3.8-27b',
            '--workspace-root',str(tmp_path/'workspaces'),'--output',str(tmp_path/'RESULT.json'),
            '--timeout-s','1200'], env=env, capture_output=True, text=True, timeout=180)
        (tmp_path/'stdout.log').write_text(completed.stdout)
        (tmp_path/'stderr.log').write_text(completed.stderr)
        (tmp_path/'requests.json').write_text(json.dumps(requests,indent=2)+'\n')
        assert completed.returncode == 0, completed.stdout + completed.stderr
        result=json.loads((tmp_path/'RESULT.json').read_text())
        assert result['status']=='passed'
        assert result['operation_mode']==('gateway' if gateway else 'offline')
        assert len(result['games'])==1
        assert result['games'][0]['solver_actions']==10
        assert result['solver_sessions']['total_actor_calls']==len(requests)
        assert 10 <= len(requests) <= 20
        assert not list((tmp_path/'workspaces').rglob('ABORT'))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if gateway_server:
            gateway_server.shutdown()
            gateway_server.server_close()
            gateway_thread.join(timeout=2)
