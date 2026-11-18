"""Run the real official Agent + frozen NOOA worker on local public LS20.

Controlled model replies exercise integration, never model capability.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import types


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--framework', type=Path, required=True)
    parser.add_argument('--environments', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    framework = args.framework.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((candidate / 'CHASSIS.json').read_text())
    for group in ('files', 'sources'):
        for name, expected in manifest[group].items():
            assert hashlib.sha256((candidate / name).read_bytes()).hexdigest() == expected, name
    expected_agent = '85655f7a11a105e50bba76efd00bebf8056d02ddbae827a8f8a5b46e849fc087'
    assert hashlib.sha256((framework / 'agents/agent.py').read_bytes()).hexdigest() == expected_agent
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(request)
            with (output / 'requests.jsonl').open('a') as stream:
                stream.write(json.dumps(request) + '\n')
            if len(requests) > 6:
                self.send_error(429, 'local qualification request cap exceeded')
                return
            code = 'grid = self.model_grid(); print(grid.shape); self.submit_actions(["UP"], rationale="predict: controlled local movement check")'
            reply = {'id': f'local-{len(requests)}', 'object': 'chat.completion', 'created': 1,
                     'model': 'controlled-local', 'choices': [{'index': 0, 'finish_reason': 'tool_calls',
                     'message': {'role': 'assistant', 'content': None, 'tool_calls': [
                     {'id': f'call-{len(requests)}', 'type': 'function', 'function': {
                     'name': 'execute_python', 'arguments': json.dumps({'code': code})}}]}}],
                     'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    os.environ.update({
        'ARC_DISABLE_DOTENV': '1', 'ARC_LLM_MODEL': 'qwen/qwen3.8-27b',
        'ARC_LLM_API_KEY': 'controlled-local', 'ARC_LLM_BASE_URL': f'http://127.0.0.1:{server.server_port}/v1',
        'ARC_LLM_EXTRA_BODY': '{"chat_template_kwargs":{"enable_thinking":false},"reasoning":{"enabled":false}}',
        'ARC_LLM_MAX_TOKENS': '16384', 'ARC_LLM_RETRIES': '0', 'ARC_LLM_TIMEOUT_SECONDS': '30',
        'ARC_SOLVER_ARM': 'loop', 'ARC_DISCOVERY_ARM': 'a', 'ARC_CONTINGENT_ARM': 'on',
        'ARC_STALL_REVIEW': '0', 'ARC_LLM_USE_RESPONSES': '0',
        'TYCHO_KAGGLE_MAX_ACTIONS': '3', 'TYCHO_GAMEPLAY_ADMISSION_SECONDS': '120',
        'TYCHO_KAGGLE_WORKSPACE_ROOT': str(output / 'workspaces'),
        'TYCHO_NOOA_WORKER_TIMEOUT_SECONDS': '40', 'TYCHO_NOOA_WORKER_PYTHON': sys.executable,
        'NOOA_JOIN_ROOT': str(candidate), 'LITELLM_LOCAL_MODEL_COST_MAP': 'True',
        'TYCHO_SANDBOX_RUNTIME': 'host', 'RECORDINGS_DIR': str(output / 'recordings'),
    })
    os.environ.pop('TYCHO_KAGGLE_HARDENED_HOST', None)
    # Load the pinned official classes without its unrelated template registry.
    agents = types.ModuleType('agents')
    agents.__path__ = [str(framework / 'agents')]
    sys.modules['agents'] = agents
    runtime = types.ModuleType('runtime')
    runtime.__path__ = [str(candidate)]
    sys.modules['runtime'] = runtime
    module('runtime.local_solver', candidate / 'shared_local_solver.py')
    module('local_solver', candidate / 'local_solver.py')
    adapter = module('submission_local_adapter', candidate / 'adapter.py')
    from arc_agi import Arcade, OperationMode
    arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=str(args.environments.resolve()))
    scorecard = arcade.open_scorecard()
    env = arcade.make('ls20', scorecard_id=scorecard)
    assert env is not None
    agent = adapter.MyAgent(card_id=scorecard, game_id='ls20', agent_name='local-qualification',
                            ROOT_URL='http://127.0.0.1', record=True, arc_env=env, tags=['local-qualification'])
    started = time.monotonic()
    try:
        agent.main()
        transitions = [json.loads(line) for p in (output / 'workspaces').rglob('ipc/transitions.jsonl')
                       for line in p.read_text().splitlines()]
        executed = [r for r in transitions if not r.get('automatic')]
        assert len(executed) == 3, transitions
        assert len(requests) == 3, len(requests)
        assert not list((output / 'workspaces').rglob('ABORT'))
        result = dict(status='passed', claim='real official game loop and packaged worker with controlled responses',
                      candidate=str(candidate), contract_sha256=manifest['contract_sha256'],
                      official_agent_sha256=expected_agent, seconds=time.monotonic()-started,
                      model_calls=len(requests), executed_solver_actions=len(executed),
                      levels_completed=agent.frames[-1].levels_completed,
                      live_model=False, raw_transitions=transitions)
        (output / 'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k != 'raw_transitions'}, indent=2))
    finally:
        agent.cleanup()
        arcade.close_scorecard(scorecard)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == '__main__':
    main()
