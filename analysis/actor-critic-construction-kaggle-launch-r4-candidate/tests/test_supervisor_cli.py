"""Exact notebook-delivered bootstrap argv through the unmodified real CLI parser.
No model, GPU, installer, or provider call is substituted by a passing mock.
"""
from __future__ import annotations
import ast
import base64
import json
import subprocess
import sys
import pytest
from test_runtime import materialize, ROOT


def command_from_notebook(notebook, root):
    tree = ast.parse(''.join(notebook['cells'][1]['source']))
    assets = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ASSETS' for t in n.targets)))
    source = base64.b64decode(assets['donor_bootstrap.py'], validate=True).decode()
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    assignment = next(n for n in main.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'command' for t in n.targets))
    # Evaluate the actual generated expression, not a hand-copied command or a
    # run_probe(args) fixture that bypasses argument validation.
    return eval(compile(ast.Expression(assignment.value), 'generated-bootstrap-command', 'eval'), {'sys': sys, 'HERE': root})


def parse(command):
    code = "import json,runpy,sys; ns=runpy.run_path(sys.argv[1]); sys.argv=sys.argv[1:]; args=ns['parse_args'](); print(json.dumps(vars(args),default=str,sort_keys=True))"
    return subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, *command[1:]], capture_output=True, text=True, timeout=15)


def test_generated_predecessor_rejects_successor_accepts_same_real_parser(tmp_path):
    ns, notebook = materialize(tmp_path)
    successor = command_from_notebook(notebook, ns['root'])
    predecessor_nb = json.loads((ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-smoke/construction.ipynb').read_text())
    predecessor = command_from_notebook(predecessor_nb, ns['root'])
    difference = [(a, b) for a, b in zip(predecessor, successor) if a != b]
    assert len(predecessor) == len(successor) and difference == [('30', '180')]
    rejected, accepted = parse(predecessor), parse(successor)
    assert rejected.returncode == 2
    assert '--r8-l3b requires --startup-timeout-s 1200 --request-timeout-s 180' in rejected.stderr
    assert accepted.returncode == 0, accepted.stderr
    args = json.loads(accepted.stdout)
    assert (args['startup_timeout_s'], args['request_timeout_s'], args['launcher_timeout_s'], args['port']) == (1200, 180, 1020, 30000)
    assert args['r8_l3b'] and args['attention_backend'] == 'triton' and args['sampling_backend'] == 'pytorch'


@pytest.mark.parametrize('flag,value,message', [
    ('--startup-timeout-s', '30', '--startup-timeout-s 1200'),
    ('--launcher-timeout-s', '0', '--launcher-timeout-s must be positive'),
    ('--attention-backend', 'flashinfer', 'require --attention-backend triton'),
    ('--sampling-backend', 'flashinfer', '--sampling-backend pytorch'),
    ('--port', '30001', '--r8-l3b requires --port 30000'),
])
def test_next_cli_guards_remain_fail_closed(tmp_path, flag, value, message):
    ns, notebook = materialize(tmp_path)
    command = command_from_notebook(notebook, ns['root'])
    if flag in command:
        command[command.index(flag) + 1] = value
    else:
        command += [flag, value]
    result = parse(command)
    assert result.returncode == 2 and message in result.stderr


def test_real_supervisor_entrypoint_reaches_first_unavailable_prerequisite(tmp_path):
    ns, notebook = materialize(tmp_path)
    command = command_from_notebook(notebook, ns['root'])
    out = tmp_path / 'real-supervisor-receipt'
    command[command.index('--output-dir') + 1] = str(out)
    # Real main -> real parser -> real system snapshot -> real GPU preflight.
    # This host lacks the required GPU. No synthetic success or run_probe patch.
    result = subprocess.run(command[:1] + ['-I', '-S', '-B'] + command[1:], env={'PATH': '/opt/bin:/usr/local/bin:/usr/bin:/bin', 'HOME': str(ns['root']), 'LANG': 'C.UTF-8', 'ARC_DISABLE_DOTENV': '1'}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stdout + result.stderr
    config = json.loads((out / 'probe-config.json').read_text())
    assert config['request_timeout_s'] == 180 and config['startup_timeout_s'] == 1200
    assert (out / 'system-before.json').exists()
    receipt = json.loads((out / 'result.json').read_text())
    assert receipt['failure_stage'] == 'accelerator_preflight', receipt
    assert receipt['status'] == 'invalid'
    assert not (out / 'runtime-verification.json').exists()
    assert not (out / 'server.log').exists()
