"""Frozen per-action checks in a disposable process, never in the protocol host.

This is a resource boundary, not an OS security sandbox. The existing helper
source policy is retained; a match is consistency, not proof of game semantics.
"""
from __future__ import annotations

import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


def _die_with_parent(expected_pid):
    """Install before exec; close the fork-to-registration parent-death race."""
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError('could not install checker parent-death signal')
    if os.getppid() != expected_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def bind_prediction(ipc, batch):
    """Snapshot the exact prospective record before executing its first action."""
    path = Path(ipc) / 'model_predictions.jsonl'
    try:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        matching = [r for r in records if all(r.get(k) == batch.get(k)
                    for k in ('turn', 'step', 'actions', 'levels_completed'))]
        if len(matching) != 1:
            raise ValueError('missing or ambiguous current prediction')
        record = matching[0]
        if record.get('status') != 'committed':
            raise ValueError('current prediction is not committed')
        if hashlib.sha256(record['source'].encode()).hexdigest() != record['model_sha256']:
            raise ValueError('frozen source hash mismatch')
        if len(record['predicted']) != len(batch['actions']):
            raise ValueError('prediction length mismatch')
        return record
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {'status': 'unavailable', 'reason': str(error)}


def check_step(record, transition, timeout=2.0):
    result = {'status': 'unresolved', 'step': transition['step'],
              'turn': transition['turn'], 'model_sha256': record.get('model_sha256')}
    try:
        if record.get('status') != 'committed':
            raise ValueError(record.get('reason', 'no committed prediction'))
        index = transition['step'] - record['step'] - 1
        if (transition['turn'] != record['turn'] or index < 0
                or index >= len(record['actions'])
                or transition['action'] != record['actions'][index]
                or transition['before']['levels_completed'] != record['levels_completed']
                or hashlib.sha256(record['source'].encode()).hexdigest() != record['model_sha256']):
            raise ValueError('transition does not belong to frozen prediction')
        payload = {'source': record['source'], 'grid': transition['after']['grid'],
                   'expected': record['predicted'][index]}
        # File-backed output and child RLIMIT_FSIZE bound even noisy helpers.
        with tempfile.TemporaryDirectory(prefix='arc-step-') as directory, tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as result_file:
            parent_pid = os.getpid()
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--child', str(result_file.fileno())],
                stdin=subprocess.PIPE, stdout=output, stderr=output, cwd=directory,
                pass_fds=(result_file.fileno(),),
                preexec_fn=lambda: _die_with_parent(parent_pid),
                start_new_session=True, env={'PATH': os.defpath, 'OPENBLAS_NUM_THREADS': '1',
                                             'OMP_NUM_THREADS': '1'})
            try:
                process.communicate(json.dumps(payload).encode(), timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise TimeoutError('frozen encoder exceeded wall-clock budget')
            result_file.seek(0)
            data = result_file.read(65537)
            if process.returncode or len(data) > 65536:
                raise ValueError('encoder process failed or exceeded output budget')
            result.update(json.loads(data))
            output.seek(0)
            diagnostics = output.read(65536)
            if diagnostics:
                result['diagnostics'] = diagnostics.decode('utf-8', errors='replace')
    except (OSError, ValueError, KeyError, TypeError, TimeoutError, subprocess.SubprocessError) as error:
        result['reason'] = f'{type(error).__name__}: {error}'
    return result


def _child():
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    payload = json.load(sys.stdin)
    # Load only trusted validator definitions, not the Actor or its client.
    tree = ast.parse(Path(__file__).with_name('solver_agent.py').read_text())
    names = {'_HELPER_IMPORT_ALLOW', '_HELPER_BANNED_NAMES', '_HELPER_BANNED_DUNDERS'}
    definitions = [node for node in tree.body if
        (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name == '_check_helper_ast')]
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<trusted-validator>', 'exec'), namespace)
    error = namespace['_check_helper_ast'](payload['source'])
    if error:
        raise ValueError(error)
    import numpy as np
    from prediction_feedback import plain
    module = {'__name__': 'frozen_conjecture'}
    exec(compile(payload['source'], '<frozen-conjecture>', 'exec'), module)
    actual = plain(module['encode'](np.array(payload['grid'])))
    expected = payload['expected']
    # Helpers may print or emit warnings. Only trusted completion writes this
    # separate channel; neither stdout nor stderr participates in the verdict.
    with os.fdopen(int(sys.argv[2]), 'w') as result_file:
        json.dump({'status': 'matched' if actual == expected else 'mismatch',
                   'expected': expected, 'observed': actual}, result_file, allow_nan=False)


if __name__ == '__main__':
    _child()
