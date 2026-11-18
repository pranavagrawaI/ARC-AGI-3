"""Exercise the real HostSession action/observation boundary with frozen helpers."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'chassis'))
spec = importlib.util.spec_from_file_location('designer_step_worker', HERE / 'nooa_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
from step_check import bind_prediction

SOURCE = 'def encode(grid):\n    return {"x": int(grid[0,0])}\ndef predict(z, action):\n    return {"x": z["x"] + 1}\n'


def observation(x, state='NOT_FINISHED', level=0):
    return SimpleNamespace(grid=[[x]], state=state, levels_completed=level,
                           available_actions=['RIGHT'])


def session(tmp_path, monkeypatch, source=SOURCE, arm='loop'):
    monkeypatch.setenv('ARC_SOLVER_ARM', arm)
    host = worker.HostSession(tmp_path)
    batch = {'turn': 0, 'step': 0, 'levels_completed': 0, 'actions': ['RIGHT', 'RIGHT']}
    record = {**batch, 'status': 'committed', 'source': source,
              'model_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'predicted': [{'x': 1}, {'x': 2}]}
    (host.ipc / 'model_predictions.jsonl').write_text(json.dumps(record) + '\n')
    host.actions_path.write_text(json.dumps(batch) + '\n')
    host.decide(SimpleNamespace(sequence=1, observation=observation(0), level=0,
                               decision_timeout_seconds=1))
    return host


def observe(host, x=1, state='NOT_FINISHED', level=0):
    return host.observe(SimpleNamespace(sequence=2, before=observation(0),
                        after=observation(x, state, level)))


def test_match_retains_queue_and_frozen_version(tmp_path, monkeypatch):
    host = session(tmp_path, monkeypatch)
    # A rewritten on-disk helper/record must not replace the admitted snapshot.
    (host.ipc / 'model_predictions.jsonl').write_text('{}\n')
    observe(host)
    assert host._queue == ['RIGHT'] and host._nooa_turn == 0
    assert json.loads((host.ipc / 'step_checks.jsonl').read_text())['status'] == 'matched'


@pytest.mark.parametrize('source,x,status', [(SOURCE, 3, 'mismatch'),
    ('def encode(grid):\n    raise ValueError("bad")\n', 1, 'unresolved'),
    ('def encode(grid):\n    while True: pass\n', 1, 'unresolved')])
def test_failure_stops_and_publishes_successor(tmp_path, monkeypatch, source, x, status):
    host = session(tmp_path, monkeypatch, source)
    observe(host, x)
    assert host._queue == [] and host._nooa_turn == 1
    check = json.loads((host.ipc / 'step_checks.jsonl').read_text())
    assert check['status'] == status and check['skipped_actions'] == 1
    states = [json.loads(r) for r in host.states_path.read_text().splitlines()]
    assert states[-1]['turn'] == 1 and status in states[-1]['note']


@pytest.mark.parametrize('state,level', [('WIN', 0), ('GAME_OVER', 0), ('NOT_FINISHED', 1)])
def test_terminal_precedes_encoder(tmp_path, monkeypatch, state, level):
    host = session(tmp_path, monkeypatch, 'def encode(grid):\n    while True: pass\n')
    observe(host, state=state, level=level)
    assert not host._queue
    assert not (host.ipc / 'step_checks.jsonl').exists()


def test_baseline_does_not_interrupt_mismatch(tmp_path, monkeypatch):
    host = session(tmp_path, monkeypatch, arm='baseline')
    observe(host, 3)
    assert host._queue == ['RIGHT']
    assert not (host.ipc / 'step_checks.jsonl').exists()


def test_wall_timeout_is_killed_and_retires_queue(tmp_path, monkeypatch):
    host = session(tmp_path, monkeypatch, 'def encode(grid):\n    while True: pass\n')
    real_check = worker.check_step
    monkeypatch.setattr(worker, 'check_step', lambda record, transition:
                        real_check(record, transition, timeout=0.3))
    observe(host)
    checked = json.loads((host.ipc / 'step_checks.jsonl').read_text())
    assert checked['status'] == 'unresolved'
    assert 'TimeoutError' in checked['reason']
    assert not host._queue and host._nooa_turn == 1


def test_encoder_prints_are_diagnostics_not_verdict(tmp_path, monkeypatch):
    source = SOURCE.replace('return {"x": int(grid[0,0])}',
                            'print("encoding grid")\n    return {"x": int(grid[0,0])}')
    host = session(tmp_path, monkeypatch, source)
    observe(host)
    checked = json.loads((host.ipc / 'step_checks.jsonl').read_text())
    assert checked['status'] == 'matched'
    assert checked['diagnostics'] == 'encoding grid\n'
    assert host._queue == ['RIGHT']


@pytest.mark.parametrize('termination', [signal.SIGTERM, signal.SIGKILL])
def test_interrupted_parent_kills_real_checker(tmp_path, termination):
    # Popen returns only after pre-exec setup. Its PID line is our admission
    # barrier, not an arbitrary delay hoping the subprocess has started.
    script = '''
import hashlib, json, subprocess, sys
sys.path.insert(0, sys.argv[1])
import step_check
original = subprocess.Popen
def admitted(*args, **kwargs):
    child = original(*args, **kwargs)
    print(child.pid, flush=True)
    return child
subprocess.Popen = admitted
source = 'def encode(grid):\\n    while True: pass\\n'
record = dict(status='committed', source=source,
    model_sha256=hashlib.sha256(source.encode()).hexdigest(), turn=0, step=0,
    levels_completed=0, actions=['RIGHT'], predicted=[{'x':1}])
transition = dict(turn=0, step=1, action='RIGHT',
    before={'levels_completed':0}, after={'grid':[[1]]})
step_check.check_step(record, transition, timeout=30)
'''
    parent = subprocess.Popen([sys.executable, '-c', script, str(HERE / 'chassis')],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child_pid = None
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(parent.stdout, selectors.EVENT_READ)
            assert selector.select(5), 'checker admission barrier timed out'
            line = parent.stdout.readline().strip()
            assert line.isdigit(), parent.stderr.read()
            child_pid = int(line)
        os.kill(parent.pid, termination)
        parent.wait(timeout=3)
        deadline = time.monotonic() + 0.8
        while time.monotonic() < deadline:
            try:
                status = Path(f'/proc/{child_pid}/stat').read_text().split()[2]
            except FileNotFoundError:
                break
            if status == 'Z':  # Exited; init may not have reaped it yet.
                break
            time.sleep(0.01)
        else:
            pytest.fail('checker survived parent interruption')
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=3)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize('change', [{'turn': 3}, {'step': 1}, {'actions': ['UP']},
                                  {'model_sha256': 'bad'}])
def test_stale_or_corrupt_prediction_not_bound(tmp_path, monkeypatch, change):
    host = session(tmp_path, monkeypatch)
    record = {**host._prediction, **change}
    (host.ipc / 'model_predictions.jsonl').write_text(json.dumps(record) + '\n')
    assert bind_prediction(host.ipc, host.evidence.batch)['status'] == 'unavailable'
