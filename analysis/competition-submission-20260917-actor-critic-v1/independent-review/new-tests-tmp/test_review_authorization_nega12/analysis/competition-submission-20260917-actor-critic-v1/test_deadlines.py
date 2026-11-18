"""Deadline hierarchy, including real outer failure cleanup without a GPU."""
import ast
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time

import pytest

RUNTIME = Path(__file__).resolve().parents[2] / 'deployment/kaggle/submission-20260917-actor-critic-competition-v1/runtime'


def outer():
    spec = importlib.util.spec_from_file_location('deadline_outer', RUNTIME/'exact_request_outer_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




def test_startup_reduces_launcher_allowance():
    m = outer()
    assert m.bounded_launcher_timeout(2700, 3100, 350) == 2700
    assert m.bounded_launcher_timeout(2700, 3100, 900) == 2170
    with pytest.raises(TimeoutError):
        m.bounded_launcher_timeout(2700, 3100, 3010)
    with pytest.raises(ValueError):
        m.bounded_launcher_timeout(2700, float('nan'), 350)






def test_actual_outer_alarm_enters_cleanup(tmp_path, monkeypatch):
    m = outer()
    monkeypatch.setattr(sys, 'argv', ['probe', '--self-test', '--output-dir', str(tmp_path/'raw')])
    monkeypatch.setenv('ARC_NOTEBOOK_WORK_DEADLINE', str(time.monotonic()+300))
    monkeypatch.setattr(m, 'system_snapshot', lambda: {})
    servers = []
    original = m.start_fake_server
    def start():
        value = original()
        servers.append(value)
        return value
    monkeypatch.setattr(m, 'start_fake_server', start)
    def slow_validation(_response):
        # Accelerate the already-installed real work alarm at a stable stage.
        signal.setitimer(signal.ITIMER_REAL, .05)
        time.sleep(2)
    monkeypatch.setattr(m, 'validate_response', slow_validation)
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError, match='Notebook work deadline'):
        m.run_probe(m.parse_args())
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    assert signal.getsignal(signal.SIGALRM) == previous
    assert servers and not servers[0][1].is_alive()
    assert servers[0][0].socket.fileno() == -1
    result = json.loads((tmp_path/'raw/result.json').read_text())
    assert result['status'] != 'passed' and result['error_type'] == 'TimeoutError'
    assert (tmp_path/'raw/shutdown.json').exists()
