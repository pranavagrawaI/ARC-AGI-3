import importlib.util
import json
import signal
import time
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[3]/'deployment/kaggle/q6-prediction-feedback-20260906/chassis/prediction_feedback.py'
spec = importlib.util.spec_from_file_location('bounded_feedback', SOURCE)
feedback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feedback)


def make(tmp_path, source):
    helpers = tmp_path/'helpers'
    helpers.mkdir()
    (helpers/'world_model.py').write_text(source)
    return feedback.PredictionFeedback(tmp_path, helpers, lambda source: None)


@pytest.mark.parametrize('source', [
    'while True: pass\ndef encode(g): return {"x":1}\ndef predict(z,a): return z',
    'def encode(g):\n while True: pass\ndef predict(z,a): return z',
    'def encode(g): return {"x":1}\ndef predict(z,a):\n while True: pass',
])
def test_hanging_helper_becomes_unavailable_and_restores_alarm(tmp_path, source):
    model = make(tmp_path, source)
    handler = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    result = model.prepare({'turn':0,'step':0,'grid':[[1]]}, ['UP'])
    assert result['status'] == 'unavailable'
    assert 'HelperTimeoutError' in result['reason']
    assert time.monotonic() - started < 3
    assert signal.getsignal(signal.SIGALRM) == handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


def test_successor_dependent_encoder_loop_cannot_wedge_feedback(tmp_path):
    model = make(tmp_path, 'def encode(g):\n if int(g.sum())==2:\n  while True: pass\n return {"x":1}\ndef predict(z,a): return z')
    model.commit(model.prepare({'turn':0,'step':0,'grid':[[1]]}, ['UP']))
    (tmp_path/'transitions.jsonl').write_text(json.dumps({'turn':0,'step':1,'action':'UP','after':{'grid':[[2]]}})+'\n')
    result = model.observe()
    assert result['status'] == 'unavailable' and not result['batch_allowed']
    assert 'HelperTimeoutError' in result['reason']


def test_existing_alarm_is_preserved_and_model_not_executed(tmp_path):
    model = make(tmp_path, 'raise AssertionError("should not execute")')
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        result = model.prepare({'turn':0,'step':0,'grid':[[1]]}, ['UP'])
        assert 'another alarm is active' in result['reason']
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 59
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@pytest.mark.parametrize('exception', ['SystemExit', 'KeyboardInterrupt'])
def test_helper_base_exception_does_not_exit_actor(tmp_path, exception):
    model = make(tmp_path, f'raise {exception}()')
    result = model.prepare({'turn':0,'step':0,'grid':[[1]]}, ['UP'])
    assert result['status'] == 'unavailable' and exception in result['reason']


def test_postrun_final_prefix_replay_preserves_raw_and_is_not_actor_feedback(tmp_path):
    import replay_final_feedback
    session = tmp_path/'session'
    ipc = session/'ipc'
    ipc.mkdir(parents=True)
    model = make(ipc, 'def encode(g): return {"x":int(g[0,0])}\ndef predict(z,a): return {"x":z["x"]+1}')
    model.commit(model.prepare({'turn':0,'step':0,'grid':[[1]]}, ['UP','UP','UP']))
    (ipc/'transitions.jsonl').write_text(json.dumps({'turn':0,'step':1,'action':'UP','after':{'grid':[[2]]}})+'\n')
    before = {p.name:p.read_bytes() for p in ipc.iterdir() if p.is_file()}
    result = replay_final_feedback.replay(session, SOURCE.parent)
    assert result['generated_post_run'] and not result['delivered_to_actor']
    assert result['feedback']['status'] == 'matched' and result['feedback']['checked'] == 1
    assert {p.name:p.read_bytes() for p in ipc.iterdir() if p.is_file()} == before
