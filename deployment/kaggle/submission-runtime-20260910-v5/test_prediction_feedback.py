import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('feedback', HERE / 'chassis/prediction_feedback.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

SOURCE = '''import numpy as np
def encode(grid):
    return {"x": int(np.where(grid == 12)[1].min()), "right": int(np.where(grid == 11)[1].max())}
def predict(z, action):
    return {"x": z["x"] + 1, "right": z["right"] - 2}
'''

def setup(tmp_path):
    ipc, helpers = tmp_path / 'ipc', tmp_path / 'helpers'
    ipc.mkdir(); helpers.mkdir()
    (helpers / 'world_model.py').write_text(SOURCE)
    return m.PredictionFeedback(ipc, helpers, lambda source: None)


def state():
    return {'turn': 0, 'step': 0, 'grid': [[12, 0, 11, 11]], 'levels_completed': 0}


def transition(feedback, step=1, after=None):
    m.write_row(feedback.ipc / 'transitions.jsonl', {'step': step, 'turn': 0, 'action': 'RIGHT',
        'before': {'grid': state()['grid']},
        'after': {'grid': after or [[0, 12, 0, 11]], 'levels_completed': 0}})


def test_frozen_encoder_exposes_error_even_after_rewrite(tmp_path):
    f = setup(tmp_path)
    f.commit(f.prepare(state(), ['RIGHT']))
    (f.helpers / 'world_model.py').write_text(SOURCE.replace('z["right"] - 2', 'z["right"]'))
    transition(f)
    r = f.observe()
    assert r['status'] == 'mismatch'
    assert r['differences'][0]['fields']['right']['predicted'] == 1
    assert r['differences'][0]['fields']['right']['observed'] == 3
    assert f.batch_error(f.prepare(state(), ['RIGHT', 'RIGHT']), r)


def test_same_version_match_earns_batch_but_changed_source_does_not(tmp_path):
    f = setup(tmp_path)
    (f.helpers / 'world_model.py').write_text(SOURCE.replace('z["right"] - 2', 'z["right"]'))
    f.commit(f.prepare(state(), ['RIGHT', 'RIGHT']))
    transition(f)
    r = f.observe()
    assert r['checked'] == 1  # an unexecuted tail is not scored
    assert r['batch_allowed']
    assert f.batch_error(f.prepare(state(), ['RIGHT']), r) is None
    (f.helpers / 'world_model.py').write_text(SOURCE + '\n# a different version\n')
    assert f.batch_error(f.prepare(state(), ['RIGHT']), r)


def test_empty_model_and_bad_transition_cannot_pass(tmp_path):
    f = setup(tmp_path)
    f.commit(f.prepare(state(), ['RIGHT', 'RIGHT']))
    transition(f, step=2)
    assert f.observe()['status'] == 'unavailable'
    (f.helpers / 'world_model.py').write_text('def encode(g): return {}\ndef predict(z,a): return {}')
    assert f.prepare(state(), ['RIGHT'])['status'] == 'unavailable'


def test_replay_real_band_contradiction(tmp_path):
    f = setup(tmp_path)
    root = HERE.parents[2]
    raw = root / 'analysis/deployment-probes/20260905-q6-quick-host/raw'
    helper = next(raw.rglob('helpers/worldmodel.py'))
    (f.helpers / 'world_model.py').write_text(helper.read_text())
    t = json.loads(next(raw.rglob('ipc/transitions.jsonl')).read_text().splitlines()[0])
    before = {**t['before'], 'turn': t['turn'], 'step': t['step'] - 1}
    f.commit(f.prepare(before, [t['action']]))
    m.write_row(f.ipc / 'transitions.jsonl', t)
    r = f.observe()
    assert r['status'] == 'mismatch'
    assert r['differences'][0]['fields']['band_right']['predicted'] == 52
    assert r['differences'][0]['fields']['band_right']['observed'] == 54


def test_partial_append_is_pending_but_corrupt_committed_row_is_error(tmp_path):
    import pytest
    f = setup(tmp_path)
    f.commit(f.prepare(state(), ['RIGHT']))
    path = f.ipc / 'transitions.jsonl'
    path.write_text('{"step":1')
    assert f.observe()['status'] == 'pending'
    path.write_text('{"step":1\n')
    with pytest.raises(json.JSONDecodeError):
        f.observe()
