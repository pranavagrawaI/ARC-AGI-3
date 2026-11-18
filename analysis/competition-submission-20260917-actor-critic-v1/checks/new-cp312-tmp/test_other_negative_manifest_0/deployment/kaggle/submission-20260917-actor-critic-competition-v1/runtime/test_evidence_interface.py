import importlib.util
import json
from pathlib import Path
import sys

import pytest
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'chassis'))
from decision_packet import transition_view
from prediction_feedback import PredictionFeedback


def test_actual_transition_exposes_existing_before_and_exact_diff():
    root = HERE.parents[3]
    p = root/'analysis/deployment-probes/20260907-designer-player-reasoning-recheck'
    row = json.loads(next(p.rglob('transitions.jsonl')).read_text().splitlines()[0])
    result = transition_view(row, True)
    assert result['kind'] == 'measured_transition'
    assert result['changed_cells'] == 52
    assert result['changed_bbox'] == [40,13,62,38]
    assert result['changes_omitted'] == 0
    assert len(result['changes']) == 52
    before, after = result['before_grid_rows'], result['after_grid_rows']
    cells = lambda grid: [(r,c) for r,line in enumerate(grid) for c,v in enumerate(line) if v in '01']
    assert cells(before) == cells(after)
    assert all(before[r][c] == format(a,'x') and after[r][c] == format(b,'x') for r,c,a,b in result['changes'])
    assert result['before_sha256'] != result['after_sha256']
    compact = transition_view(row)
    assert 'before_grid_rows' not in compact and 'after_grid_rows' not in compact


@pytest.mark.parametrize('before,after,count,omitted', [([[0]],[[0]],0,0),
    ([[0]],[[0,1]],None,0), ([[0]*100],[[1]*100],100,36)])
def test_transition_shape_and_output_bounds(before,after,count,omitted):
    row = dict(step=1,action='UP',before={'grid':before},after={'grid':after})
    snapshot = json.dumps(row)
    result = transition_view(row)
    assert result['changed_cells'] == count
    assert result['changes_omitted'] == omitted
    assert len(result['changes']) <= 64
    assert json.dumps(row) == snapshot


def test_preflight_uses_runtime_json_contract_without_committing(tmp_path):
    helpers = tmp_path/'helpers'; helpers.mkdir()
    source = 'def encode(g): return {"cells": [(0,0)]}\ndef predict(z,a): return {"n":len(set(z["cells"]))}\n'
    (helpers/'world_model.py').write_text(source)
    feedback = PredictionFeedback(tmp_path, helpers, lambda _:None)
    state = dict(turn=0,step=0,grid=[[0]])
    direct = {}; exec(source,direct)
    assert direct['predict'](direct['encode']([[0]]),'UP') == {'n':1}
    preflight = feedback.preflight(state,['UP'])
    actual = feedback.prepare(state,['UP'])
    assert preflight['status'] == 'unavailable'
    assert preflight['reason'] == actual['reason'] == "TypeError: unhashable type: 'list'"
    # Batch qualification remains unearned; actual single-action freedom is
    # exercised through the real worker in test_reasoning.py.
    assert feedback.batch_error(actual, {}) is not None
    assert not feedback.commits.exists() and not feedback.results.exists()
    (helpers/'world_model.py').write_text(source.replace('set(z["cells"])','set(map(tuple,z["cells"]))'))
    good = feedback.preflight(state,['UP'])
    assert good['status'] == 'ready'
    assert good['model_sha256'] != preflight['model_sha256']
    assert not feedback.commits.exists()


def test_preflight_bounds_and_missing_state(tmp_path):
    f = PredictionFeedback(tmp_path,tmp_path,lambda _:None)
    assert f.preflight(None,['UP'])['status'] == 'unavailable'
    for actions in ([], 'UP', ['UP']*21):
        with pytest.raises(ValueError): f.preflight({},actions)


def test_preflight_reports_canonical_input_contract(tmp_path):
    helpers = tmp_path/'helpers'; helpers.mkdir()
    (helpers/'world_model.py').write_text(
        'def encode(g): return {"shape": list(g.shape)}\n'
        'def predict(z,a): return z\n')
    feedback = PredictionFeedback(tmp_path, helpers, lambda _: None)
    result = feedback.preflight({'turn': 0, 'step': 0, 'grid_rows': ['0f', '10']}, ['UP'])
    assert result['status'] == 'ready'
    assert result['input_contract']['encode_input'] == 'numpy.ndarray[int] (detached)'
    assert result['input_contract']['shape'] == [2, 2]
    assert result['input_contract']['dtype'] == 'int64'
    assert result['input_contract']['predict_input'].startswith('JSON-normalized')


@pytest.mark.asyncio
async def test_region_summary_and_bounded_render(tmp_path):
    from nooa.unifiedllm import FakeLLMClient
    from solver_agent import MdArcSolverAgent
    agent = MdArcSolverAgent(llm=FakeLLMClient(), run_dir=tmp_path, game_id='opaque', alias='game')
    try:
        rows = ['0123', '3401', '1111']
        summary = agent.grid_summary(rows)
        assert summary['shape'] == [3, 4]
        assert summary['color_counts']['1'] == 6
        assert summary['color_bboxes']['1'] == [0, 0, 2, 3]
        rendered = agent.render_grid(rows, max_chars=20)
        assert '[TRUNCATED:' in rendered

        transition = {
            'step': 1, 'turn': 0, 'action': 'UP',
            'before': {'grid': [[0, 0, 0], [0, 0, 0], [0, 0, 0]]},
            'after': {'grid': [[0, 1, 0], [0, 0, 0], [0, 0, 0]]},
        }
        (agent.ipc_dir/'transitions.jsonl').write_text(json.dumps(transition) + '\n')
        region = agent.changed_region(padding=1)
        assert region['region_bbox'] == [0, 0, 2, 3]
        assert region['before_region_rows'] == ['000', '000']
        assert region['after_region_rows'] == ['010', '000']
    finally:
        await agent.queue_manager.shutdown()


@pytest.mark.asyncio
async def test_real_agent_methods_and_legacy_trajectory(tmp_path):
    from nooa.unifiedllm import FakeLLMClient
    from solver_agent import MdArcSolverAgent
    agent = MdArcSolverAgent(llm=FakeLLMClient(),run_dir=tmp_path,game_id='opaque',alias='game')
    try:
        assert agent.last_transition() is None
        assert agent.preflight_model(['UP'])['status'] == 'unavailable'
        (agent._states_path).write_text(json.dumps({'turn': 0, 'step': 0,
            'grid_rows': ['01', '10'], 'state': 'NOT_FINISHED'}) + '\n')
        assert agent.model_grid().dtype == np.int64
        assert agent.model_grid().tolist() == [[0, 1], [1, 0]]
        row = dict(step=1,turn=0,action='UP',before={'grid':[[0,1]]},after={'grid':[[1,0]],'levels_completed':0})
        (agent.ipc_dir/'transitions.jsonl').write_text(json.dumps(row)+'\n')
        view = agent.last_transition(True)
        assert view['before_grid_rows'] == ['01'] and view['after_grid_rows'] == ['10']
        old = agent.trajectory()
        assert len(old)==1 and old[0]['before_grid_rows']==['01'] and old[0]['grid_rows']==['10']
        assert agent.decision_packet()['last_transition']['changed_cells'] == 2
        procedure = agent.save_procedure(
            'route.py', 'def next_action(state):\n    return "UP"\n',
            'movement rule remains verified', 'advance one cell',
            'stop on any prediction mismatch', ['transition-step-1'])
        assert procedure['filename'] == 'route.py'
        assert procedure['stop_condition'] == 'stop on any prediction mismatch'
        assert (agent.ipc_dir/'procedures.jsonl').exists()
        assert 'self.h.route' in agent.load_helpers()
    finally:
        await agent.queue_manager.shutdown()
