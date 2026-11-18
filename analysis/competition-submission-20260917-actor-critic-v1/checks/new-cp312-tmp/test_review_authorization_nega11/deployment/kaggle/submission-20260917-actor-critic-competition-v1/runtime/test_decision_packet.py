import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'chassis'))
from decision_packet import packet, record_experiment


def test_packet_preserves_raw_and_marks_old_question(tmp_path):
    state = dict(turn=2, step=3, levels_completed=0, grid_rows=['001'], available_actions=['UP'])
    snapshot = json.loads(json.dumps(state))
    outcomes = {'target':'any_change', 'metric':'grid_changed', 'predictions':{'contact matters':True, 'no effect':False},
                'updates':{'contact matters':'support contact here', 'no effect':'reject contact here'},
                'otherwise':'revise the measurement'}
    exp = record_experiment(tmp_path, state, 'Does contact change the target?',
                            ['contact matters', 'no effect'], 'UP', outcomes)
    transitions = [dict(step=3, action='UP', before={'grid':[[0,0,1]]}, after={'grid':[[0,1,0]], 'levels_completed':0})]
    result = packet(state, transitions, {'status':'mismatch'}, {}, tmp_path, tmp_path)
    assert result['experiment']['id'] == exp['id']
    assert result['experiment_from_current_observation']
    assert result['recent_transitions'][0]['changed_bbox'] == [0,1,0,2]
    assert result['recent_transitions'][0]['changed_cells'] == 2
    assert 'grid_rows' not in result['state']
    assert state == snapshot
    later = packet({**state, 'step':4}, transitions, {}, {}, tmp_path, tmp_path)
    assert not later['experiment_from_current_observation']
    assert later['experiment']['step'] == 3


def test_unknown_shape_is_not_zero_change(tmp_path):
    result = packet({}, [dict(step=1, action='UP', before={'grid':[[0]]}, after={'grid':[[0,1]]})], {}, {}, tmp_path, tmp_path)
    assert result['recent_transitions'][0]['shape_changed']
    assert result['recent_transitions'][0]['changed_cells'] is None


@pytest.mark.parametrize('question,hypotheses,outcomes', [('', ['a'], {'a':'b','c':'d'}), ('why', [], {'a':'b','c':'d'}), ('why',['a'],{'a':'b'}), ('x'*5000,['a'],{'a':'b','c':'d'})])
def test_bad_records_do_not_append(tmp_path,question,hypotheses,outcomes):
    with pytest.raises(ValueError):
        record_experiment(tmp_path, {'turn':0,'step':0}, question,hypotheses,'UP',outcomes)
    assert not (tmp_path/'decision_experiments.jsonl').exists()
