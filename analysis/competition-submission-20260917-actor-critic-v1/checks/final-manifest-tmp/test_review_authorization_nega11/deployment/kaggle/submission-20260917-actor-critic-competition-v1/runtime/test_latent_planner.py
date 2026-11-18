"""Contract checks against literal graph results and the real agent surface."""
import asyncio
import json
import os
from pathlib import Path
import sys
import threading

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'chassis'))
import latent_planner as p
from prediction_feedback import PredictionFeedback

SOURCE = '''def encode(grid): return {"x": int(grid[0,0])}
def predict(z, action):
    z["x"] += 1 if action == "RIGHT" else 10
    return z
'''
TARGET = dict(path=['x'], op='eq', value=3)
STATE = dict(turn=0, step=0, levels_completed=0, grid=[[0]],
             available_actions=['RIGHT','LEFT'], state='NOT_FINISHED')


def search(tmp_path, source=SOURCE, target=None, actions=None, limits=None):
    (tmp_path/'world_model.py').write_text(source)
    return p.run(STATE, tmp_path, lambda s: None, target or TARGET,
                 actions if actions is not None else ['RIGHT'], limits)


def test_corridor_replayed_detached_and_no_fabricated_host_outcomes(tmp_path):
    r = search(tmp_path)
    assert r['status'] == 'path_found', r
    assert r['plan'] == ['RIGHT']*3
    assert r['predicted_latents'] == [{'x':1},{'x':2},{'x':3}]
    assert r['initial_latent'] == {'x':0}
    assert 'predicted_path' not in r
    assert r['child_reaped']
    with pytest.raises(ProcessLookupError): os.kill(r['child_pid'],0)


def test_depth_is_limited_not_exhausted(tmp_path):
    assert search(tmp_path, limits={'depth':1})['status'] == 'limited'


def test_closed_graph_is_exhausted(tmp_path):
    r = search(tmp_path, source='def encode(g): return {"x":0}\ndef predict(z,a): return z')
    assert r['status'] == 'model_exhausted' and r['complete']
    assert r['seen_states'] == 1


@pytest.mark.parametrize('source,stage', [
    ('raise ValueError("load")\n'+SOURCE, 'load'),
    ('def encode(g): raise ValueError("encoding")\ndef predict(z,a): return z', 'encode'),
    ('def encode(g): return {"x":0}\ndef predict(z,a): raise ValueError("unknown")', 'predict'),
    ('def encode(g): return {"x":0}\ndef predict(z,a): return {"x":float("nan")}', 'predict'),
])
def test_errors_never_exhaust_graph(tmp_path, source, stage):
    r=search(tmp_path,source)
    assert r['status']=='model_error' and r['stage']==stage, r


@pytest.mark.parametrize('source', [
    'while True: pass\n'+SOURCE,
    'def encode(g):\n while True: pass\ndef predict(z,a): return z',
    'def encode(g): return {"x":0}\ndef predict(z,a):\n while True: pass',
    # A model swallowing the per-callback alarm still meets parent containment.
    'def encode(g):\n while True:\n  try:\n   while True: pass\n  except BaseException: pass\ndef predict(z,a): return z',
])
def test_hangs_are_killed_and_reaped(tmp_path, source):
    r=search(tmp_path, source, limits={'callback_seconds':.03,'wall_seconds':.2})
    assert r['status']=='limited', r
    assert r['elapsed_seconds'] < 1
    with pytest.raises(ProcessLookupError): os.kill(r['child_pid'],0)


def test_mutation_cannot_make_left_mean_eleven(tmp_path):
    r=search(tmp_path,target=dict(path=['x'],op='eq',value=11),actions=['RIGHT','LEFT'])
    assert r['status']=='path_found', r
    assert len(r['plan'])==2
    assert r['predicted_latents'][-1]=={'x':11}


def test_stateful_model_fails_fresh_replay(tmp_path):
    source='''counter=0
def encode(g): return {"x":0}
def predict(z,a):
 global counter
 counter+=1
 return {"x":counter}
'''
    r=search(tmp_path,source,target=dict(path=['x'],op='eq',value=2),actions=['RIGHT','LEFT'])
    assert r['status']=='model_error' and r['stage']=='replay_predict', r


@pytest.mark.parametrize('limits,reason', [({'generated_states':1},'generated_states'),({'frontier':1},'frontier')])
def test_resource_caps_bound_generation(tmp_path, limits, reason):
    r=search(tmp_path,target=dict(path=['x'],op='eq',value=99),actions=['RIGHT','LEFT'],limits=limits)
    assert r['status']=='limited' and r['reason']==reason, r
    assert r['generated_states'] <= limits.get('generated_states',2)


@pytest.mark.parametrize('target', [dict(path=['missing'],op='eq',value=0),
    {'any':[dict(path=['x'],op='eq',value=0),dict(path=['missing'],op='eq',value=0)]}])
def test_missing_target_fields_never_silently_false(tmp_path,target):
    assert search(tmp_path,target=target)['status']=='invalid_request'


def test_unavailable_off_main_thread_does_not_execute_helper(tmp_path):
    results=[]
    t=threading.Thread(target=lambda: results.append(search(tmp_path,source='while True: pass')))
    t.start();t.join(1)
    assert not t.is_alive()
    assert results[0]['status']=='unavailable'


def test_already_satisfied_and_click_candidates(tmp_path):
    assert search(tmp_path,target=dict(path=['x'],op='eq',value=0))['status']=='already_satisfied'
    state={**STATE,'available_actions':['CLICK']}
    r=p.run(state,tmp_path,lambda s:None,TARGET)
    assert r['status']=='invalid_request'
    r=p.run(state,tmp_path,lambda s:None,TARGET,['CLICK 0 0'])
    assert r['status'] != 'invalid_request'


def test_shared_selector_preparation_and_search_use_same_model(tmp_path):
    (tmp_path/'worldmodel.py').write_text('def encode(g): return {"x":100}\ndef predict(z,a): return z')
    r=search(tmp_path)
    f=PredictionFeedback(tmp_path,tmp_path,lambda s:None)
    prepared=f.prepare(STATE,['RIGHT'])
    assert r['snapshot']['model_sha256']==prepared['model_sha256']
    assert r['initial_latent']==prepared['before']=={'x':0}


def test_real_agent_exploration_and_search_join(tmp_path, monkeypatch):
    from solver_agent import MdArcSolverAgent, _ReturnResultSignal
    from nooa.unifiedllm import FakeLLMClient
    async def check():
        agent=MdArcSolverAgent(llm=FakeLLMClient(),run_dir=tmp_path,
                              game_id='opaque',alias='game',enable_summarizer=False)
        state=dict(STATE)
        def publish():
            agent._states_path.write_text(json.dumps(state)+'\n')
        try:
            publish()
            # Ordinary exploration has no helper, experiment or winning goal.
            with pytest.raises(_ReturnResultSignal):
                agent.submit_actions(['RIGHT'], rationale='observe whether a local change occurs')
            state.update(turn=1,step=1,grid=[[1]])
            publish()
            agent.write_helper('world_model.py',SOURCE)
            r=agent.plan_actions(TARGET,['RIGHT'])
            assert r['status']=='path_found',r
            assert 'plan_actions(target)' in agent._helpers_status_line()
            path=[{'expect':'move','z':z} for z in r['predicted_latents']]
            assert agent.submit_actions(r['plan'],'conditional target',path,search_id=r['search_id']).startswith('REJECTED:')
            # An externally observed transition qualifies this exact helper.
            old={**state,'turn':0,'step':0,'grid':[[0]]}
            agent._model_feedback.commit(agent._model_feedback.prepare(old,['RIGHT']))
            (agent.ipc_dir/'transitions.jsonl').write_text(json.dumps(dict(
                turn=0,step=1,action='RIGHT',before=old,after=state))+'\n')
            assert agent.prediction_feedback()['batch_allowed']
            # Mutating the returned plan does not mutate host authority.
            r['plan'][0]='LEFT'
            assert agent.submit_actions(r['plan'],'tampered',path,search_id=r['search_id']).startswith('REJECTED:')
            r['plan'][0]='RIGHT'
            state['grid']=[[2]];publish()
            assert agent.submit_actions(r['plan'],'stale grid',path,search_id=r['search_id']).startswith('REJECTED:')
            state['grid']=[[1]];publish()
            agent.write_helper('world_model.py',SOURCE+'\n# changed version\n')
            assert agent.submit_actions(r['plan'],'stale helper',path,search_id=r['search_id']).startswith('REJECTED:')
            agent.write_helper('world_model.py',SOURCE)
            with pytest.raises(_ReturnResultSignal):
                agent.submit_actions(r['plan'],'conditional local target',path,search_id=r['search_id'])
            action=json.loads(agent._actions_path.read_text().splitlines()[-1])
            assert action['search_id']==r['search_id']
        finally:
            await agent.queue_manager.shutdown()
    asyncio.run(check())
