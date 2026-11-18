import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'chassis'))
from contingent_controller import compile_plan, select_branch


def outcome(cost, completion=1, recovery=0):
    return dict(completion=completion, remaining_actions=cost, recovery_actions=recovery)


def candidate(name, action, predictions, costs, metric='changed_cells'):
    return dict(id=name, action=action, metric=metric, predictions=predictions,
                assumptions=['Supplied worlds cover this bounded decision; costs are provisional'],
                continuations=[dict(action=a, outcomes={h: outcome(cost) for h, cost in world_costs.items()})
                               for a, world_costs in costs.items()])


def relevant_proposal():
    return dict(hypotheses={'a':.5,'b':.5}, candidates=[
        candidate('commit','RIGHT',{'a':0,'b':0}, {'LEFT':{'a':1,'b':5},'RIGHT':{'a':5,'b':1}}),
        candidate('probe','UP',{'a':1,'b':2}, {'LEFT':{'a':1,'b':5},'RIGHT':{'a':5,'b':1}})])


def test_probe_and_policy_only_observe_public_branch():
    plan = compile_plan(relevant_proposal())
    assert plan['selected']['id'] == 'probe'
    assert plan['selected']['cost'] == 2
    assert plan['evaluations'][0]['cost'] == 4  # hidden-world oracle would falsely give 2
    assert select_branch(plan, 1)['action'] == 'LEFT'
    assert select_branch(plan, 2)['action'] == 'RIGHT'
    assert select_branch(plan, 3) is None


def test_irrelevant_information_and_same_action_reject_probe():
    p = relevant_proposal()
    for c in p['candidates']:
        c['continuations'] = [dict(action='LEFT', outcomes={h:outcome(1 if c['id']=='commit' else 2) for h in p['hypotheses']})]
    plan = compile_plan(p)
    assert plan['selected']['id'] == 'commit'


def test_progress_that_teaches_beats_dedicated_probe():
    p = relevant_proposal()
    p['candidates'][1]['continuations'][0]['outcomes']['a'] = outcome(2)
    p['candidates'][1]['continuations'][1]['outcomes']['b'] = outcome(2)
    p['candidates'].append(candidate('progress','DOWN',{'a':1,'b':2},
        {'LEFT':{'a':1,'b':5},'RIGHT':{'a':5,'b':1}}))
    assert compile_plan(p)['selected']['id'] == 'progress'


def test_recovery_is_cost_and_completion_is_not_pixel_progress():
    p = relevant_proposal()
    for option in p['candidates'][1]['continuations']:
        for estimate in option['outcomes'].values(): estimate['recovery_actions'] = 10
    assert compile_plan(p)['selected']['id'] == 'commit'
    for option in p['candidates'][0]['continuations']:
        for estimate in option['outcomes'].values(): estimate['completion'] = .5
    assert compile_plan(p)['selected']['id'] == 'probe'


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),True,-1])
def test_invalid_estimates_rejected(bad):
    p = relevant_proposal();p['candidates'][0]['continuations'][0]['outcomes']['a']['completion']=bad
    with pytest.raises(ValueError): compile_plan(p)


def test_bounds_and_no_mutation():
    p=relevant_proposal(); original=copy.deepcopy(p);plan=compile_plan(p)
    assert p==original
    p['candidates'][0]['predictions']['a']=99
    assert plan['proposal']==original
    p['candidates'] *= 3
    with pytest.raises(ValueError):compile_plan(p)


async def make_agent(tmp_path):
    from nooa.unifiedllm import FakeLLMClient
    from solver_agent import MdArcSolverAgent
    a=MdArcSolverAgent(llm=FakeLLMClient(),run_dir=tmp_path,game_id='opaque',alias='game')
    state={'turn':0,'step':0,'levels_completed':0,'state':'NOT_FINISHED',
           'grid_rows':['00','00'],'available_actions':['UP','LEFT','RIGHT','DOWN']}
    (a.ipc_dir/'states.jsonl').write_text(json.dumps(state)+'\n')
    return a,state


def advance(a,state, after=None, **changes):
    state={**state,'turn':1,'step':1,'grid_rows':['10','00'],**changes}
    before={'grid':[[0,0],[0,0]],'levels_completed':0,'state':'NOT_FINISHED'}
    after=after or {'grid':[[1,0],[0,0]],'levels_completed':state['levels_completed'],'state':state['state']}
    row={'turn':0,'step':1,'action':'UP','before':before,'after':after}
    (a.ipc_dir/'transitions.jsonl').write_text(json.dumps(row)+'\n')
    with (a.ipc_dir/'states.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
    return row


@pytest.mark.asyncio
async def test_actual_methods_submit_one_branch_only_and_keep_evidence(tmp_path):
    from solver_agent import _ReturnResultSignal
    a,state=await make_agent(tmp_path)
    try:
        with pytest.raises(_ReturnResultSignal):a.submit_contingent_plan(relevant_proposal())
        advance(a,state)
        assert a._continue_contingent_plan({'game_states':['successor']})
        assert not a._continue_contingent_plan({'game_states':['duplicate']})
        actions=[json.loads(l) for l in (a.ipc_dir/'actions.jsonl').read_text().splitlines()]
        assert [x['actions'] for x in actions]==[['UP'],['LEFT']]
        assert a.decision_packet()['contingent_controller']['status']=='continuation_submitted'
    finally:await a.queue_manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['unknown','boundary','stale','guidance','illegal','known_bad','wrong_action','mismatch','unresolved_model','no_change'])
async def test_stop_and_existing_gates(tmp_path,reason):
    from solver_agent import _ReturnResultSignal
    a,state=await make_agent(tmp_path)
    try:
        with pytest.raises(_ReturnResultSignal):a.submit_contingent_plan(relevant_proposal())
        notification={'game_states':['successor']}
        if reason=='unknown':
            advance(a,state,{'grid':[[1,1],[1,0]],'levels_completed':0,'state':'NOT_FINISHED'},grid_rows=['11','10'])
        elif reason=='boundary':advance(a,state,levels_completed=1)
        elif reason=='stale':advance(a,state,step=2)
        elif reason=='illegal':advance(a,state,available_actions=['UP'])
        elif reason=='no_change':advance(a,state,{'grid':[[0,0],[0,0]],'levels_completed':0},grid_rows=['00','00'])
        else:advance(a,state)
        if reason=='guidance':notification['user_messages']=['stop']
        if reason=='known_bad':
            (a.ipc_dir/'observations.json').write_text(json.dumps({'known_from_here':{'LEFT':'fatal'}}))
        if reason=='wrong_action':
            p=a.ipc_dir/'transitions.jsonl';row=json.loads(p.read_text());row['action']='DOWN';p.write_text(json.dumps(row)+'\n')
        if reason=='unresolved_model':
            (a.ipc_dir/'model_predictions.jsonl').write_text(json.dumps({'turn':0,'step':0,'status':'committed'})+'\n')
            (a.ipc_dir/'step_checks.jsonl').write_text(json.dumps({'turn':0,'step':1,'status':'unresolved'})+'\n')
        if reason=='mismatch':
            (a.ipc_dir/'step_checks.jsonl').write_text(json.dumps({'turn':0,'step':1,'status':'mismatch'})+'\n')
        assert not a._continue_contingent_plan(notification)
        assert len((a.ipc_dir/'actions.jsonl').read_text().splitlines())==1
        assert not (a.ipc_dir/'contingent_pending.json').exists()
    finally:await a.queue_manager.shutdown()


def test_fast_successor_cannot_reopen_the_just_submitted_actor_turn():
    from types import SimpleNamespace
    from solver_agent import _wait_requires_submission, InvariantError
    agent=SimpleNamespace(_active_dispatch_turn=0,
        _latest_state=lambda: {'turn':1}, _last_submitted_turn=lambda:0)
    _wait_requires_submission(agent, {'kind':'WAIT'}, None)
    # The old mutable-latest-state check reproduced the spurious rejection.
    agent._active_dispatch_turn=None
    with pytest.raises(InvariantError):
        _wait_requires_submission(agent, {'kind':'WAIT'}, None)
    agent._active_dispatch_turn=0
    agent._last_submitted_turn=lambda:None
    with pytest.raises(InvariantError):
        _wait_requires_submission(agent, {'kind':'WAIT'}, None)
