"""Behavioral qualification imports only extracted notebook payload code."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import os
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent
PAYLOAD=Path(os.environ.get('Q6_TEST_PAYLOAD', str(HERE/'extracted')))
ROOT=HERE.parents[2]
sys.path[:0]=[str(PAYLOAD),str(PAYLOAD/'chassis'),str(ROOT/'deployment/kaggle/tests'),str(ROOT/'deployment/kaggle')]
from host_contract import HostEvidence, ObservationRecord
from state_delivery import StateDelivery
from solver_agent import MdArcSolverAgent
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def load(name):
    spec=importlib.util.spec_from_file_location('quick_'+name,PAYLOAD/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


RAW=ROOT/'analysis/deployment-probes/20260905-q6-speed-probe/raw/solver-workspaces/ls20-c8bieiv8/ipc'


def test_real_horizontal_motion_does_not_stop_batch(tmp_path):
    ipc=tmp_path/'ipc';ipc.mkdir();host=HostEvidence(ipc)
    transitions=rows(RAW/'transitions.jsonl')
    for t in transitions[:5]:
        host.begin({'turn':t['turn'],'predicted_path':[{'expect':'move'}]})
        stopped=host.observe(t['before'],t['after'],t['action'],3)
        assert not stopped
        assert host.results[-1]['outcome']=='move'
    assert [45,46,47,48,49]==host.record.hud_rows[:5]
    assert host.record.totals['truncations']==0
    assert host.record.known_from_here(transitions[4]['after']['grid'],['LEFT'])=={}
    assert not (ipc/'batch_stops.jsonl').exists()


@pytest.mark.parametrize('state,level,expected',[('NOT_FINISHED',0,'no_impact'),('GAME_OVER',0,'terminal'),('WIN',1,'terminal'),('NOT_FINISHED',1,'level_up')])
def test_real_stop_causes_remain_authoritative(tmp_path,state,level,expected):
    ipc=tmp_path/'ipc';ipc.mkdir();host=HostEvidence(ipc)
    before={'grid':[[0]],'state':'NOT_FINISHED','levels_completed':0,'available_actions':['UP']}
    after={**before,'state':state,'levels_completed':level}
    host.begin({'turn':0})
    assert host.observe(before,after,'UP',2)
    assert host.results[-1]['outcome']==expected
    assert rows(ipc/'batch_stops.jsonl')[0]['skipped_actions']==2


def test_inert_action_memory_uses_entire_grid():
    record=ObservationRecord()
    for i in range(5):record.observe([[i]],[[i+1]])
    record.note([[8]],'UP','no_impact')
    assert record.known_from_here([[8]],['UP'])=={'UP':'no_impact'}
    assert record.known_from_here([[9]],['UP'])=={}


def test_history_contains_before_and_after_and_handles_partial_tail(tmp_path):
    (tmp_path/'transitions.jsonl').write_bytes((RAW/'transitions.jsonl').read_bytes()+b'{')
    actor=MdArcSolverAgent.__new__(MdArcSolverAgent)
    actor.ipc_dir=tmp_path;actor._states_path=RAW/'states.jsonl'
    history=actor.trajectory(include_events=True)
    assert len(history)==10 and actor.trajectory(last_n=0)==[]
    assert history[4]['step']==5 and history[4]['diff_pixels']==52
    assert history[0]['before_grid_rows']==[''.join(format(v,'x') for v in row) for row in rows(RAW/'transitions.jsonl')[0]['before']['grid']]
    assert not history[0]['animation_available']
    assert 'grid_rows' not in actor.trajectory(include_grids=False)[0]
    assert len(actor.trajectory(per_step=False))==6


def test_state_delivery_replaces_stale_and_keeps_new_user_message():
    delivery=StateDelivery();latest={'turn':4,'step':8}
    result=delivery.prepare({'game_states':[json.dumps({'turn':3,'step':5})]},latest)
    assert json.loads(result['game_states'][0])==latest
    assert delivery.prepare({'game_states':['stale']},latest) is None
    assert delivery.prepare({'user_messages':['new guidance']},latest)['user_messages']==['new guidance']


def response(code):
    return LLMResponse(raw_response=None,content='',tool_calls=[ToolCall(id='one',name='execute_python',arguments=json.dumps({'code':code}))],finish_reason='tool_calls',assistant_message={'role':'assistant','content':''})


def test_real_actor_and_summarizer_calls_are_both_recorded(tmp_path,monkeypatch):
    monkeypatch.setenv('ARC_STALL_REVIEW','0')
    async def run():
        fake=FakeLLMClient(scripted_responses=[response("self.submit_actions(['UP'], rationale='predict: move')"),LLMResponse(raw_response=None,content='"Preserved summary"',tool_calls=[],finish_reason='stop',assistant_message={'role':'assistant','content':'"Preserved summary"'})])
        actor=MdArcSolverAgent(llm=fake,run_dir=tmp_path)
        await asyncio.sleep(0)
        state={'turn':0,'step':0,'grid_rows':['00','00'],'state':'NOT_FINISHED','level':0,'levels_completed':0,'available_actions':['UP'],'action_results':[],'note':''}
        (tmp_path/'ipc/states.jsonl').write_text(json.dumps(state)+'\n')
        try:
            await actor.handle({'user_messages':['Make one legal move.']})
            summary=await actor._summarizers[0].summarize('UP was chosen after inspecting the current state.',target_chars=100)
            assert 'Preserved summary' in summary
            calls=rows(tmp_path/'ipc/model_calls.jsonl')
            assert len(calls)==2 and [r['summarizer'] for r in calls]==[False,True]
            assert fake.call_count==2
        finally:
            await actor.queue_manager.shutdown()
    asyncio.run(run())


def test_accounting_reconciles_episodes_and_rejects_missing_evidence(tmp_path,monkeypatch):
    monkeypatch.setenv('TYCHO_PUBLIC_GAME_IDS','ls20')
    probe=load('nooa_public_probe');runtime=load('exact_request_outer_runtime')
    import test_q6_actor_call_evidence as support
    for suffix,turns in [('a',[0,0,1]),('b',[0,1])]:
        support.write_session(tmp_path,game='ls20',suffix=suffix,turns=turns,actions=['UP'])
    sessions=probe.summarize_nooa_sessions(tmp_path)
    assert runtime.expected_launcher_solver_requests({'solver_sessions':sessions})==5
    warm={'chat_completion_accesses':1,'successful_chat_completion_accesses':1}
    post={'chat_completion_accesses':6,'successful_chat_completion_accesses':6,'successful_freeze_gc_accesses':1,'ready_marker_count':1}
    assert runtime.reconcile_launcher_request_count({'solver_sessions':sessions},warm,post)['whole_model_request_count_verified']
    post['chat_completion_accesses']=7
    with pytest.raises(RuntimeError):runtime.reconcile_launcher_request_count({'solver_sessions':sessions},warm,post)
    (tmp_path/'ls20-a/ipc/model_calls.jsonl').unlink()
    with pytest.raises(RuntimeError,match='evidence is absent'):probe.summarize_nooa_sessions(tmp_path)


def test_old_probe_metrics_distinguish_cap_tail_from_host_skips():
    probe=load('nooa_public_probe')
    # Old artifact lacks batch_stops; supply a temporary copy in other tests for new logs.
    metrics=probe._quick_metrics(RAW.parent,rows(RAW/'actions.jsonl'))
    assert metrics['actor_batches']==6 and metrics['executed_actions']==10
    assert metrics['eligible_detector_attempts']==0




def test_summary_reasoning_is_recorded_and_latches_abort(tmp_path):
    async def run():
        fake=FakeLLMClient(scripted_responses=[LLMResponse(raw_response=None,content='"summary"',tool_calls=[],finish_reason='stop',reasoning='unexpected reasoning',assistant_message={'role':'assistant','content':'"summary"'})])
        actor=MdArcSolverAgent(llm=fake,run_dir=tmp_path)
        await asyncio.sleep(0)
        try:
            with pytest.raises(Exception):
                await actor._summarizers[0].summarize('history',target_chars=50)
            assert (tmp_path/'ipc/ABORT').exists()
            calls=rows(tmp_path/'ipc/model_calls.jsonl')
            assert calls[0]['summarizer'] and calls[0]['reasoning_tokens']>0
        finally:await actor.queue_manager.shutdown()
    asyncio.run(run())


def test_new_stop_metrics_distinguish_skips_and_budget_tail(tmp_path):
    import shutil
    shutil.copytree(RAW,tmp_path/'ipc')
    (tmp_path/'ipc/batch_stops.jsonl').write_text(json.dumps({'turn':2,'step':5,'reason':'no_impact','skipped_actions':3})+'\n')
    metrics=load('nooa_public_probe')._quick_metrics(tmp_path,rows(tmp_path/'ipc/actions.jsonl'))
    assert metrics['host_skipped_actions']==3 and metrics['unexecuted_tail']==3
