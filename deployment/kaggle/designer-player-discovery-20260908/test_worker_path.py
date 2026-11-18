"""Candidate adapter + shared transport + real worker subprocess qualification."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'deployment/kaggle/tests'),str(ROOT/'deployment/kaggle')]


def test_packaged_path_has_exact_identity_and_continuity(monkeypatch,tmp_path):
    import test_my_agent
    import test_nooa_join
    import test_join_qualification
    spec=importlib.util.spec_from_file_location('runtime.local_solver',HERE/'shared_local_solver.py')
    runtime=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,'runtime.local_solver',runtime)
    spec.loader.exec_module(runtime)
    monkeypatch.setattr(test_my_agent,'AGENT_PATH',HERE/'adapter.py')
    monkeypatch.setattr(test_nooa_join,'CANDIDATE',HERE)
    monkeypatch.setenv('NOOA_JOIN_ROOT',str(HERE))
    monkeypatch.setenv('NOOA_AGENT_COMMAND',json.dumps([sys.executable,str(HERE/'stub_agent.py')]))
    monkeypatch.setenv('ARC_SOLVER_ARM','mindset')
    monkeypatch.setenv('TYCHO_NOOA_WORKER_TIMEOUT_SECONDS','15')
    monkeypatch.setenv('LITELLM_LOCAL_MODEL_COST_MAP','True')
    test_join_qualification.test_ordinary_play_holds_continuity_across_many_turns(tmp_path,None,monkeypatch)
    metadata=[json.loads(p.read_text()) for p in tmp_path.rglob('worker-metadata.json')]
    assert metadata and all(m['chassis']=='designer-player-reasoning-20260907' for m in metadata)
    launches=[json.loads(p.read_text()) for p in tmp_path.rglob('launch.json')]
    assert launches and all(m['solver_arm']=='mindset' for m in launches)


@pytest.mark.asyncio
async def test_real_actor_methods_are_callable(tmp_path):
    sys.path.insert(0,str(HERE/'chassis'))
    from nooa.unifiedllm import FakeLLMClient
    from solver_agent import MdArcSolverAgent
    agent=MdArcSolverAgent(llm=FakeLLMClient(),run_dir=tmp_path,game_id='opaque',alias='the game')
    assert agent.decision_packet()['schema']=='decision-packet-v1'
    (tmp_path/'ipc/states.jsonl').write_text(json.dumps({'turn':0,'step':0,'levels_completed':0,'grid_rows':['01'],'available_actions':['UP']})+'\n')
    record=agent.set_experiment('contact?', ['contact matters'], 'UP', {'change':'support','none':'inconclusive'})
    assert agent.decision_packet()['experiment']['id']==record['id']
    await agent.queue_manager.shutdown()
