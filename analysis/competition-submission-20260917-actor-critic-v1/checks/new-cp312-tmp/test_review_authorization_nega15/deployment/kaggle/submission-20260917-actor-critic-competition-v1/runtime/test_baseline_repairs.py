"""Regression checks for the exact endurance successor, without model/GPU calls."""
import importlib.util
import os
import asyncio
import json
from pathlib import Path

import pytest

HERE = Path(os.environ.get('REPAIR_TEST_RUNTIME', Path(__file__).resolve().parent))


def module(name):
    spec = importlib.util.spec_from_file_location('repair_' + name, HERE/'chassis'/f'{name}.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize('before,after', [
    ([[1]], [[1], [2]]), ([[1], [2]], [[1]]),
    ([[1]], [[1, 2]]), ([[1, 2]], [[1]]),
])
def test_shape_change_is_not_inert_and_does_not_poison_row_history(tmp_path, before, after):
    host = module('host_contract').HostEvidence(tmp_path)
    host.begin({'turn': 0})
    obs = lambda grid: dict(grid=grid, state='NOT_FINISHED', levels_completed=0)
    assert host.observe(obs(before), obs(after), 'UP', remaining=1) is False
    assert host.results[-1]['outcome'] == 'move'
    assert host.results[-1]['diff_pixels'] is None  # no aligned pixel count across shapes
    assert host.record.known_from_here(before, ['UP']) == {}
    changed = [row[:] for row in after]
    changed[-1][-1] = 3
    host.observe(obs(after), obs(changed), 'UP', remaining=0)
    assert host.results[-1]['diff_pixels'] == 1


@pytest.mark.parametrize('state,levels,outcome', [('GAME_OVER', 0, 'terminal'),
    ('WIN', 1, 'terminal'), ('NOT_FINISHED', 1, 'level_up')])
def test_shape_change_preserves_terminal_and_level_priority(tmp_path, state, levels, outcome):
    host = module('host_contract').HostEvidence(tmp_path)
    host.begin({'turn': 0})
    assert host.observe(dict(grid=[[1]], state='NOT_FINISHED', levels_completed=0),
                        dict(grid=[[1],[2]], state=state, levels_completed=levels), 'UP', 1)
    assert host.results[-1]['outcome'] == outcome


def test_configured_context_window_is_client_metadata(monkeypatch):
    monkeypatch.setenv('ARC_DISABLE_DOTENV', '1')
    monkeypatch.setenv('ARC_LLM_MODEL', 'qwen/qwen3.8-27b')
    monkeypatch.setenv('ARC_LLM_BASE_URL', 'http://127.0.0.1:9/v1')
    monkeypatch.setenv('ARC_LLM_API_KEY', 'local-fixture')
    monkeypatch.setenv('ARC_LLM_CONTEXT_WINDOW', '262144')
    monkeypatch.setenv('ARC_LLM_MAX_TOKENS', '16384')
    monkeypatch.setenv('ARC_LLM_RETRIES', '0')
    client = module('arc_llm').build_llm()
    try:
        assert client.context_window == 262144
        assert 'context_window' not in client.config  # must not leak into provider kwargs
    finally:
        client.close()


def test_actor_trajectory_reports_resize_as_unaligned(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE/'chassis'))
    from solver_agent import MdArcSolverAgent
    from nooa.unifiedllm import FakeLLMClient
    async def check():
        agent = MdArcSolverAgent(llm=FakeLLMClient(), run_dir=tmp_path,
                                 game_id='opaque', alias='game', enable_summarizer=False)
        try:
            (agent.ipc_dir/'transitions.jsonl').write_text(json.dumps(dict(
                step=1, turn=0, action='UP', before={'grid': [[1]]},
                after={'grid': [[1], [2]], 'levels_completed': 0}))+'\n')
            assert agent.last_transition()['changed_cells'] is None
            assert agent.trajectory()[0]['diff_pixels'] is None
        finally:
            await agent.queue_manager.shutdown()
    asyncio.run(check())


def test_successful_submission_resets_refusal_streak(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE/'chassis'))
    from solver_agent import MdArcSolverAgent, _ReturnResultSignal
    from nooa.unifiedllm import FakeLLMClient
    async def check():
        agent = MdArcSolverAgent(llm=FakeLLMClient(), run_dir=tmp_path,
                                 game_id='opaque', alias='game', enable_summarizer=False)
        def state(turn):
            agent._states_path.write_text(json.dumps(dict(turn=turn, step=turn,
                grid_rows=['01'], levels_completed=0, state='NOT_FINISHED',
                available_actions=['UP', 'DOWN']))+'\n')
        def reject():
            assert agent.submit_actions(['UP', 'DOWN'], rationale='predict: unqualified batch').startswith('REJECTED:')
        try:
            # Use turn 1 to expose the stale streak, rather than turn-zero's
            # separate falsey-value behavior in the old implementation.
            state(1)
            for _ in range(7): reject()
            with pytest.raises(_ReturnResultSignal):
                agent.submit_actions(['UP'], rationale='predict: one exploratory action')
            state(2)
            reject()
            assert not (agent.ipc_dir/'ABORT').exists()
            for _ in range(7): reject()
            assert '8 consecutive' in (agent.ipc_dir/'ABORT').read_text()
        finally:
            await agent.queue_manager.shutdown()
    asyncio.run(check())
