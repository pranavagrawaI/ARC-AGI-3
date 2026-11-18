"""Deadline is clean admission control, not cancellation of a model call."""
from pathlib import Path
import json
import test_my_agent as fixtures
from tycho_nooa.models import ActionDecision

SOURCE = Path(__file__).resolve().parents[3] / 'deployment/kaggle/q6-prediction-feedback-20260906/adapter.py'


def test_wall_budget_reconciles_and_closes_without_another_decision(monkeypatch, tmp_path):
    monkeypatch.setattr(fixtures, 'AGENT_PATH', SOURCE)
    monkeypatch.setenv('TYCHO_GAMEPLAY_ADMISSION_SECONDS', '10')
    module = fixtures.load_agent_module(monkeypatch)
    now = [100.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    solver = fixtures.FakeSolver([ActionDecision(action='UP')])
    agent = fixtures.new_agent(module, fixtures.RecordingFactory(solver), tmp_path)
    first = fixtures.frame([[1, 0]])
    after = fixtures.frame([[0, 1]])
    now[0] = 900.0  # Startup before first playable action does not consume horizon.
    assert not agent.is_done([first], first)
    agent.choose_action([first], first)
    assert agent._gameplay_started == 900.0
    now[0] = 920.0  # Pending decision may overrun; successor must still be recorded.
    agent.frames = [first, after]
    agent.action_counter = 1
    assert agent.is_done(agent.frames, after)
    assert [kind for kind, _ in solver.events] == ['decide', 'observe']
    assert agent.is_done(agent.frames, after)
    agent.cleanup()
    assert solver.close_calls == 1 and len(solver.finished) == 1
    rows = [json.loads(line) for line in (tmp_path/'game-stops.jsonl').read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]['reason'] == 'wall_budget'
    assert rows[0]['executed_solver_actions'] == 1
    assert rows[0]['gameplay_elapsed_seconds'] == 20.0


def test_step_budget_is_distinct(monkeypatch, tmp_path):
    monkeypatch.setattr(fixtures, 'AGENT_PATH', SOURCE)
    monkeypatch.setenv('TYCHO_KAGGLE_MAX_ACTIONS', '1')
    module = fixtures.load_agent_module(monkeypatch)
    solver = fixtures.FakeSolver([ActionDecision(action='UP')])
    agent = fixtures.new_agent(module, fixtures.RecordingFactory(solver), tmp_path)
    first = fixtures.frame([[1, 0]])
    after = fixtures.frame([[0, 1]])
    agent.choose_action([first], first)
    assert agent.is_done([first, after], after)
    assert json.loads((tmp_path/'game-stops.jsonl').read_text())['reason'] == 'step_budget'


def test_real_worker_wall_stop_reconciles_then_cleans_up(monkeypatch, tmp_path):
    import sys
    import time
    import test_nooa_join as join
    import test_join_qualification as qualification
    candidate = SOURCE.parent
    monkeypatch.setattr(fixtures, 'AGENT_PATH', SOURCE)
    monkeypatch.setattr(join, 'CANDIDATE', candidate)
    monkeypatch.setenv('NOOA_AGENT_COMMAND', json.dumps([sys.executable, str(candidate/'stub_agent.py')]))
    monkeypatch.setenv('NOOA_STUB_BATCH', 'UP,DOWN,LEFT')
    monkeypatch.setenv('NOOA_AGENT_READY_TIMEOUT_SECONDS', '10')
    monkeypatch.setenv('TYCHO_NOOA_WORKER_TIMEOUT_SECONDS', '15')
    monkeypatch.setenv('TYCHO_GAMEPLAY_ADMISSION_SECONDS', '7200')
    monkeypatch.syspath_prepend(str(candidate))
    first = qualification.frame([[0, 1], [0, 0]], actions=qualification.MOVES)
    after = qualification.frame([[0, 1], [1, 0]], actions=qualification.MOVES)
    agent = qualification.build_agent(monkeypatch, tmp_path, first)
    try:
        qualification.drive(agent, [first])
        session = agent._solver.session_root
        # Advance only the agent's origin; preserve monotonic time for subprocess
        # deadlines and queues so this test exercises actual worker lifecycle.
        agent._gameplay_started = time.monotonic() - 7201
        agent.frames.append(after)
        assert agent.is_done(agent.frames, after)
    finally:
        agent.cleanup()
    assert json.loads((session/'stop.json').read_text())['reason'] == 'competition_agent_cleanup'
    transitions = [json.loads(line) for line in (session/'ipc/transitions.jsonl').read_text().splitlines()]
    assert len(transitions) == 1
    assert transitions[0]['after']['grid'] == [[0, 1], [1, 0]]
    stop = json.loads((tmp_path/'workspaces/game-stops.jsonl').read_text())
    assert stop['reason'] == 'wall_budget' and stop['executed_solver_actions'] == 1
    assert not (session/'ipc/ABORT').exists()
