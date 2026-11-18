"""The transcript is bounded by model-free eviction, and nothing measured is lost.

v11 (summarizer off) grew ~5.5k prompt tokens per game turn with no ceiling
but the served context. These tests drive real ``handle()`` turns through a
scripted LLM and check the prompt stops growing, the digest is rebuilt from
host records, and the retained window is exactly the last N game turns.
"""
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _module(name):
    spec = importlib.util.spec_from_file_location('eviction_' + name, HERE / 'chassis' / f'{name}.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def _cell(code, ident):
    from nooa.unifiedllm.unifiedllm import LLMResponse, ToolCall
    call = ToolCall(id=f'c{ident}', name='execute_python', arguments=json.dumps({'code': code}))
    return LLMResponse(
        raw_response=None, content='', tool_calls=[call], finish_reason='tool_calls',
        assistant_message={'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': call.id, 'type': 'function',
             'function': {'name': 'execute_python', 'arguments': call.arguments}}]},
        reasoning=None, usage=None)


def _turn_responses(turns, ident_base=0):
    """Each game turn: one noisy inspection cell, then one submit cell."""
    out = []
    for t in range(turns):
        out.append(_cell(
            "st=json.loads(notification['game_states'][-1]); g=self.grid_array(st['grid_rows']);"
            " print('inspect turn', st['turn'], g.shape, 'x'*3000)", ident_base + 2 * t))
        out.append(_cell(
            "print(self.submit_actions(['UP'], rationale='explore: probe',"
            " predicted_path=[{'expect':'move','z':None}]))", ident_base + 2 * t + 1))
    return out


def _state(turn, grid):
    return json.dumps(dict(turn=turn, step=turn, level=0, state='NOT_FINISHED', levels_completed=0,
                           available_actions=['UP', 'DOWN'], grid_rows=grid))


def _transition(turn, before, after):
    grid = lambda rows: [[int(c, 16) for c in row] for row in rows]
    return json.dumps(dict(step=turn, turn=turn, action='UP',
                           before={'grid': grid(before), 'levels_completed': 0, 'state': 'NOT_FINISHED'},
                           after={'grid': grid(after), 'levels_completed': 0, 'state': 'NOT_FINISHED'}))


async def _play(agent, llm, turns, *, run_dir):
    """Run ``turns`` handle() wakes; the host appends a transition after each."""
    grids = [['0' * 64] * 64]
    prompt_chars = []
    for t in range(turns):
        line = _state(t, grids[-1])
        with agent._states_path.open('a') as f:
            f.write(line + '\n')
        result = await agent.handle({'game_states': [line]})
        assert str(getattr(result, 'kind', result.get('kind') if isinstance(result, dict) else result)).endswith('WAIT')
        prompt_chars.append(sum(len(json.dumps(m)) for m in llm.last_messages))
        nxt = [row[:] for row in grids[-1]]
        nxt[t % 64] = '1' + nxt[t % 64][1:]  # one cell changes per action
        with (run_dir / 'ipc' / 'transitions.jsonl').open('a') as f:
            f.write(_transition(t, grids[-1], nxt) + '\n')
        grids.append(nxt)
    return prompt_chars


async def _session(agent_class, fake_class, run_dir, turns, keep):
    # The agent spawns its state tail on construction, so build it on the loop.
    llm = fake_class(scripted_responses=_turn_responses(turns))
    agent = agent_class(llm=llm, run_dir=run_dir, game_id='g', alias='g',
                        enable_summarizer=False, transcript_keep_turns=keep)
    try:
        chars = await _play(agent, llm, turns, run_dir=run_dir)
    finally:
        await agent.queue_manager.shutdown()
    return agent, llm, chars


@pytest.fixture
def chassis(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE / 'chassis'))
    monkeypatch.setenv('ARC_DISABLE_DOTENV', '1')
    from solver_agent import MdArcSolverAgent
    from nooa.unifiedllm import FakeLLMClient
    return MdArcSolverAgent, FakeLLMClient


def test_digest_is_one_line_per_action_from_host_records_and_bounded():
    eviction = _module('transcript_eviction')
    rows = [json.loads(_transition(t, ['00'], ['10'])) for t in range(300)]
    text = eviction.digest(rows, max_chars=400)
    assert len(text) <= 400 + len(text.split(':\n')[0]) + 2
    assert text.rstrip().endswith('t299 s299 UP -> 1 cells')
    assert 'earliest omitted here; trajectory() has them' in text
    assert eviction.digest([]).endswith('none yet.')
    shape = json.loads(_transition(0, ['00'], ['000']))
    assert 'shape-change' in eviction.digest([shape])


def test_prompt_stops_growing_and_only_the_last_turns_survive(tmp_path, chassis):
    MdArcSolverAgent, FakeLLMClient = chassis
    from nooa.events import Summary, Task
    keep = 3
    turns = 9
    agent, llm, chars = asyncio.run(_session(MdArcSolverAgent, FakeLLMClient, tmp_path, turns, keep))

    # Growth before the window fills; a plateau after. The last three prompts
    # differ by less than one game turn's worth of history.
    assert chars[keep] > chars[0]
    plateau = chars[keep + 1:]
    assert max(plateau) - min(plateau) < (chars[1] - chars[0]) / 2, chars

    tags = agent.event_manager.keys()
    events = [agent.event_manager[t] for t in tags]
    assert isinstance(events[0], Summary), tags[:3]
    assert sum(isinstance(e, Task) for e in events) == keep
    assert not any(isinstance(e, Summary) for e in events[1:])

    # The digest is host-authored from transitions, one line per executed action.
    summary = events[0].summary_text
    for t in range(turns - keep):
        assert f't{t} s{t} UP -> 1 cells' in summary
    assert 'x' * 100 not in summary  # evicted cell noise is gone, not summarised

    record = [json.loads(l) for l in (tmp_path / 'ipc' / 'transcript_evictions.jsonl').read_text().splitlines()]
    assert record and all(r['event'] == 'EVICTED' for r in record)
    assert (tmp_path / 'ipc' / 'integrity.jsonl').exists() is False or all(
        json.loads(l).get('coverage') != 'UNOBSERVED' or json.loads(l).get('reason') == 'output evicted or absent'
        for l in (tmp_path / 'ipc' / 'integrity.jsonl').read_text().splitlines())


def test_zero_keeps_v11_growth(tmp_path, chassis):
    MdArcSolverAgent, FakeLLMClient = chassis
    from nooa.events import Summary
    turns = 6
    agent, llm, chars = asyncio.run(_session(MdArcSolverAgent, FakeLLMClient, tmp_path, turns, 0))
    assert all(b > a for a, b in zip(chars, chars[1:])), chars
    assert not any(isinstance(agent.event_manager[t], Summary) for t in agent.event_manager.keys())
    assert agent._transcript_eviction is None


def test_headless_defaults_eviction_on_with_summarizer_off():
    source = (HERE / 'headless.py').read_text()
    assert 'enable_summarizer=False' in source
    assert 'transcript_keep_turns=int(os.environ.get("ARC_TRANSCRIPT_KEEP_TURNS", "4"))' in source


def test_every_frozen_chassis_file_ships_in_the_notebook():
    """The notebook packs an explicit file list; a module in the freeze manifest
    but not in that list would pass qualification here and ImportError on Kaggle."""
    import build_notebook
    import freeze_candidate
    missing = set(freeze_candidate.FILES) - set(build_notebook.JOIN_FILES)
    assert not missing, sorted(missing)
