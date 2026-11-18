import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from one_tool_runtime import (
    Admission, Config, StopRun, decode_reply, observation, play_game, validate_actions,
)


def frame(state="NOT_FINISHED", level=0, actions=(1, 2, 6), frames=None):
    return SimpleNamespace(state=state, levels_completed=level,
                           available_actions=list(actions), win_levels=2, full_reset=False,
                           frame=[[[0, 10], [15, 1]]] if frames is None else frames)


def reply(actions, finish="tool_calls", reasoning="test reasoning"):
    return {"choices": [{"finish_reason": finish, "message": {
        "role": "assistant", "content": None, "reasoning": reasoning,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "do_actions", "arguments": json.dumps({"actions": actions})}}],
    }}]}


class Game:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def observe(self):
        return frame()

    def step(self, action):
        self.calls.append(action)
        result = next(self.outputs)
        if isinstance(result, Exception):
            raise result
        return result


class Backend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def complete(self, messages, key, journal):
        self.requests.append(json.loads(json.dumps(messages)))
        return next(self.outputs)


def cfg(tmp_path, **kw):
    return replace(Config(log_dir=str(tmp_path), deadline=time.monotonic()+10), **kw)


def test_sequence_uses_one_inference(tmp_path):
    game = Game([frame(), frame(), frame("WIN", 2)])
    backend = Backend([reply([{"name": "ACTION1", "repeat": 3}])])
    result = play_game(game, "g1", backend, cfg(tmp_path))
    assert result["status"] == "WIN"
    assert result["actions"] == 3
    assert len(backend.requests) == 1


def test_level_change_discards_remaining_actions(tmp_path):
    game = Game([frame(level=1), frame("WIN", 2)])
    backend = Backend([reply([{"name": "ACTION1", "repeat": 8}]),
                       reply([{"name": "ACTION2"}])])
    assert play_game(game, "g1", backend, cfg(tmp_path))["status"] == "WIN"
    assert [a.name for a in game.calls] == ["ACTION1", "ACTION2"]
    tool = json.loads(backend.requests[1][-1]["content"])
    assert (tool["executed"], tool["requested"], tool["stopped"]) == (1, 8, "level_change")
    assert backend.requests[1][-2]["reasoning"] == "test reasoning"


def test_game_over_does_not_automatically_reset(tmp_path):
    game = Game([frame("GAME_OVER"), frame(), frame("WIN", 2)])
    backend = Backend([reply([{"name": "ACTION1", "repeat": 8}]),
                       reply([{"name": "RESET"}, {"name": "ACTION2"}]),
                       reply([{"name": "ACTION2"}])])
    assert play_game(game, "g1", backend, cfg(tmp_path))["status"] == "WIN"
    assert [a.name for a in game.calls] == ["ACTION1", "RESET", "ACTION2"]
    assert len(backend.requests) == 3


@pytest.mark.parametrize("bad", [
    '{"actions":[{"name":"ACTION1","repeat":33}]}',
    '{"actions":[{"name":"ACTION6","x":64,"y":0}]}',
    '{"actions":[{"name":"ACTION6","x":true,"y":0}]}',
    '{"actions":[{"name":"ACTION1","repeat":true}]}',
    '{"actions":[{"name":"ACTION1","x":0,"y":0}]}',
    '{"actions":[{"name":"ACTION9"}]}',
    '{"actions":[]}',
    'not json',
])
def test_invalid_schema_is_rejected(bad):
    with pytest.raises(StopRun):
        validate_actions(bad, 32)


def test_validate_whole_sequence_before_execution(tmp_path):
    game = Game([])
    backend = Backend([reply([{"name": "ACTION1"}, {"name": "ACTION6"}])])
    assert play_game(game, "g1", backend, cfg(tmp_path))["status"] == "ERROR"
    assert game.calls == []


def test_length_finish_cannot_execute_even_valid_json(tmp_path):
    game = Game([])
    backend = Backend([reply([{"name": "ACTION1"}], finish="length")])
    assert play_game(game, "g1", backend, cfg(tmp_path))["status"] == "ERROR"
    assert not game.calls


@pytest.mark.parametrize("failure", [None, TimeoutError("unknown action result")])
def test_unknown_outcome_never_retries(tmp_path, failure):
    game = Game([failure])
    backend = Backend([reply([{"name": "ACTION1", "repeat": 8}])])
    assert play_game(game, "g1", backend, cfg(tmp_path))["status"] == "ERROR"
    assert len(game.calls) == len(backend.requests) == 1


def test_all_frames_preserved():
    result = observation(frame(frames=[[[0, 15]], [[10, 11]]]))
    assert result["frames"] == [["0F"], ["AB"]]


def test_expired_deadline_never_infers_or_acts(tmp_path):
    backend, game = Backend([]), Game([])
    result = play_game(game, "g1", backend, cfg(tmp_path, deadline=time.monotonic()-1))
    assert result["status"] == "DEADLINE"
    assert not backend.requests and not game.calls


def test_admission_overlaps_requests_and_bounds_inflight(tmp_path):
    config = cfg(tmp_path, concurrency=2)
    admission = Admission(config)
    lock = threading.Lock()
    active = maximum = 0
    events = []

    def job(i):
        nonlocal active, maximum
        with admission:
            with lock:
                active += 1
                maximum = max(maximum, active)
                events.append(("start", i))
            time.sleep(0.04 if i == 0 else 0.01)
            with lock:
                events.append(("end", i))
                active -= 1

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(job, range(6)))
    assert maximum == 2
    assert events.index(("start", 2)) < events.index(("end", 0))
    assert admission.active == 0 and not admission.waiting


def test_isolation_and_one_request_per_game(tmp_path):
    config = cfg(tmp_path)
    games = [Game([frame("WIN", 2)]) for _ in range(4)]
    backends = [Backend([reply([{"name": "ACTION6", "x": i, "y": i}])]) for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(play_game, game, f"game-{i}", backend, config)
                 for i, (game, backend) in enumerate(zip(games, backends))]
        assert all(t.result()["status"] == "WIN" for t in tasks)
    assert [g.calls[0].data() for g in games] == [{"x": i, "y": i} for i in range(4)]
    assert all(len(b.requests) == 1 for b in backends)
    assert len(list(tmp_path.glob("*/events.jsonl"))) == 4


def test_unavailable_action_does_not_reach_environment(tmp_path):
    game = Game([frame("WIN", 2)])
    backend = Backend([reply([{"name": "ACTION5"}]), reply([{"name": "ACTION1"}])])
    result = play_game(game, "g1", backend, cfg(tmp_path))
    assert result["status"] == "WIN" and len(game.calls) == 1
    assert json.loads(backend.requests[1][-1]["content"])["stopped"] == "action_unavailable"


def test_duplicate_game_owner_fails_before_second_request(tmp_path):
    config = cfg(tmp_path)
    backend = Backend([reply([{"name": "ACTION1"}])])
    play_game(Game([frame("WIN", 2)]), "g1", backend, config)
    with pytest.raises(FileExistsError):
        play_game(Game([]), "g1", backend, config)
    assert len(backend.requests) == 1
