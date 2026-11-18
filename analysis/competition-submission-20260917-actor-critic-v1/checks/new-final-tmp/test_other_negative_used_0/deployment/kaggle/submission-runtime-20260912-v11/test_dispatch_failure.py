"""Exercise the real dispatcher with controlled notification and handle boundaries."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent


@pytest.fixture
def dispatch(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    spec = importlib.util.spec_from_file_location("failure_probe_headless", HERE / "headless.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._headless_dispatch


class Queue:
    def __init__(self, events):
        self.events = iter(events)
        self.calls = 0

    async def race(self):
        self.calls += 1
        event = next(self.events, ValueError("queue closed"))
        if isinstance(event, BaseException):
            raise event
        return event


def agent(tmp_path, queue, handle, state=None):
    state = state if state is not None else {"turn": 1, "step": 1}
    return SimpleNamespace(
        ipc_dir=tmp_path, queue_manager=queue, handle=handle, latest_state=lambda: state
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("side_effect", [False, True])
async def test_handle_failure_is_original_exception_without_reissue_or_queue_wait(
    dispatch, tmp_path, side_effect
):
    error = RuntimeError("length-limited response without a tool call")
    calls = []
    submitted = []

    async def handle(notification):
        calls.append(notification)
        if side_effect:
            submitted.append("action already submitted")
        raise error

    queue = Queue([[('game_states', 'wake')], [('user_messages', 'another wake')]])
    with pytest.raises(RuntimeError) as caught:
        await asyncio.wait_for(dispatch(agent(tmp_path, queue, handle)), timeout=0.5)
    assert caught.value is error
    assert queue.calls == 1
    assert len(calls) == 1
    assert len(submitted) == int(side_effect)


@pytest.mark.asyncio
async def test_success_suppresses_stale_state_but_allows_user_and_successor(
    dispatch, tmp_path
):
    state = {"turn": 1, "step": 1}
    calls = []

    async def handle(notification):
        calls.append(notification)
        if "user_messages" in notification:
            state.update(turn=2, step=2)
        return SimpleNamespace(kind=None)

    queue = Queue([
        [('game_states', 'initial')],
        [('game_states', 'stale')],
        [('user_messages', 'explicit')],
        [('game_states', 'successor')],
    ])
    await dispatch(agent(tmp_path, queue, handle, state))
    assert len(calls) == 3
    assert [json.loads(call['game_states'][0])['turn'] for call in calls] == [1, 1, 2]
    assert calls[1]['user_messages'] == ['explicit']
    logs = [json.loads(line) for line in (tmp_path / 'state_delivery.jsonl').read_text().splitlines()]
    assert [log['suppressed'] for log in logs] == [False, True, False, False]


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["queue", "handle"])
async def test_cancellation_stops_without_reissue(dispatch, tmp_path, boundary):
    entered = asyncio.Event()
    calls = []
    queue = Queue([[('game_states', 'initial')]])

    async def wait_until_cancelled():
        entered.set()
        await asyncio.Future()

    if boundary == "queue":
        queue.race = wait_until_cancelled

    async def handle(notification):
        calls.append(notification)
        await wait_until_cancelled()

    task = asyncio.create_task(dispatch(agent(tmp_path, queue, handle)))
    await asyncio.wait_for(entered.wait(), timeout=0.5)
    task.cancel()
    # Preserve the existing graceful dispatcher shutdown semantics.
    await asyncio.wait_for(task, timeout=0.5)
    assert len(calls) == int(boundary == "handle")
    assert queue.calls == int(boundary == "handle")
