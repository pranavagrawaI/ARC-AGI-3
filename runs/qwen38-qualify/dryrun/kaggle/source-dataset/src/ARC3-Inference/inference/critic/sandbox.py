"""Investigation-only Python for the critic, on top of Duck's own sandbox.

The critic gets exactly the globals Duck's actor gets (`current_frame`, `previous_frame`,
`history`, `transitions`, `last_transition`, `valid_actions`, `last_action_result`),
built from a frozen packet. `action(...)` is wired to a host handler that refuses,
so inside the sandbox it raises `RuntimeError: action failed in sandbox host.`.
"""
from __future__ import annotations

from typing import Any

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.tool_agent import _ascii_frame_view_payload, _ascii_history_view_payload
from inference.critic.packet import CriticPacket

DEFAULT_TIMEOUT_SECONDS = 30


class InvestigationOnlyError(RuntimeError):
    """Raised on the host side whenever critic code calls `action(...)`."""


def packet_to_initial_state(packet: CriticPacket) -> dict[str, Any]:
    """Shape the packet like `ToolAgent._run_python_tool`'s `_serialized_runtime_state`."""
    return {
        "current_frame": _ascii_frame_view_payload(packet.current_frame_obj()),
        "history": _ascii_history_view_payload(packet.history_entries()),
        "valid_actions": list(packet.valid_actions),
        "last_action_result": dict(packet.last_action_result),
    }


def run_investigation_python(
    packet: CriticPacket,
    code: str,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run `code` against the packet. Same result dict as `run_sandboxed_python`,
    plus `refused_action_calls`: how many times the code tried to act."""
    refused: list[list[dict[str, Any]]] = []

    def _refuse(actions: list[dict[str, Any]]) -> dict[str, Any]:
        refused.append(list(actions))
        raise InvestigationOnlyError("critic sandbox is investigation-only; action() is refused")

    result = run_sandboxed_python(
        code=code,
        timeout_seconds=max(1, int(timeout_seconds)),
        initial_state=packet_to_initial_state(packet),
        action_handler=_refuse,
    )
    result["refused_action_calls"] = len(refused)
    return result


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "InvestigationOnlyError",
    "packet_to_initial_state",
    "run_investigation_python",
]
