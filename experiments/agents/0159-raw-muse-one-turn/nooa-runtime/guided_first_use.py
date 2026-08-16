"""Condition-specific procedural bridge from WM2-B to the theory lane."""

from __future__ import annotations

GUIDED_FIRST_USE_SCHEMA = "0159-guided-first-use-v1"
MAX_GUIDED_FIRST_USE_BYTES = 1_536

_GUIDANCE = """## Guided first use

You have evidence from repeated uses of the same action, no executable
commitment is active, and similar consequences may otherwise be reconstructed
again in scratch Python.

Before further reconstruction, call exactly:

    self.read_world_model()

This call is read-only and does not create a commitment. Inspect its returned
quick-theory template and current evidence frontier. Then choose one path:

1. preserve only a narrow executable rule you can defend;
2. keep competing or unresolved explanations in prose notes; or
3. explicitly defer executable commitment and continue investigating.

You decide the hypothesis, whether any edit is warranted, and the environment
action. The host will not create or promote a theory for you."""


def render_guided_first_use(
    *,
    condition_codes: tuple[str, ...],
    current_model_version_id: str | None,
    world_model_was_read: bool,
) -> str | None:
    """Render the one-time procedural bridge exactly while it is applicable."""

    applicable = (
        "repeated_action_evidence" in condition_codes
        and current_model_version_id is None
        and not world_model_was_read
    )
    if not applicable:
        return None
    if len(_GUIDANCE.encode("utf-8")) > MAX_GUIDED_FIRST_USE_BYTES:
        raise RuntimeError("guided first-use instruction exceeds its byte bound")
    return _GUIDANCE
