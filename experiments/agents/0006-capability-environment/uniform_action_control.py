"""A corrected control generation with uniformly named input events.

The frozen 0002-R control remains unchanged. This module derives a new control
specification by adding only public input-event names to its system prompt. It
adds no game effects, demonstrations, exploration policy, or treatment
capability.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from p9.arms import ArmSpec, control_spec


CORRECTED_CONTROL_ARM_ID = "0006-uniform-actions-v1"

ACTION_INPUT_CONVENTION = """\
The legal action IDs name input events only:
- 1: UP
- 2: DOWN
- 3: LEFT
- 4: RIGHT
- 5: SPACE
- 6: MOUSE, using integer grid coordinates x (column) and y (row)

These names describe only the input event sent to the game. They do not predict what that
input does in the current game or state. Infer every consequence, object role, rule, and
objective from observed action/observation transitions."""


def corrected_control_spec(
    base: ArmSpec | None = None,
    **overrides: Any,
) -> ArmSpec:
    """Derive the corrected control without mutating or re-pinning 0002-R."""

    if base is not None and overrides:
        raise ValueError("pass either a base control spec or overrides, not both")
    source = base if base is not None else control_spec(**overrides)
    if source.treatment_capabilities:
        raise ValueError("the corrected control cannot inherit treatment capabilities")
    return replace(
        source,
        arm_id=CORRECTED_CONTROL_ARM_ID,
        system_prompt=source.system_prompt.rstrip()
        + "\n\n"
        + ACTION_INPUT_CONVENTION,
    )