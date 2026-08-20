"""Which of Tycho's four orchestration topologies this slice can actually run.

The registry is upstream's (`tycho.agent.modes`), imported rather than restated
so a mode name means here exactly what it means there. What this module adds is
the honest bound: tycho_nooa wires two of the four, and asking for one of the
other two must fail at startup rather than quietly run as `trigger`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from tycho.agent.modes import MODES, ModeSpec, resolve_mode

MODE_ENV = "TYCHO_MODE"
DEFAULT_MODE = "trigger"

# The two topologies whose switches have no implementation on this side, and the
# exact missing piece for each. Both are `ModeSpec` fields tycho_nooa never reads.
_UNWIRED: dict[str, str] = {
    "single": "actor_writes_wm: the actor is never told it owns world_model.py",
    "orchestrator": "include_builder: the actor has no invoke_builder tool",
}


def mode_from_env(environment: Mapping[str, str] | None = None) -> ModeSpec:
    """The mode this process serves, defaulting to the one every run so far used."""
    env = os.environ if environment is None else environment
    return served_mode(env.get(MODE_ENV) or DEFAULT_MODE)


def served_mode(name: str) -> ModeSpec:
    """Resolve a mode name upstream, then refuse the ones only upstream can run."""
    spec = resolve_mode(name)
    missing = _UNWIRED.get(spec.name)
    if missing is not None:
        raise NotImplementedError(
            f"{MODE_ENV}={spec.name!r} is a real Tycho mode that tycho_nooa does not "
            f"implement — {missing}. Served modes: {sorted(set(MODES) - set(_UNWIRED))}."
        )
    return spec


def world_model_enabled(spec: ModeSpec) -> bool:
    """Upstream's own derivation (`agent/agent.py`: `wm_variant != "none"`).

    It gates two things together on purpose: whether the workspace seeds
    world_model.py/verify.py/plan.py, and whether edits to those files trigger
    the automatic feedback probe. A run that seeds no model must not probe one.
    """
    return spec.wm_variant != "none"
