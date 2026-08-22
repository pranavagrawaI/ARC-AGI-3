"""One mechanically enforced policy for each of Tycho's four source modes.

The mode registry remains upstream's (`tycho.agent.modes`). This Adapter adds
the local implementation status and the actor's write authorization so prompt,
workspace seeding, verifier activation, and builder ownership cannot drift into
independent switches carrying the same mode name.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePath

from tycho.agent.modes import MODES, ModeSpec, resolve_mode

MODE_ENV = "TYCHO_MODE"
DEFAULT_MODE = "trigger"

# This is intentionally not derived from ``MODES``. A newly pinned upstream
# mode must earn a local policy audit instead of becoming executable merely by
# appearing in Tycho's registry.
_AUDITED_MODES = frozenset(
    {"trigger", "no_world_model", "single", "orchestrator"}
)


@dataclass(frozen=True)
class ModePolicy:
    """The complete local behavioral contract of one upstream mode."""

    spec: ModeSpec
    unavailable_reason: str | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def wm_variant(self) -> str:
        return self.spec.wm_variant

    @property
    def include_builder(self) -> bool:
        return self.spec.include_builder

    @property
    def actor_writes_wm(self) -> bool:
        return self.spec.actor_writes_wm

    @property
    def harness_fires_builder(self) -> bool:
        return self.spec.harness_fires_builder

    @property
    def needs_builder(self) -> bool:
        return self.spec.needs_builder

    @property
    def world_model_enabled(self) -> bool:
        return self.wm_variant != "none"

    def authorize_actor_write(self, path: str) -> None:
        """Reject actor edits to builder-owned or absent executable models."""
        if self.actor_writes_wm or not _is_world_model_path(path):
            return
        if self.wm_variant == "none":
            reason = "this mode has no executable world model"
        else:
            reason = "world_model.py is builder-owned in this mode"
        raise PermissionError(f"actor cannot write {path!r}: {reason}")


MODE_POLICIES: dict[str, ModePolicy] = {
    name: ModePolicy(
        spec,
        None if name in _AUDITED_MODES else "its local ownership contract is unaudited",
    )
    for name, spec in MODES.items()
}


def _is_world_model_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return PurePath(normalized).name.lower() in {"world_model.py", "worldmodel.py"}


def mode_from_env(environment: Mapping[str, str] | None = None) -> ModePolicy:
    """The mode this process serves, defaulting to the one every run so far used."""
    env = os.environ if environment is None else environment
    return served_mode(env.get(MODE_ENV) or DEFAULT_MODE)


def served_mode(name: str) -> ModePolicy:
    """Resolve a mode name upstream, then refuse the ones only upstream can run."""
    spec = resolve_mode(name)
    if spec.name not in _AUDITED_MODES:
        raise NotImplementedError(
            f"{MODE_ENV}={spec.name!r} exists upstream but its local ownership "
            f"contract is unaudited. Audited modes: {sorted(_AUDITED_MODES)}."
        )
    policy = MODE_POLICIES[spec.name]
    missing = policy.unavailable_reason
    if missing is not None:
        raise NotImplementedError(
            f"{MODE_ENV}={spec.name!r} is a real Tycho mode that tycho_nooa does not "
            f"implement — {missing}. Served modes: {sorted(_AUDITED_MODES)}."
        )
    return policy


def world_model_enabled(spec: ModePolicy | ModeSpec) -> bool:
    """Upstream's own derivation (`agent/agent.py`: `wm_variant != "none"`).

    It gates two things together on purpose: whether the workspace seeds
    world_model.py/verify.py/plan.py, and whether edits to those files trigger
    the automatic feedback probe. A run that seeds no model must not probe one.
    """
    if isinstance(spec, ModePolicy):
        return spec.world_model_enabled
    return spec.wm_variant != "none"
