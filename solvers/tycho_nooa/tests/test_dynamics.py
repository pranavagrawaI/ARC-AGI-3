"""Exact controls for the bounded dynamics machinery seeded into workspaces."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pytest

from tycho_nooa import dynamics_template as dynamics

GLOBAL_FLATTEN = False
HELPER_FLATTEN = False
SETTINGS = ModuleType("dynamics_test_settings")
SETTINGS.FLATTEN = False


def global_projection(state):
    return 0 if GLOBAL_FLATTEN else state.x


def projection_helper(state):
    return 0 if HELPER_FLATTEN else state.x


def indirect_global_projection(state):
    return projection_helper(state)


def module_global_projection(state):
    return 0 if SETTINGS.FLATTEN else state.x


def law(certificate, name, *actions):
    return next(
        item
        for item in certificate
        if item["law"] == name and item["actions"] == list(actions)
    )


@dataclass(frozen=True)
class MoverState:
    x: int
    y: int
    counter: int = 0


class Mover:
    DELTAS: ClassVar[dict[str, tuple[int, int]]] = {
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
        "UP": (0, -1),
    }

    def __init__(self, *, count: bool = False) -> None:
        self.count = count

    def transition(self, state: MoverState, action: str) -> MoverState:
        dx, dy = self.DELTAS[action]
        return replace(
            state,
            x=state.x + dx,
            y=state.y + dy,
            counter=state.counter + int(self.count),
        )

    def actions(self, state: MoverState) -> list[str]:
        return list(self.DELTAS)

    @staticmethod
    def planner_key(state: MoverState):
        return state

    @staticmethod
    def render(state: MoverState):
        return state.x, state.y, state.counter


def test_pure_mover_finds_inverses_and_commuting_axes() -> None:
    certificates = dynamics.action_laws(
        Mover(),
        [MoverState(0, 0), MoverState(2, -3)],
        actions=["UP", "DOWN", "LEFT", "RIGHT"],
    )

    assert law(certificates, "inverse", "UP", "DOWN")["status"] == "supported"
    assert law(certificates, "inverse", "LEFT", "RIGHT")["status"] == "supported"
    assert law(certificates, "commutes", "RIGHT", "UP")["status"] == "supported"
    assert law(certificates, "idempotent", "UP")["status"] == "contradicted"


def test_mover_with_counter_is_reversible_only_after_projection() -> None:
    model = Mover(count=True)
    states = [MoverState(0, 0, 0), MoverState(2, -3, 4)]

    full = dynamics.action_laws(model, states, actions=["UP", "DOWN"])
    position = dynamics.action_laws(
        model,
        states,
        actions=["UP", "DOWN"],
        projection=lambda state: (state.x, state.y),
        representation="position",
    )

    failed = law(full, "inverse", "UP", "DOWN")
    assert failed["status"] == "contradicted"
    assert failed["counterexample"]["state"] == {"counter": 0, "x": 0, "y": 0}
    assert law(position, "inverse", "UP", "DOWN")["status"] == "supported"
    assert failed["representation_hash"] != position[0]["representation_hash"]
    assert position[0]["representation"] == "position"
    assert len(position[0]["domain"]["state_hashes"]) == len(states)
    assert len(position[0]["domain"]["projected_state_hashes"]) == len(states)
    assert position[0]["domain"]["state_values"] == [
        {"counter": 0, "x": 0, "y": 0},
        {"counter": 4, "x": 2, "y": -3},
    ]
    assert position[0]["domain"]["projected_state_values"] == [[0, 0], [2, -3]]
    assert len(position[0]["projection_commitment"]["code_sha256"]) == 64
    assert len(position[0]["theory_commitment"]["source_sha256"]) == 64
    supported = law(position, "inverse", "UP", "DOWN")
    assert len(supported["evidence"]) == len(states)
    assert all(item["left"] == item["right"] for item in supported["evidence"])


def test_representation_identity_binds_projection_code_not_only_initial_values() -> (
    None
):
    def signed_x(state):
        return state.x

    def absolute_x(state):
        return abs(state.x)

    signed = dynamics.action_laws(
        Mover(),
        [MoverState(0, 0)],
        actions=["LEFT"],
        projection=signed_x,
        representation="x",
    )
    absolute = dynamics.action_laws(
        Mover(),
        [MoverState(0, 0)],
        actions=["LEFT"],
        projection=absolute_x,
        representation="x",
    )

    assert signed[0]["domain"]["projected_state_values"] == [0]
    assert absolute[0]["domain"]["projected_state_values"] == [0]
    assert signed[0]["representation_hash"] != absolute[0]["representation_hash"]


def test_representation_identity_binds_captured_projection_values() -> None:
    def projection(flatten):
        def project(state):
            return 0 if flatten else state.x

        return project

    signed = dynamics.action_laws(
        Mover(),
        [MoverState(0, 0)],
        actions=["LEFT"],
        projection=projection(False),
        representation="factory_x",
    )
    flattened = dynamics.action_laws(
        Mover(),
        [MoverState(0, 0)],
        actions=["LEFT"],
        projection=projection(True),
        representation="factory_x",
    )

    assert signed[0]["domain"]["projected_state_values"] == [0]
    assert flattened[0]["domain"]["projected_state_values"] == [0]
    assert signed[0]["representation_hash"] != flattened[0]["representation_hash"]
    assert law(signed, "idempotent", "LEFT")["status"] == "contradicted"
    assert law(flattened, "idempotent", "LEFT")["status"] == "supported"


def test_representation_identity_binds_referenced_mutable_globals() -> None:
    global GLOBAL_FLATTEN

    GLOBAL_FLATTEN = False
    signed = dynamics.action_laws(
        Mover(),
        [MoverState(0, 0)],
        actions=["LEFT"],
        projection=global_projection,
        representation="global_x",
    )
    try:
        GLOBAL_FLATTEN = True
        flattened = dynamics.action_laws(
            Mover(),
            [MoverState(0, 0)],
            actions=["LEFT"],
            projection=global_projection,
            representation="global_x",
        )
    finally:
        GLOBAL_FLATTEN = False

    assert signed[0]["representation_hash"] != flattened[0]["representation_hash"]
    assert law(signed, "idempotent", "LEFT")["status"] == "contradicted"
    assert law(flattened, "idempotent", "LEFT")["status"] == "supported"


def test_representation_identity_binds_transitive_callable_globals() -> None:
    global HELPER_FLATTEN

    HELPER_FLATTEN = False
    signed = dynamics.action_laws(
        Mover(),
        [MoverState(0, 0)],
        actions=["LEFT"],
        projection=indirect_global_projection,
        representation="indirect_global_x",
    )
    try:
        HELPER_FLATTEN = True
        flattened = dynamics.action_laws(
            Mover(),
            [MoverState(0, 0)],
            actions=["LEFT"],
            projection=indirect_global_projection,
            representation="indirect_global_x",
        )
    finally:
        HELPER_FLATTEN = False

    assert signed[0]["representation_hash"] != flattened[0]["representation_hash"]
    assert law(signed, "idempotent", "LEFT")["status"] == "contradicted"
    assert law(flattened, "idempotent", "LEFT")["status"] == "supported"


def test_representation_identity_binds_referenced_module_state() -> None:
    SETTINGS.FLATTEN = False
    signed = dynamics.action_laws(
        Mover(),
        [MoverState(0, 0)],
        actions=["LEFT"],
        projection=module_global_projection,
        representation="module_global_x",
    )
    try:
        SETTINGS.FLATTEN = True
        flattened = dynamics.action_laws(
            Mover(),
            [MoverState(0, 0)],
            actions=["LEFT"],
            projection=module_global_projection,
            representation="module_global_x",
        )
    finally:
        SETTINGS.FLATTEN = False

    assert signed[0]["representation_hash"] != flattened[0]["representation_hash"]
    assert law(signed, "idempotent", "LEFT")["status"] == "contradicted"
    assert law(flattened, "idempotent", "LEFT")["status"] == "supported"


def test_representation_identity_binds_nested_callable_object_state() -> None:
    class ArrayLike:
        def tolist(self):
            return [1, 2]

    class Settings:
        def __init__(self, flatten):
            self.flatten = flatten

    class Projector:
        def __init__(self, flatten):
            self.settings = {
                "config": Settings(flatten),
                "origin": MoverState(0, 0),
                "samples": ArrayLike(),
                "tags": {"bounded"},
                "token": object(),
            }

        def __call__(self, state):
            return 0 if self.settings["config"].flatten else state.x

    signed = dynamics.action_laws(
        Mover(),
        [MoverState(0, 0)],
        actions=["LEFT"],
        projection=Projector(False),
        representation="stateful_x",
    )
    flattened = dynamics.action_laws(
        Mover(),
        [MoverState(0, 0)],
        actions=["LEFT"],
        projection=Projector(True),
        representation="stateful_x",
    )

    assert signed[0]["representation_hash"] != flattened[0]["representation_hash"]
    assert law(signed, "idempotent", "LEFT")["status"] == "contradicted"
    assert law(flattened, "idempotent", "LEFT")["status"] == "supported"


def test_callable_identity_ignores_absolute_source_location(tmp_path: Path) -> None:
    source = "def project(state):\n    return state.x\n"

    def load(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        spec = importlib.util.spec_from_file_location("shared_projection", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.project

    left = dynamics._callable_commitment(load(tmp_path / "left.py"))
    right = dynamics._callable_commitment(load(tmp_path / "nested" / "right.py"))

    assert left["source_path"] != right["source_path"]
    assert left["source_sha256"] == right["source_sha256"]
    assert left["identity_sha256"] == right["identity_sha256"]


@dataclass(frozen=True)
class LatchState:
    value: int


class ResetLatch:
    @staticmethod
    def transition(state: LatchState, action: str) -> LatchState:
        assert action == "RESET"
        return LatchState(0)


def test_reset_is_idempotent_and_many_to_one_not_an_inverse() -> None:
    certificates = dynamics.action_laws(
        ResetLatch(), [LatchState(0), LatchState(1)], actions=["RESET"]
    )

    assert law(certificates, "idempotent", "RESET")["status"] == "supported"
    many_to_one = law(certificates, "many_to_one", "RESET")
    assert many_to_one["status"] == "supported"
    assert many_to_one["witness"] == {
        "left_state": {"value": 0},
        "right_state": {"value": 1},
        "result": {"value": 0},
    }
    assert law(certificates, "inverse", "RESET", "RESET")["status"] == "contradicted"


@dataclass(frozen=True)
class UndoState:
    visible: int
    previous: int


class HiddenUndo:
    @staticmethod
    def transition(state: UndoState, action: str) -> UndoState:
        if action == "WAIT":
            return state
        assert action == "UNDO"
        return UndoState(state.previous, state.visible)

    @staticmethod
    def planner_key(state: UndoState) -> int:
        return state.visible

    @staticmethod
    def render(state: UndoState) -> int:
        return state.visible

    @staticmethod
    def actions(state: UndoState) -> list[str]:
        return ["WAIT", "UNDO"]


def test_hidden_undo_returns_the_shortest_separating_suffix() -> None:
    certificate = dynamics.key_counterexample(
        HiddenUndo(),
        [UndoState(5, 1), UndoState(5, 9)],
        actions=["WAIT", "UNDO"],
        max_depth=3,
        inference_transition_ids=["L0:T7:UNDO"],
    )

    assert certificate["status"] == "contradicted"
    assert certificate["counterexample"] == {
        "left_observation": {"actions": ["UNDO", "WAIT"], "outcome": None, "render": 1},
        "right_observation": {
            "actions": ["UNDO", "WAIT"],
            "outcome": None,
            "render": 9,
        },
        "suffix": ["UNDO"],
    }
    assert certificate["inference_transition_ids"] == ["L0:T7:UNDO"]
    assert certificate["resource_cap"]["nodes_checked"] == 2
    assert certificate["representation"] == "model.render+outcome+actions"
    assert len(certificate["domain"]["state_hashes"]) == 2
    assert len(certificate["domain"]["projected_state_hashes"]) == 2
    assert certificate["domain"]["state_values"] == [
        {"previous": 1, "visible": 5},
        {"previous": 9, "visible": 5},
    ]


def test_shorter_observation_separator_beats_earlier_deep_transition_failure() -> None:
    @dataclass(frozen=True)
    class Branch:
        hidden: int
        path: tuple[str, ...] = ()

    class AdversarialOrder:
        @staticmethod
        def planner_key(state):
            return 0

        @staticmethod
        def render(state):
            return state.hidden if state.path == ("B",) else 0

        @staticmethod
        def transition(state, action):
            if state.path == ("A",) and action == "A" and state.hidden == 0:
                raise ValueError("left branch ends")
            return replace(state, path=(*state.path, action))

    certificate = dynamics.key_counterexample(
        AdversarialOrder(),
        [Branch(0), Branch(1)],
        actions=["A", "B"],
        max_depth=2,
    )

    assert certificate["status"] == "contradicted"
    assert certificate["counterexample"]["suffix"] == ["B"]


def test_no_bounded_separator_remains_inconclusive() -> None:
    certificate = dynamics.key_counterexample(
        HiddenUndo(),
        [UndoState(5, 1), UndoState(5, 1)],
        actions=["WAIT", "UNDO"],
        max_depth=2,
    )

    assert certificate["status"] == "inconclusive"
    assert certificate["counterexample"] is None
    assert certificate["resource_cap"]["depth_exhausted"] is True


def test_counterexample_search_reports_node_cap_and_partial_transition() -> None:
    capped = dynamics.key_counterexample(
        HiddenUndo(),
        [UndoState(5, 1), UndoState(5, 1)],
        actions=["WAIT"],
        max_depth=3,
        max_nodes=1,
    )
    assert capped["status"] == "inconclusive"
    assert capped["resource_cap"]["node_cap_exhausted"] is True

    class Partial:
        @staticmethod
        def planner_key(state):
            return 0

        @staticmethod
        def render(state):
            return 0

        @staticmethod
        def transition(state, action):
            if state == "blocked":
                raise ValueError("closed")
            return state

    failed = dynamics.key_counterexample(
        Partial(), ["blocked", "open"], actions=["TRY"], max_depth=1
    )
    assert failed["status"] == "contradicted"
    assert failed["counterexample"]["suffix"] == ["TRY"]
    assert failed["counterexample"]["left_observation"]["error"] == "ValueError"

    class RejectsBoth(Partial):
        @staticmethod
        def transition(state, action):
            raise ValueError("closed")

    same_failure = dynamics.key_counterexample(
        RejectsBoth(), ["left", "right"], actions=["TRY"], max_depth=1
    )
    assert same_failure["status"] == "inconclusive"
    assert same_failure["counterexample"] is None

    no_actions = dynamics.key_counterexample(
        HiddenUndo(),
        [UndoState(5, 1), UndoState(5, 9)],
        actions=[],
        max_depth=1,
    )
    assert no_actions["status"] == "inconclusive"
    assert no_actions["resource_cap"]["nodes_checked"] == 1

    with pytest.raises(ValueError, match="max_actions"):
        dynamics.key_counterexample(
            HiddenUndo(),
            [UndoState(5, 1), UndoState(5, 9)],
            actions=["UNDO", "WAIT"],
            max_actions=1,
        )

    with pytest.raises(ValueError, match="representation"):
        dynamics.key_counterexample(
            HiddenUndo(),
            [UndoState(5, 1), UndoState(5, 9)],
            actions=["UNDO"],
            observe=lambda state: state.visible,
        )


def test_law_caps_and_partial_domains_remain_explicit() -> None:
    with pytest.raises(ValueError, match="max_actions"):
        dynamics.action_laws(
            Mover(), [MoverState(0, 0)], actions=["UP", "DOWN"], max_actions=1
        )

    class Partial:
        @staticmethod
        def transition(state, action):
            if state == 1:
                raise ValueError("outside domain")
            return state

    certificates = dynamics.action_laws(Partial(), [0, 1], actions=["TRY"])
    idempotent = law(certificates, "idempotent", "TRY")
    assert idempotent["status"] == "inconclusive"
    assert idempotent["exclusions"][0]["error"] == "ValueError"
    assert law(certificates, "many_to_one", "TRY")["status"] == "inconclusive"


def test_certificates_canonicalize_model_owned_container_states() -> None:
    class ArrayLike:
        def __init__(self, value):
            self.value = value

        def tolist(self):
            return [self.value]

    class Projection:
        def __init__(self, state):
            self.payload = {state: ArrayLike(state)}
            self.tags = {"sample"}

    class Identity:
        @staticmethod
        def transition(state, action):
            return state

    certificates = dynamics.action_laws(
        Identity(),
        [0, 1],
        actions=["WAIT"],
        projection=Projection,
        representation="container_projection",
    )

    assert json.loads(json.dumps(certificates)) == certificates
    assert len(certificates[0]["representation_hash"]) == 64


def test_mirrored_render_quotient_can_be_refuted_by_one_action() -> None:
    certificate = dynamics.key_counterexample(
        Mover(),
        [MoverState(-1, 0), MoverState(1, 0)],
        actions=["RIGHT"],
        key=lambda state: (abs(state.x), state.y),
        observe=lambda state: (abs(state.x), state.y),
        representation="absolute_x_and_y",
        max_depth=1,
    )

    assert certificate["status"] == "contradicted"
    assert certificate["counterexample"]["suffix"] == ["RIGHT"]


def test_current_runtime_seeds_and_preloads_dynamics(tmp_path: Path) -> None:
    from nooa.unifiedllm import FakeLLMClient

    from tycho_nooa import TychoNooaAgent
    from tycho_nooa.trusted_runtime import current_runtime

    agent = TychoNooaAgent(
        llm=FakeLLMClient(scripted_responses=[]),
        trigger_builder=False,
        game_id="dynamics-runtime",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )

    module = next(
        module for module in current_runtime().modules if module.path == "dynamics.py"
    )
    assert (agent.workspace_dir / "dynamics.py").read_bytes() == module.content
    result = agent._tool_surface.run_python("print(dynamics.__name__)")
    assert "dynamics" in result
    assert "dynamics.key_counterexample" in agent._resolve_system_prompt()


def test_prior_declared_runtime_remains_exactly_resumable(tmp_path: Path) -> None:
    from tycho_nooa import runtime_notice
    from tycho_nooa.trusted_runtime import (
        TrustedRuntime,
        current_runtime,
        inspect_runtime,
    )

    prior_modules = tuple(
        module for module in current_runtime().modules if module.path != "dynamics.py"
    )
    prior = TrustedRuntime.create(prior_modules)
    prior.write_to(tmp_path)

    assert inspect_runtime(tmp_path).runtime == prior
    assert not (tmp_path / "dynamics.py").exists()
    assert "predates `dynamics.py`" in runtime_notice.prompt_brief(tmp_path)
    assert "preload is None" in runtime_notice.tool_note(tmp_path)["run_python"]
