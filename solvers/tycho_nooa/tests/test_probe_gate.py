"""The two planner preconditions upstream never checks: a goal that can exist,
and an operator alphabet worth searching."""

from __future__ import annotations

import importlib.util
import json
from collections import namedtuple
from pathlib import Path

import pytest
import tycho as seed_world_model
from nooa.unifiedllm import FakeLLMClient
from tycho.workspace import agent_tools
from tycho.workspace import wmlib_template as wmlib

from tycho_nooa import TychoNooaAgent
from tycho_nooa.probe_gate import (
    _field_view,
    wmgate_audit_lines,
    wmgate_blocked_lines,
    wmgate_report,
)
from tycho_nooa.wm_probe import (
    GatedToolExecutor,
    _require_upstream_wm_feedback,
    gated_probe_source,
)

SEED_OUTCOME = """
def outcome(state):
    return "ongoing"
"""

GOAL_OUTCOME = """
def outcome(state):
    if state.row == 6:
        return "level_complete"
    return "ongoing"
"""

DYNAMIC_GOAL_OUTCOME = """
def outcome(state):
    return "level_" + "complete"
"""

# A sliding model shaped like the one a live ar25 builder wrote: a move counter
# that only ever rises, an UNDO that restores positions but not the counter, a
# CLICK that does nothing, and a RESET back to the level's first frame.
COUNTED_MODEL = """
from dataclasses import dataclass, field


@dataclass
class State:
    row: int
    col: int
    moves: int = 0
    history: list = field(default_factory=list)


def init_state(grid, level):
    return State(row=0, col=0)


def transition(state, action):
    name = action["action"] if isinstance(action, dict) else action
    if name == "CLICK":
        return state
    if name == "RESET":
        return State(row=0, col=0)
    if name == "UNDO":
        if state.history:
            state.row, state.col = state.history.pop()
        return state
    state.history.append((state.row, state.col))
    state.row += -1 if name == "UP" else 1
    state.moves += 1
    return state


def render(state):
    return [[state.row, state.col]]


def actions(state):
    return [
        {"action": "UP", "row": None, "col": None},
        {"action": "DOWN", "row": None, "col": None},
        {"action": "CLICK", "row": None, "col": None},
        {"action": "UNDO", "row": None, "col": None},
        {"action": "RESET", "row": None, "col": None},
    ]
"""

# The same dynamics with no counter and no history: UP and DOWN are exact
# inverses, so the key closes and the roles fall out of position alone.
INVERTIBLE_MODEL = """
from dataclasses import dataclass


@dataclass
class State:
    row: int
    col: int


def init_state(grid, level):
    return State(row=0, col=0)


def transition(state, action):
    name = action["action"] if isinstance(action, dict) else action
    if name == "UP":
        return State(row=state.row - 1, col=state.col)
    if name == "DOWN":
        return State(row=state.row + 1, col=state.col)
    return state


def render(state):
    return [[state.row, state.col]]


def actions(state):
    return [
        {"action": "DOWN", "row": None, "col": None},
        {"action": "UP", "row": None, "col": None},
        {"action": "CLICK", "row": None, "col": None},
    ]
"""

# Two independent axes and nothing else: every declared operator is forward.
MONOTONE_MODEL = """
from dataclasses import dataclass


@dataclass
class State:
    row: int
    col: int


def init_state(grid, level):
    return State(row=0, col=0)


def transition(state, action):
    name = action["action"] if isinstance(action, dict) else action
    if name == "DOWN":
        return State(row=state.row + 1, col=state.col)
    return State(row=state.row, col=state.col + 1)


def render(state):
    return [[state.row, state.col]]


def actions(state):
    return [
        {"action": "DOWN", "row": None, "col": None},
        {"action": "RIGHT", "row": None, "col": None},
    ]
"""


def test_seed_outcome_blocks_the_search_with_its_reason(gate) -> None:
    report = gate(SEED_OUTCOME)

    assert report["goal_satisfiable"] is False
    text = "\n".join(wmgate_blocked_lines(report))
    assert "PLANNER_STATUS: not_run_goal_unsatisfiable" in text
    assert 'outcome() returns the constant "ongoing" unconditionally' in text
    assert '"level_complete" appears in no executable string' in text
    # The line the gate exists to replace, named so the builder cannot read this
    # as a slow search: six wasted searches reported planner_timeout instead.
    assert "planner_timeout" in text
    assert "write a candidate terminal predicate" in text


def test_a_reachable_goal_leaves_the_search_alone(gate) -> None:
    report = gate(GOAL_OUTCOME)

    assert report["goal_satisfiable"] is True
    assert "PLANNER_STATUS" not in "\n".join(wmgate_audit_lines(report))


def test_a_dynamically_constructed_goal_cannot_be_statically_blocked(gate) -> None:
    report = gate(DYNAMIC_GOAL_OUTCOME)

    assert report["goal_status"] == "candidate_present"
    assert report["goal_satisfiable"] is True
    assert report["goal"]["literal_in_module"] is False
    assert report["goal"]["observed_values"] == ["level_complete"]


def test_the_seed_world_model_is_blocked(gate) -> None:
    """The regression a live run found and every replay had masked.

    Pinned Tycho seeds `subgoals()` as `[lambda s: outcome(s) == "level_complete"]`
    while `outcome()` is still `return "ongoing"`. A module-wide literal search
    sees the goal string and clears the one model with provably no goal state.
    """
    seed = (
        Path(seed_world_model.__file__).parent
        / "workspace"
        / "templates"
        / "seed_world_model.py.tmpl"
    ).read_text()
    assert 'outcome(s) == "level_complete"' in seed
    assert 'def outcome(state: State) -> str:\n    """' in seed

    report = gate("", model=seed)

    assert report["goal_satisfiable"] is False
    assert report["goal"]["literal_in_module"] is True
    assert report["goal"]["constant_return"] == "ongoing"
    text = "\n".join(wmgate_blocked_lines(report))
    assert 'outcome() returns the constant "ongoing" unconditionally' in text
    # The weaker evidence line must stay off: the literal IS in this module.
    assert "appears in no executable string" not in text


def test_a_goal_named_only_in_a_docstring_is_not_a_goal(gate) -> None:
    outcome = '''
def outcome(state):
    """Returns level_complete when the piece lands."""
    return "ongoing"
'''
    assert gate(outcome)["goal_satisfiable"] is False


def test_operators_are_classified_by_what_they_do_to_the_planner_key(gate) -> None:
    roles = _roles(gate(GOAL_OUTCOME))

    assert roles["CLICK"] == "no_op"
    assert roles["RESET"] == "restart"
    assert roles["UP"] == "forward"
    assert roles["DOWN"] == "forward"
    # UNDO restores the position and pops the history but never lowers `moves`,
    # so under the planner's own key it is not a return at all: it is a second
    # forward operator whose successors can never be merged with anything.
    assert roles["UNDO"] == "retrograde_partial"


def test_a_partial_return_names_the_field_that_blocks_closure(gate) -> None:
    note = next(
        entry["note"]
        for entry in gate(GOAL_OUTCOME)["operators"]
        if entry["label"] == "UNDO"
    )

    assert "moves does not follow" in note
    assert "the key never closes" in note


def test_an_exact_inverse_reads_as_retrograde(gate) -> None:
    """Roles are behavioural: nothing here is named UNDO, and UNDO above was not
    retrograde. The sample walks DOWN, so UP returns it to a key it has held."""
    roles = _roles(gate(GOAL_OUTCOME, model=INVERTIBLE_MODEL))

    assert roles["UP"] == "retrograde"
    assert roles["DOWN"] == "forward"
    assert roles["CLICK"] == "no_op"


def test_the_forward_alphabet_gets_its_own_closure_statistic(gate) -> None:
    report = gate(GOAL_OUTCOME)

    assert report["closure_forward"]["operators"] == 2
    text = "\n".join(wmgate_audit_lines(report))
    assert "all 5 declared operators" in text
    # The number that matters: what search costs after the recommended cleanup.
    assert "the 2 forward operators only" in text


def test_a_uniform_alphabet_gets_no_second_closure_pass(gate) -> None:
    """Every operator forward means the contrast would repeat the first line."""
    report = gate(GOAL_OUTCOME, model=MONOTONE_MODEL)

    assert _roles(report) == {"DOWN": "forward", "RIGHT": "forward"}
    assert report["closure_forward"] is None


def test_a_gate_failure_never_blocks_the_search(gate) -> None:
    class Exploding:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    report = wmgate_report(_load_model(GOAL_OUTCOME, Path.cwd()), Exploding())

    assert report["goal_satisfiable"] is True
    assert report["error"] == "RuntimeError: boom"
    assert wmgate_audit_lines(report) == [
        "PLANNER_CONTRACT:",
        "  PROBE_ERROR: RuntimeError: boom",
    ]


def test_the_spliced_probe_is_upstreams_probe_plus_the_gate() -> None:
    source = gated_probe_source()

    compile(source, "<probe>", "exec")
    assert len(source) > len(agent_tools._WM_FEEDBACK_PROBE)
    assert "not_run_goal_unsatisfiable" in source
    # Upstream's own search still runs, exactly once, behind the gate.
    assert source.count("pr = _planner_probe()") == 1
    assert "if planner_ready:\n        pr = _planner_probe()" not in source


def test_the_splice_refuses_a_drifted_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_tools,
        "_WM_FEEDBACK_PROBE",
        agent_tools._WM_FEEDBACK_PROBE.replace("pr = _planner_probe()", "pass"),
    )

    with pytest.raises(RuntimeError, match="expected exactly once"):
        gated_probe_source()


def test_the_gated_probe_is_pinned_to_the_upstream_it_was_read_against() -> None:
    _require_upstream_wm_feedback()


def test_the_agent_wires_the_gate_into_every_automatic_probe(tmp_path: Path) -> None:
    agent = TychoNooaAgent(
        llm=FakeLLMClient(scripted_responses=[]),
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )

    assert isinstance(agent._tool_executor, GatedToolExecutor)


def test_state_field_projection_supports_python_record_shapes() -> None:
    class WMLib:
        @staticmethod
        def _hashable(value):
            return tuple(value) if isinstance(value, list) else value

    Point = namedtuple("Point", "row col")

    class PlainState:
        def __init__(self) -> None:
            self.row = 2
            self.history = [1, 2]

    assert _field_view(WMLib, Point(1, 3), None) == {"row": 1, "col": 3}
    assert _field_view(WMLib, PlainState(), ["history"]) == {"history": (1, 2)}
    assert _field_view(WMLib, object(), None) is None


def test_state_field_projection_degrades_when_hashing_is_unsupported() -> None:
    class WMLib:
        @staticmethod
        def _hashable(value):
            raise TypeError("unsupported")

    Point = namedtuple("Point", "row")
    assert _field_view(WMLib, Point(1), None) is None


def test_gated_feedback_preserves_skip_guards_and_runs_the_spliced_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = TychoNooaAgent(
        llm=FakeLLMClient(scripted_responses=[]),
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP"],
    )
    executor = agent._tool_executor
    executor.wm_feedback_enabled = False
    assert executor._wm_feedback("world_model.py") == ""

    executor.wm_feedback_enabled = True
    assert executor._wm_feedback("notes.md") == ""
    monkeypatch.setattr(executor, "_wm_ast", lambda path: (True, "same"))
    skipped = executor._wm_feedback("world_model.py", (True, "same"))
    assert "syntax tree is unchanged" in skipped

    monkeypatch.setattr(executor, "_run_python", lambda source, timeout: " probe ok ")
    feedback = executor._wm_feedback("world_model.py", (True, "before"))
    assert feedback.endswith("probe ok")


@pytest.fixture
def gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run the gate in a real workspace, against tycho's real wmlib.

    A fake wmlib would only check this code against contracts I assumed; the
    roles below depend on `_planner_key` and `_planner_successor` being the ones
    the planner actually uses.
    """
    root = _workspace(tmp_path)
    monkeypatch.chdir(root)
    # The probe runs inside a sandbox whose cwd is the workspace, so the seeded
    # wmlib.py is importable by name. Reproduce that, do not work around it.
    monkeypatch.syspath_prepend(str(root))

    def run(outcome: str, model: str = COUNTED_MODEL) -> dict:
        return wmgate_report(_load_model(model + outcome, root), wmlib)

    return run


def _workspace(root: Path) -> Path:
    """One level, two recorded DOWN moves, in the layout wmlib's frames() reads.

    Two rather than one so the sample starts away from the level's initial key:
    one move in, every backward operator is also a restart.
    """
    # Tycho seeds wmlib.py beside world_model.py; the seed template imports it.
    (root / "wmlib.py").write_text(Path(wmlib.__file__).read_text())
    level = root / "level_0"
    level.mkdir(parents=True)
    for turn in range(3):
        rows = ["00"] * 3
        rows[turn] = "10"
        (level / f"turn_{turn:03d}.txt").write_text("\n".join(rows) + "\n")
        (level / f"turn_{turn:03d}.json").write_text(json.dumps({"action": "DOWN"}))
    return root


def _load_model(source: str, root: Path):
    path = root / "world_model.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("probe_gate_world_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _roles(report: dict) -> dict[str, str]:
    return {entry["label"]: entry["role"] for entry in report["operators"]}
