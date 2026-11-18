"""The q6 join must persist Actor-call evidence the outer runtime can reconcile.

A Kaggle run on 2026-09-05 played ls20 correctly for 10 actions and then failed in
post-run accounting, because the reconciler knew only Tycho's session schema. The
same run also showed the recorder undercounting: the summarizer reached SGLang
without passing the interceptor, so 106 calls were persisted against 107 requests.
These tests pin both halves, and that they compose.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

KAGGLE = Path(__file__).resolve().parents[1]
PROBE = KAGGLE / "q6-nooa-candidate" / "nooa_public_probe.py"
RUNTIME = KAGGLE / "q6-nooa-candidate" / "exact_request_outer_runtime.py"
SOLVER_AGENT = KAGGLE / "q6-nooa-candidate" / "chassis" / "solver_agent.py"
CONTRACT = "62e4ad17228bfd9c230ece18e6cb96d6404470621068f9d33ed761a264952f4c"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_probe(monkeypatch, *, games: str = "ls20") -> ModuleType:
    monkeypatch.setenv("TYCHO_PUBLIC_GAME_IDS", games)
    monkeypatch.setenv("TYCHO_KAGGLE_MAX_ACTIONS", "10")
    return _load(PROBE, f"q6_probe_{games.replace(',', '_')}")


def write_session(
    root: Path,
    *,
    game: str,
    suffix: str,
    turns: list[int | None],
    actions: list[str],
) -> None:
    """One solver workspace whose model calls carry the given turn numbers."""
    session = root / f"{game}-{suffix}"
    (session / "workspace" / game).mkdir(parents=True)
    ipc = session / "ipc"
    ipc.mkdir()
    (session / "worker-metadata.json").write_text(
        json.dumps(
            {
                "join": "q6-nooa-join",
                "chassis": "q6-stall-authority",
                "contract_sha256": CONTRACT,
            }
        )
    )
    (session / "launch.json").write_text(json.dumps({"contract_sha256": CONTRACT}))
    (ipc / "actions.jsonl").write_text(json.dumps({"actions": actions}) + "\n")
    (ipc / "states.jsonl").write_text(json.dumps({"grid_rows": ["01"]}) + "\n")
    (ipc / "model_calls.jsonl").write_text(
        "".join(json.dumps({"turn": turn, "step": 0}) + "\n" for turn in turns)
    )


def test_model_evidence_counts_calls_by_decision_sequence(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_probe(monkeypatch)
    root = tmp_path / "workspaces"
    root.mkdir()
    write_session(
        root, game="ls20", suffix="a", turns=[0, 0, 1, 1, 1, 2], actions=["UP"]
    )

    sessions = module.summarize_nooa_sessions(root)
    evidence = sessions["games"]["ls20"]["model_evidence"]
    assert evidence["actor_call_count"] == 6
    assert evidence["actor_calls_by_sequence"] == {"0": 2, "1": 3, "2": 1}
    assert sum(evidence["actor_calls_by_sequence"].values()) == 6


def test_a_call_recorded_before_the_first_state_is_still_counted(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_probe(monkeypatch)
    root = tmp_path / "workspaces"
    root.mkdir()
    write_session(root, game="ls20", suffix="a", turns=[None, 0, 1], actions=["UP"])

    evidence = module.summarize_nooa_sessions(root)["games"]["ls20"]["model_evidence"]
    assert evidence["actor_call_count"] == 3
    assert evidence["actor_calls_by_sequence"] == {"0": 2, "1": 1}


def test_repeated_sequences_across_episodes_add_rather_than_overwrite(
    tmp_path: Path, monkeypatch
) -> None:
    """A GAME_OVER reset restarts turn numbering; only the totals must reconcile."""
    module = load_probe(monkeypatch)
    root = tmp_path / "workspaces"
    root.mkdir()
    write_session(root, game="ls20", suffix="a", turns=[0, 1], actions=["UP"])
    write_session(root, game="ls20", suffix="b", turns=[0, 0, 1], actions=["DOWN"])

    sessions = module.summarize_nooa_sessions(root)
    evidence = sessions["games"]["ls20"]["model_evidence"]
    assert sessions["games"]["ls20"]["episode_count"] == 2
    assert evidence["actor_call_count"] == 5
    assert evidence["actor_calls_by_sequence"] == {"0": 3, "1": 2}
    assert sessions["total_actor_calls"] == 5


def test_absent_or_empty_model_call_evidence_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_probe(monkeypatch)
    root = tmp_path / "workspaces"
    root.mkdir()
    write_session(root, game="ls20", suffix="a", turns=[0], actions=["UP"])
    calls = root / "ls20-a" / "ipc" / "model_calls.jsonl"

    calls.write_text("")
    with pytest.raises(RuntimeError, match="recorded no model calls"):
        module.summarize_nooa_sessions(root)

    calls.unlink()
    with pytest.raises(RuntimeError, match="model-call evidence is absent"):
        module.summarize_nooa_sessions(root)


def test_join_evidence_reconciles_through_the_outer_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    """The whole point: what the join emits, the reconciler accepts and totals."""
    module = load_probe(monkeypatch, games="ls20,s5i5")
    runtime = _load(RUNTIME, "exact_request_outer_runtime")
    root = tmp_path / "workspaces"
    root.mkdir()
    write_session(root, game="ls20", suffix="a", turns=[0, 0, 1], actions=["UP"])
    write_session(root, game="s5i5", suffix="a", turns=[0, 1], actions=["LEFT"])

    sessions = module.summarize_nooa_sessions(root)
    assert sessions["schema"] == "nooa-join-sessions-v4"

    expected = runtime.expected_launcher_solver_requests({"solver_sessions": sessions})
    assert expected == 5 == sessions["total_actor_calls"]


def test_the_previous_join_schema_is_no_longer_accepted() -> None:
    """v3 carried no Actor-call evidence, so accepting it would count nothing."""
    runtime = _load(RUNTIME, "exact_request_outer_runtime")
    with pytest.raises(RuntimeError, match="Solver session schema differs"):
        runtime.expected_launcher_solver_requests(
            {
                "solver_sessions": {
                    "schema": "nooa-join-sessions-v3",
                    "game_count": 1,
                    "total_submitted_batches": 6,
                    "games": {"ls20": {"submitted_batches": 6}},
                }
            }
        )


def test_the_summarizer_is_guarded_like_every_other_model_call() -> None:
    """Its calls reach SGLang, so they must be recorded and integrity-checked."""
    source = SOLVER_AGENT.read_text()
    assert 'self.event_manager.intercept("llm_call", _guard(summarizer=False))' in source
    assert 'for summarizer in getattr(self, "_summarizers", ()):' in source
    assert 'summarizer.event_manager.intercept("llm_call", _guard(summarizer=True))' in source
    assert '"summarizer": summarizer,' in source
