from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROBE = Path(__file__).resolve().parents[1] / "public25_launcher_probe.py"


@pytest.fixture(autouse=True)
def exact_public_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYCHO_KAGGLE_MAX_ACTIONS", "32")


def load_probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("v11_action_horizon_probe", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request_body(sequence: int) -> dict:
    remaining = 32 - sequence
    horizon = (
        f"Action horizon: {sequence} of 32 permitted environment actions have been "
        f"used; {remaining} remain including this decision."
    )
    terminal = (
        " This is the final permitted environment action. Its result will be "
        "recorded, but no next model decision is guaranteed."
        if sequence == 31
        else ""
    )
    if sequence == 31:
        description = (
            "Commit exactly one environment action. This is the final permitted "
            "environment action. This ends the turn. Its result will be recorded, "
            "but no next model decision is guaranteed."
        )
    else:
        description = (
            "Commit exactly one environment action. This ends the turn and its result "
            "will be recorded. If the game remains active, the next model decision "
            f"will have {remaining - 1} permitted environment actions remaining."
        )
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "turn = {"
                    f"'sequence': {sequence}, 'action_limit': 32, "
                    f"'actions_used': {sequence}, 'actions_remaining': {remaining}, "
                    f"'final_permitted_action': {sequence == 31}"
                    "}\n\n"
                    f"{horizon}{terminal}"
                ),
            }
        ],
        "parameters": {"max_tokens": 16_000, "tool_choice": "required"},
        "tools": [
            {
                "name": "take_action",
                "description": description,
                "parameters": {},
            }
        ],
    }


@pytest.mark.parametrize("sequence", (0, 30, 31))
def test_request_horizon_accepts_initial_penultimate_and_final(sequence: int) -> None:
    probe = load_probe()

    facts = probe.verify_action_horizon_request(
        game="ls20",
        sequence=sequence,
        body=request_body(sequence),
    )

    assert facts == {
        "action_limit": 32,
        "actions_used": sequence,
        "actions_remaining": 32 - sequence,
        "final_permitted_action": sequence == 31,
        "action_horizon_verified": True,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda body: body["messages"][0].update(content="horizon omitted"),
        lambda body: body["messages"][0].update(
            content=body["messages"][0]["content"].replace(
                "'actions_remaining': 1", "'actions_remaining': 2"
            )
        ),
        lambda body: body["tools"][0].update(
            description="This ends the turn; the successor arrives on the next call."
        ),
    ),
)
def test_final_request_rejects_missing_or_false_horizon_semantics(mutation) -> None:
    probe = load_probe()
    body = request_body(31)
    mutation(body)

    with pytest.raises(RuntimeError, match="action horizon"):
        probe.verify_action_horizon_request(game="ls20", sequence=31, body=body)


def test_all_32_request_bodies_have_sequence_exact_horizons() -> None:
    probe = load_probe()

    facts = [
        probe.verify_action_horizon_request(
            game="ls20",
            sequence=sequence,
            body=json.loads(json.dumps(request_body(sequence))),
        )
        for sequence in range(32)
    ]

    assert [fact["actions_remaining"] for fact in facts] == list(range(32, 0, -1))
    assert [fact["final_permitted_action"] for fact in facts] == [False] * 31 + [
        True
    ]


def test_session_summary_verifies_every_request_before_accepting_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = load_probe()
    monkeypatch.setenv("ARC_LLM_MAX_ACTOR_CALLS", "6")
    monkeypatch.setenv("ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL", "1")
    monkeypatch.setenv("ARC_LLM_MAX_TOKENS", "16000")
    monkeypatch.setenv("ARC_LLM_MAX_TOKENS_AFTER_INITIAL", "16000")
    monkeypatch.setenv("ARC_LLM_CONTEXT_WINDOW", "32768")
    trace = []
    calls_root = tmp_path / "decision-checkpoints/calls"
    calls_root.mkdir(parents=True)
    for sequence in range(32):
        call_index = sequence + 1
        trace.append(
            {
                "call_index": call_index,
                "source": "actor",
                "phase": "decision",
                "sequence": sequence,
                "tool_calls": ["take_action"],
            }
        )
        call_dir = calls_root / f"{call_index:06d}"
        call_dir.mkdir()
        request = {
            "call_index": call_index,
            "source": "actor",
            "decision_point": {"sequence": sequence},
            "replay_context": {
                "client_config": {"context_window": 32768, "num_retries": 0}
            },
            "request_body": request_body(sequence),
        }
        (call_dir / "request.json").write_text(json.dumps(request))
        (call_dir / "response.json").write_text("{}")
        (call_dir / "workspace.json").write_text("{}")
    (tmp_path / "llm-trace.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in trace)
    )
    (tmp_path / "worker-stdout.log").write_text("")
    (tmp_path / "worker-stderr.log").write_text("")

    summary = probe.summarize_model_evidence(tmp_path, "ls20", 32)

    assert summary["actor_call_count"] == 32
    assert len(summary["request_facts"]) == 32
    assert all(
        fact["action_horizon_verified"] for fact in summary["request_facts"]
    )
    assert summary["request_facts"][-1]["final_permitted_action"] is True
    assert summary["request_facts"][-1]["actions_remaining"] == 1
