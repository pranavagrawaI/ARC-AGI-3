from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROBE = Path(__file__).resolve().parents[1] / "public25_launcher_probe.py"


@pytest.fixture(autouse=True)
def exact_public_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYCHO_KAGGLE_MAX_ACTIONS", "8")


def load_probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("v12_multiaction_probe", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request_body(sequence: int) -> dict:
    remaining = 8 - sequence
    horizon = (
        f"Action horizon: {sequence} of 8 permitted environment actions have been "
        f"used; {remaining} remain including this decision."
    )
    terminal = (
        " This is the final permitted environment action. Its result will be "
        "recorded, but no next model decision is guaranteed."
        if sequence == 7
        else ""
    )
    if sequence == 7:
        single_description = (
            "Commit exactly one environment action. This is the final permitted "
            "environment action. This ends the turn. Its result will be recorded, "
            "but no next model decision is guaranteed."
        )
    else:
        single_description = (
            "Commit exactly one environment action. This ends the turn and its result "
            "will be recorded. If the game remains active, the next model decision "
            f"will have {remaining - 1} permitted environment actions remaining."
        )
    batch_cap = min(4, remaining)
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "turn = {"
                    f"'sequence': {sequence}, 'action_limit': 8, "
                    f"'actions_used': {sequence}, 'actions_remaining': {remaining}, "
                    f"'final_permitted_action': {sequence == 7}"
                    "}\n\n"
                    f"{horizon}{terminal}"
                ),
            }
        ],
        "parameters": {"max_tokens": 16_000, "tool_choice": "required"},
        "tools": [
            {
                "name": "take_action",
                "description": single_description,
                "parameters": {},
            },
            {
                "name": "take_actions",
                "description": (
                    f"Commit 1-{batch_cap} ordered environment actions. The Solver "
                    "observes its public result before releasing another."
                ),
                "parameters": {
                    "properties": {"actions": {"type": "array", "maxItems": batch_cap}}
                },
            },
        ],
    }


@pytest.mark.parametrize("sequence", (0, 6, 7))
def test_request_horizon_accepts_initial_penultimate_and_final(sequence: int) -> None:
    probe = load_probe()
    facts = probe.verify_action_horizon_request(
        game="ls20", sequence=sequence, body=request_body(sequence)
    )
    assert facts == {
        "action_limit": 8,
        "actions_used": sequence,
        "actions_remaining": 8 - sequence,
        "final_permitted_action": sequence == 7,
        "action_horizon_verified": True,
        "batch_cap": min(4, 8 - sequence),
        "batch_contract_verified": True,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda body: body["messages"][0].update(content="horizon omitted"),
        lambda body: body["tools"][0].update(
            description="This ends the turn; the successor arrives on the next call."
        ),
        lambda body: body["tools"][1]["parameters"]["properties"]["actions"].update(
            maxItems=2
        ),
    ),
)
def test_final_request_rejects_missing_or_false_batch_semantics(mutation) -> None:
    probe = load_probe()
    body = request_body(7)
    mutation(body)
    with pytest.raises(RuntimeError, match="action horizon"):
        probe.verify_action_horizon_request(game="ls20", sequence=7, body=body)


def test_all_request_bodies_have_sequence_exact_horizons_and_batch_caps() -> None:
    probe = load_probe()
    facts = [
        probe.verify_action_horizon_request(
            game="ls20",
            sequence=sequence,
            body=json.loads(json.dumps(request_body(sequence))),
        )
        for sequence in range(8)
    ]
    assert [fact["actions_remaining"] for fact in facts] == list(range(8, 0, -1))
    assert [fact["batch_cap"] for fact in facts] == [4, 4, 4, 4, 4, 3, 2, 1]
    assert [fact["final_permitted_action"] for fact in facts] == [False] * 7 + [True]


def test_model_evidence_accepts_zero_call_batch_continuations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = load_probe()
    monkeypatch.setenv("ARC_LLM_MAX_ACTOR_CALLS", "6")
    monkeypatch.setenv("ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL", "1")
    monkeypatch.setenv("ARC_LLM_MAX_TOKENS", "16000")
    monkeypatch.setenv("ARC_LLM_MAX_TOKENS_AFTER_INITIAL", "16000")
    monkeypatch.setenv("ARC_LLM_CONTEXT_WINDOW", "32768")
    sequences = [0, 3, 6]
    trace = []
    calls_root = tmp_path / "decision-checkpoints/calls"
    calls_root.mkdir(parents=True)
    for call_index, sequence in enumerate(sequences, 1):
        trace.append(
            {
                "call_index": call_index,
                "source": "actor",
                "phase": "decision",
                "sequence": sequence,
                "tool_calls": ["take_actions"],
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
    summary = probe.summarize_model_evidence(tmp_path, "ls20", 8)
    assert summary["actor_call_count"] == 3
    assert summary["model_decision_sequences"] == sequences
    assert summary["zero_call_action_sequences"] == [1, 2, 4, 5, 7]
    assert all(fact["batch_contract_verified"] for fact in summary["request_facts"])


def test_action_summary_reconstructs_single_and_batch_releases(tmp_path: Path) -> None:
    probe = load_probe()
    (tmp_path / "tool-trace.jsonl").write_text(
        json.dumps(
            {
                "name": "take_action",
                "status": "complete",
                "sequence": 3,
                "validated_arguments": {"action": "LEFT"},
            }
        )
        + "\n"
    )
    batch_events = [
        {"event": "submitted", "batch_id": "a", "batch_length": 3, "sequence": 0},
        {
            "event": "released",
            "batch_id": "a",
            "offset": 0,
            "sequence": 0,
            "action": {"action": "UP"},
        },
        {
            "event": "released",
            "batch_id": "a",
            "offset": 1,
            "sequence": 1,
            "action": {"action": "UP"},
        },
        {
            "event": "released",
            "batch_id": "a",
            "offset": 2,
            "sequence": 2,
            "action": {"action": "DOWN"},
        },
        {
            "event": "exhausted",
            "batch_id": "a",
            "sequence": 2,
            "model_calls_avoided": 2,
        },
    ]
    (tmp_path / "action-batch-trace.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in batch_events)
    )
    actions, batches = probe.summarize_environment_actions(tmp_path, "ls20", 4)
    assert [action["source"] for action in actions] == [
        "take_actions",
        "take_actions",
        "take_actions",
        "take_action",
    ]
    assert batches["submission_count"] == 1
    assert batches["model_calls_avoided"] == 2
    assert batches["uptake"] is True
