from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "s0-public3-successor-budget-v9-candidate"
BUILDER = CANDIDATE / "build_notebook.py"
PROBE = CANDIDATE / "public25_launcher_probe.py"
NOTEBOOK = CANDIDATE / "s0_public3_continued_play.ipynb"
METADATA = CANDIDATE / "kernel-metadata.json"


def load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v9_notebook_is_rebuildable_and_has_the_expanded_successor_budget() -> None:
    builder = load(BUILDER, "successor_budget_v9_builder")
    before = NOTEBOOK.read_bytes()
    builder.build()
    assert NOTEBOOK.read_bytes() == before
    notebook = json.loads(before)
    source = "\n".join(
        "".join(cell.get("source", "")) for cell in notebook["cells"]
    )
    assert "TYCHO_KAGGLE_MAX_ACTIONS'] = '2'" in source
    assert "ARC_LLM_MAX_TOKENS_AFTER_INITIAL'] = '16000'" in source
    assert "ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL'] = '1'" in source
    assert "ARC_LLM_FORCE_ACTION_TOOL'] = '1'" in source
    assert "ARC_LLM_RETRIES'] = '0'" in source
    assert sha256(PROBE) == builder.PROBE_SHA256
    metadata = json.loads(METADATA.read_text())
    assert metadata["id"] == (
        "pranavagrawai/prototype-tycho-nooa-public-3-continued-play"
    )
    assert metadata["machine_shape"] == "NvidiaRtxPro6000"


def test_probe_requires_one_16k_action_only_successor_per_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = {
        "TYCHO_PUBLIC_GAME_IDS": "ls20,s5i5,sb26",
        "TYCHO_KAGGLE_MAX_ACTIONS": "2",
        "ARC_LLM_CONTEXT_WINDOW": "32768",
        "ARC_LLM_FORCE_ACTION_TOOL": "1",
        "ARC_LLM_MAX_ACTOR_CALLS": "6",
        "ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL": "1",
        "ARC_LLM_MAX_TOKENS": "16000",
        "ARC_LLM_MAX_TOKENS_AFTER_INITIAL": "16000",
        "ARC_LLM_RETRIES": "0",
        "ARC_LLM_TIMEOUT_SECONDS": "600",
        "TYCHO_MODE": "single",
        "TYCHO_NOOA_TREATMENT": "single_decision_discipline",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    probe = load(PROBE, "successor_budget_v9_probe")

    for game in probe.EXPECTED_GAMES:
        session = tmp_path / f"{game}-session"
        workspace = session / "workspace" / game
        workspace.mkdir(parents=True)
        metadata = {
            "agent": "tycho_nooa.TychoNooaAgent",
            "client_kind": "live",
            "decision_envelope": {
                "force_action_tool": True,
                "max_actor_calls": 6,
                "max_actor_calls_after_initial": 1,
                "max_output_tokens_per_call": 16_000,
                "max_output_tokens_per_call_after_initial": 16_000,
                "model_retries": 0,
                "model_timeout_seconds": 600.0,
            },
            "mode": "single",
            "sandbox_runtime": "host",
            "treatment": "single_decision_discipline",
            "level_boundary_policy": "deterministic_level_checkpoint",
            "workspace": str(workspace),
        }
        (session / "worker-metadata.json").write_text(json.dumps(metadata) + "\n")
        (session / "worker-stdout.log").write_text("")
        (session / "worker-stderr.log").write_text("")
        (session / "tool-trace.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "name": "take_action",
                        "status": "complete",
                        "sequence": sequence,
                        "validated_arguments": {"action": "UP"},
                    }
                )
                + "\n"
                for sequence in range(2)
            )
        )
        (session / "llm-trace.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "schema_version": 2,
                        "call_index": sequence + 1,
                        "source": "actor",
                        "phase": "decision",
                        "sequence": sequence,
                        "tool_calls": ["take_action"],
                    }
                )
                + "\n"
                for sequence in range(2)
            )
        )
        calls = session / "decision-checkpoints" / "calls"
        for sequence in range(2):
            call = calls / f"{sequence + 1:06d}"
            call.mkdir(parents=True)
            request = {
                "schema_version": 2,
                "call_index": sequence + 1,
                "source": "actor",
                "decision_point": {"sequence": sequence},
                "replay_context": {
                    "client_config": {"context_window": 32_768, "num_retries": 0}
                },
                "request_body": {
                    "parameters": {
                        "max_tokens": 16_000,
                        "tool_choice": "auto" if sequence == 0 else "required",
                    },
                    "tools": (
                        [{"name": "ls"}, {"name": "take_action"}]
                        if sequence == 0
                        else [{"name": "take_action"}]
                    ),
                },
            }
            (call / "request.json").write_text(json.dumps(request) + "\n")
            (call / "response.json").write_text("{}\n")
            (call / "workspace.json").write_text("{}\n")

    summary = probe.summarize_solver_sessions(tmp_path)
    assert summary["total_solver_actions"] == 6
    assert summary["total_actor_calls"] == 6
    for game in summary["games"].values():
        successor = game["model_evidence"]["request_facts"][1]
        assert successor["max_tokens"] == 16_000
        assert successor["tool_choice"] == "required"
        assert successor["tool_names"] == ["take_action"]

    request = (
        tmp_path
        / "ls20-session/decision-checkpoints/calls/000002/request.json"
    )
    drifted = json.loads(request.read_text())
    drifted["request_body"]["parameters"]["max_tokens"] = 4_096
    request.write_text(json.dumps(drifted) + "\n")
    with pytest.raises(RuntimeError, match="token ceiling differs"):
        probe.summarize_solver_sessions(tmp_path)
