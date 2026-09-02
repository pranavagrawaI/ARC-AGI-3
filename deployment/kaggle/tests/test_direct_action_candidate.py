from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = DEPLOYMENT_ROOT / "s0-public3-direct-action-candidate"
BUILDER_PATH = CANDIDATE_ROOT / "build_notebook.py"
NOTEBOOK_PATH = CANDIDATE_ROOT / "s0_public3_direct_action.ipynb"
PROBE_PATH = CANDIDATE_ROOT / "public25_launcher_probe.py"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_action_notebook_is_deterministic_and_exact() -> None:
    builder = load_module(BUILDER_PATH, "direct_action_builder")
    notebook = builder.build()
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert notebook == builder.build()
    assert notebook == json.loads(NOTEBOOK_PATH.read_text())
    builder.validate(notebook)
    assert "os.environ['ARC_LLM_MAX_ACTOR_CALLS'] = '1'" in source
    assert "os.environ['TYCHO_MODE'] = 'no_world_model'" in source
    assert "os.environ['TYCHO_NOOA_TREATMENT'] = 'control'" in source
    assert "os.environ['ARC_LLM_FORCE_ACTION_TOOL'] = '1'" in source
    assert "decision_discipline" not in source


def test_direct_action_probe_binds_the_worker_decision_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment = {
        "TYCHO_PUBLIC_GAME_IDS": "ls20,s5i5,sb26",
        "TYCHO_KAGGLE_MAX_ACTIONS": "1",
        "ARC_LLM_FORCE_ACTION_TOOL": "1",
        "ARC_LLM_MAX_ACTOR_CALLS": "1",
        "ARC_LLM_MAX_TOKENS": "16000",
        "ARC_LLM_RETRIES": "0",
        "ARC_LLM_TIMEOUT_SECONDS": "600",
        "TYCHO_MODE": "no_world_model",
        "TYCHO_NOOA_TREATMENT": "control",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    probe = load_module(PROBE_PATH, "direct_action_probe")
    metadata = {
        "agent": "tycho_nooa.TychoNooaAgent",
        "client_kind": "live",
        "decision_envelope": {
            "force_action_tool": True,
            "max_actor_calls": 1,
            "max_output_tokens_per_call": 16000,
            "model_retries": 0,
            "model_timeout_seconds": 600.0,
        },
        "mode": "no_world_model",
        "sandbox_runtime": "host",
        "treatment": "control",
    }
    for game in probe.EXPECTED_GAMES:
        session = tmp_path / f"{game}-session"
        (session / "workspace" / game).mkdir(parents=True)
        (session / "worker-metadata.json").write_text(
            json.dumps({**metadata, "workspace": str(session / "workspace" / game)})
            + "\n"
        )
        (session / "tool-trace.jsonl").write_text(
            json.dumps(
                {
                    "name": "take_action",
                    "status": "complete",
                    "sequence": 0,
                    "validated_arguments": {"action": "UP"},
                }
            )
            + "\n"
        )

    summary = probe.summarize_solver_sessions(tmp_path)
    assert summary["game_count"] == 3
    assert summary["total_solver_actions"] == 3

    drifted_path = tmp_path / "ls20-session" / "worker-metadata.json"
    drifted = json.loads(drifted_path.read_text())
    drifted["decision_envelope"]["max_actor_calls"] = 26
    drifted_path.write_text(json.dumps(drifted) + "\n")
    with pytest.raises(RuntimeError, match="decision envelope differs"):
        probe.summarize_solver_sessions(tmp_path)
