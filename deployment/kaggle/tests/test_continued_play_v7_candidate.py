from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

import pytest

DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = DEPLOYMENT_ROOT / "s0-public3-continued-play-v7-candidate"
BUILDER_PATH = CANDIDATE_ROOT / "build_notebook.py"
NOTEBOOK_PATH = CANDIDATE_ROOT / "s0_public3_continued_play.ipynb"
PROBE_PATH = CANDIDATE_ROOT / "public25_launcher_probe.py"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def notebook_source() -> str:
    builder = load_module(BUILDER_PATH, "continued_play_v7_builder_source")
    notebook = builder.build()
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_continued_play_notebook_is_deterministic_and_compiles() -> None:
    builder = load_module(BUILDER_PATH, "continued_play_v7_builder")
    first = builder.build()
    second = builder.build()

    assert first == second
    assert first == json.loads(NOTEBOOK_PATH.read_text())
    builder.validate(first)
    for index, cell in enumerate(first["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"continued-play-cell-{index}", "exec")


def test_continued_play_candidate_binds_the_exact_envelope() -> None:
    source = notebook_source()
    metadata = json.loads((CANDIDATE_ROOT / "kernel-metadata.json").read_text())

    assert "public-25 qualifier refuses competition rerun mode" in source
    assert "OPERATION_MODE=offline" in source
    assert "OPERATION_MODE=competition" not in source
    assert "kaggle competitions submit" not in source
    assert "os.environ['TYCHO_PUBLIC_GAME_IDS'] = 'ls20,s5i5,sb26'" in source
    assert "os.environ['TYCHO_KAGGLE_MAX_ACTIONS'] = '32'" in source
    assert "os.environ['ARC_LLM_CONTEXT_WINDOW'] = '32768'" in source
    assert "os.environ['ARC_LLM_MAX_ACTOR_CALLS'] = '6'" in source
    assert "os.environ['ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL'] = '1'" in source
    assert "os.environ['ARC_LLM_MAX_TOKENS'] = '16000'" in source
    assert "os.environ['ARC_LLM_MAX_TOKENS_AFTER_INITIAL'] = '4096'" in source
    assert "os.environ['ARC_LLM_FORCE_ACTION_TOOL'] = '1'" in source
    assert "os.environ['ARC_LLM_RETRIES'] = '0'" in source
    assert "os.environ['ARC_LLM_TIMEOUT_SECONDS'] = '600'" in source
    assert "os.environ['TYCHO_MODE'] = 'single'" in source
    assert (
        "os.environ['TYCHO_NOOA_TREATMENT'] = 'single_decision_discipline'"
        in source
    )
    assert "'--launcher-timeout-s', '21600'" in source
    assert (
        "e440ef5bd8c2e898b7fdbb20d2f0c8528cfbd3dda6f70fdb4b96f8112a942851"
        in source
    )
    assert (
        "175374b1df444fb47c4ce6c37e656b3c7d9ab8418ae697a86fa201d7276f9580"
        in source
    )
    assert "'--no-cache', '--no-index'" in source
    assert "str(solver_wheel_path)" in source
    assert "source_manifest['packaged_sources']" in source
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaRtxPro6000"
    assert (
        "pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v7"
        in metadata["dataset_sources"]
    )


def test_probe_propagates_its_evidence_root_to_the_official_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TYCHO_PUBLIC_GAME_IDS", "ls20,s5i5,sb26")
    probe = load_module(PROBE_PATH, "continued_play_v7_probe_workspace_root")
    workspace_root = tmp_path / "solver-workspaces"

    environment = probe.official_launcher_environment(
        base_url="http://127.0.0.1:30000/v1",
        model="local/model",
        workspace_root=workspace_root,
        worker_python=Path("/tmp/solver-venv/bin/python"),
    )

    assert environment["TYCHO_KAGGLE_WORKSPACE_ROOT"] == str(workspace_root)


def test_recordings_expose_level_advance_and_win_boundaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TYCHO_PUBLIC_GAME_IDS", "ls20,s5i5,sb26")
    probe = load_module(PROBE_PATH, "continued_play_v7_recording_boundaries")
    rows_by_game = {
        "ls20": [
            {"levels_completed": 0, "state": "NOT_FINISHED"},
            {"levels_completed": 1, "state": "WIN"},
        ],
        "s5i5": [
            {"levels_completed": 0, "state": "NOT_FINISHED"},
            {"levels_completed": 0, "state": "NOT_FINISHED"},
        ],
        "sb26": [
            {"levels_completed": 0, "state": "NOT_FINISHED"},
            {"levels_completed": 1, "state": "NOT_FINISHED"},
        ],
    }
    for game, rows in rows_by_game.items():
        path = tmp_path / f"{game}.recording.jsonl"
        records = []
        for index, row in enumerate(rows):
            records.append(
                {
                    "data": {
                        "game_id": f"{game}-fixture",
                        "action_input": {"id": "RESET" if index == 0 else "UP"},
                        "win_levels": 1,
                        "full_reset": index == 0,
                        **row,
                    }
                }
            )
        path.write_text("".join(json.dumps(record) + "\n" for record in records))

    summary = probe.summarize_recordings(tmp_path)
    assert summary["games"]["ls20"]["boundary_levels"] == [0]
    assert summary["games"]["ls20"]["boundary_events"] == [
        {
            "result_index": 1,
            "level": 0,
            "level_complete": True,
            "win": True,
        }
    ]
    assert summary["games"]["s5i5"]["boundary_levels"] == []
    assert summary["games"]["sb26"]["boundary_levels"] == [0]


def test_continued_play_probe_requires_96_actions_and_successor_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment = {
        "TYCHO_PUBLIC_GAME_IDS": "ls20,s5i5,sb26",
        "TYCHO_KAGGLE_MAX_ACTIONS": "32",
        "ARC_LLM_CONTEXT_WINDOW": "32768",
        "ARC_LLM_FORCE_ACTION_TOOL": "1",
        "ARC_LLM_MAX_ACTOR_CALLS": "6",
        "ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL": "1",
        "ARC_LLM_MAX_TOKENS": "16000",
        "ARC_LLM_MAX_TOKENS_AFTER_INITIAL": "4096",
        "ARC_LLM_RETRIES": "0",
        "ARC_LLM_TIMEOUT_SECONDS": "600",
        "TYCHO_MODE": "single",
        "TYCHO_NOOA_TREATMENT": "single_decision_discipline",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    probe = load_module(PROBE_PATH, "continued_play_v7_probe")
    envelope = {
        "force_action_tool": True,
        "max_actor_calls": 6,
        "max_actor_calls_after_initial": 1,
        "max_output_tokens_per_call": 16_000,
        "max_output_tokens_per_call_after_initial": 4_096,
        "model_retries": 0,
        "model_timeout_seconds": 600.0,
    }
    metadata = {
        "agent": "tycho_nooa.TychoNooaAgent",
        "client_kind": "live",
        "decision_envelope": envelope,
        "mode": "single",
        "sandbox_runtime": "host",
        "treatment": "single_decision_discipline",
        "level_boundary_policy": "deterministic_level_checkpoint",
    }
    for game in probe.EXPECTED_GAMES:
        session = tmp_path / f"{game}-session"
        workspace = session / "workspace" / game
        workspace.mkdir(parents=True)
        (session / "worker-metadata.json").write_text(
            json.dumps({**metadata, "workspace": str(workspace)}) + "\n"
        )
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
                for sequence in range(32)
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
                for sequence in range(32)
            )
        )
        (session / "worker-stdout.log").write_text("")
        (session / "worker-stderr.log").write_text("")
        calls_root = session / "decision-checkpoints" / "calls"
        for sequence in range(32):
            call = calls_root / f"{sequence + 1:06d}"
            call.mkdir(parents=True)
            request = {
                "schema_version": 2,
                "call_index": sequence + 1,
                "source": "actor",
                "decision_point": {
                    "sequence": sequence,
                    "level": 0,
                    "turn_in_level": sequence,
                    "phase": "decision",
                },
                "replay_context": {
                    "client_config": {
                        "context_window": 32_768,
                        "num_retries": 0,
                    }
                },
                "request_body": {
                    "parameters": {
                        "max_tokens": 16_000 if sequence == 0 else 4_096,
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
    assert summary["game_count"] == 3
    assert summary["total_solver_actions"] == 96
    assert summary["total_actor_calls"] == 96
    assert all(
        game["worker_metadata"]["decision_envelope"] == envelope
        for game in summary["games"].values()
    )

    drifted_path = tmp_path / "ls20-session" / "worker-metadata.json"
    drifted = json.loads(drifted_path.read_text())
    drifted["decision_envelope"]["max_actor_calls_after_initial"] = 6
    drifted_path.write_text(json.dumps(drifted) + "\n")
    with pytest.raises(RuntimeError, match="decision envelope differs"):
        probe.summarize_solver_sessions(tmp_path)

    drifted_path.write_text(
        json.dumps({**metadata, "workspace": str(tmp_path / "restored")}) + "\n"
    )
    with pytest.raises(RuntimeError, match="workspace identity differs"):
        probe.summarize_solver_sessions(tmp_path)

    expected_workspace = tmp_path / "ls20-session" / "workspace" / "ls20"
    drifted_path.write_text(
        json.dumps({**metadata, "workspace": str(expected_workspace)}) + "\n"
    )
    drifted = json.loads(drifted_path.read_text())
    drifted["level_boundary_policy"] = "model_scribe"
    drifted_path.write_text(json.dumps(drifted) + "\n")
    with pytest.raises(RuntimeError, match="decision envelope differs"):
        probe.summarize_solver_sessions(tmp_path)

    drifted_path.write_text(
        json.dumps({**metadata, "workspace": str(expected_workspace)}) + "\n"
    )
    llm_trace_path = tmp_path / "ls20-session" / "llm-trace.jsonl"
    trace = [json.loads(line) for line in llm_trace_path.read_text().splitlines()]
    trace[1]["source"] = "builder"
    llm_trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in trace)
    )
    with pytest.raises(RuntimeError, match="non-Actor model call"):
        probe.summarize_solver_sessions(tmp_path)

    trace[1]["source"] = "actor"
    llm_trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in trace[:2])
    )
    tool_trace_path = tmp_path / "ls20-session" / "tool-trace.jsonl"
    tool_trace_path.write_text(
        "\n".join(tool_trace_path.read_text().splitlines()[:2]) + "\n"
    )
    calls_root = tmp_path / "ls20-session" / "decision-checkpoints" / "calls"
    for call in sorted(calls_root.iterdir())[2:]:
        shutil.rmtree(call)
    checkpoint = expected_workspace / "level_0/level_boundary_checkpoint.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "tycho_nooa host treatment",
                "treatment": "single_decision_discipline",
                "policy": "deterministic_level_checkpoint",
                "model_scribe_invoked": False,
                "level": 0,
                "carry_forward_note": "notes/level_0_insights.md",
            }
        )
        + "\n"
    )
    note = expected_workspace / "notes/level_0_insights.md"
    note.parent.mkdir()
    note.write_text("# Level 0 Insights\n\nDeterministic carry-forward evidence.\n")
    boundaries = {"ls20": [0], "s5i5": [], "sb26": []}
    terminal_summary = probe.summarize_solver_sessions(
        tmp_path,
        terminal_states={
            "ls20": "WIN",
            "s5i5": "NOT_FINISHED",
            "sb26": "GAME_OVER",
        },
        expected_boundary_levels=boundaries,
    )
    assert terminal_summary["total_solver_actions"] == 66
    assert terminal_summary["total_actor_calls"] == 66
    assert terminal_summary["games"]["ls20"]["level_boundaries"]["levels"] == [0]

    note.unlink()
    with pytest.raises(RuntimeError, match="carry-forward note"):
        probe.summarize_solver_sessions(
            tmp_path,
            terminal_states={
                "ls20": "WIN",
                "s5i5": "NOT_FINISHED",
                "sb26": "GAME_OVER",
            },
            expected_boundary_levels=boundaries,
        )
    external_note = tmp_path / "external-note.md"
    external_note.write_text("outside workspace\n")
    note.symlink_to(external_note)
    with pytest.raises(RuntimeError, match="traverses a symlink"):
        probe.summarize_solver_sessions(
            tmp_path,
            terminal_states={
                "ls20": "WIN",
                "s5i5": "NOT_FINISHED",
                "sb26": "GAME_OVER",
            },
            expected_boundary_levels=boundaries,
        )
    note.unlink()
    note.write_text("restored\n")
    changed = json.loads(checkpoint.read_text())
    changed["model_scribe_invoked"] = True
    checkpoint.write_text(json.dumps(changed) + "\n")
    with pytest.raises(RuntimeError, match="checkpoint differs"):
        probe.summarize_solver_sessions(
            tmp_path,
            terminal_states={
                "ls20": "WIN",
                "s5i5": "NOT_FINISHED",
                "sb26": "GAME_OVER",
            },
            expected_boundary_levels=boundaries,
        )
