from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
ROOT = DEPLOYMENT_ROOT.parents[1]
CANDIDATE_ROOT = DEPLOYMENT_ROOT / "s0-public25-kernel"
BUILDER_PATH = CANDIDATE_ROOT / "build_notebook.py"
NOTEBOOK_PATH = CANDIDATE_ROOT / "s0_public25.ipynb"
PROBE_PATH = CANDIDATE_ROOT / "public25_launcher_probe.py"
EVIDENCE_ROOT = (
    ROOT
    / "analysis/deployment-probes/20260901T054322Z-public25-kaggle-v1"
)


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public25_builder_is_deterministic_and_compiles() -> None:
    builder = load_module(BUILDER_PATH, "public25_builder")
    first = builder.build()
    second = builder.build()

    assert first == second
    builder.validate(first)
    assert first == json.loads(NOTEBOOK_PATH.read_text())
    for index, cell in enumerate(first["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"public25-cell-{index}", "exec")


def test_public25_candidate_is_phase_a_only_and_binds_all_games() -> None:
    builder = load_module(BUILDER_PATH, "public25_builder_contract")
    notebook = builder.build()
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    metadata = json.loads((CANDIDATE_ROOT / "kernel-metadata.json").read_text())
    public_manifest = json.loads(
        (EVIDENCE_ROOT / "public-environment-manifest.json").read_text()
    )

    assert public_manifest["game_count"] == 25
    assert public_manifest["file_count"] == 50
    assert "OPERATION_MODE=offline" in source
    assert "public-25 qualifier refuses competition rerun mode" in source
    assert "OPERATION_MODE=competition" not in source
    assert "--launcher-timeout-s', '7800'" in source
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaRtxPro6000"
    assert (
        "pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v5"
        in metadata["dataset_sources"]
    )
    assert "os.environ['TYCHO_PUBLIC_GAME_IDS'] = 'ls20,s5i5,sb26'" in source
    assert "os.environ['TYCHO_KAGGLE_MAX_ACTIONS'] = '1'" in source
    assert "os.environ['ARC_LLM_RETRIES'] = '0'" in source
    assert "os.environ['ARC_LLM_FORCE_ACTION_TOOL'] = '1'" in source
    assert "os.environ['ARC_LLM_MAX_ACTOR_CALLS'] = '26'" in source
    assert "os.environ['ARC_LLM_MAX_TOKENS'] = '16000'" in source
    assert "os.environ['ARC_LLM_TIMEOUT_SECONDS'] = '600'" in source
    assert "os.environ['TYCHO_NOOA_TREATMENT'] = 'decision_discipline'" in source
    assert "os.environ['TYCHO_NOOA_WORKER_TIMEOUT_SECONDS'] = '7200'" in source


def test_public25_probe_refuses_competition_rerun(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            "--base-url",
            "http://127.0.0.1:30000/v1",
            "--model",
            "model",
            "--workspace-root",
            str(tmp_path / "workspace"),
            "--output",
            str(tmp_path / "result.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "KAGGLE_IS_COMPETITION_RERUN": "1"},
    )

    assert completed.returncode != 0
    assert "refuses competition rerun" in completed.stderr
    assert not (tmp_path / "result.json").exists()


def test_scorecard_parser_rejects_partial_but_preserves_empty_local_result(
    tmp_path: Path,
) -> None:
    probe = load_module(PROBE_PATH, "public25_probe_parser")
    scorecard = {
        "score": 0.25,
        "environments": [
            {"id": game, "runs": []} for game in probe.EXPECTED_GAMES
        ],
    }
    log = tmp_path / "logs.log"
    log.write_text(
        "2026-09-01 | INFO | --- FINAL SCORECARD REPORT ---\n"
        "2026-09-01 | INFO | "
        + json.dumps(scorecard, indent=2)
        + "\n2026-09-01 | INFO | done\n"
    )

    assert probe.parse_scorecard(log) == scorecard
    scorecard["environments"] = []
    log.write_text(
        "--- FINAL SCORECARD REPORT ---\n" + json.dumps(scorecard, indent=2)
    )
    assert probe.parse_scorecard(log) == scorecard
    scorecard["environments"] = [
        {"id": game, "runs": []} for game in probe.EXPECTED_GAMES[:-1]
    ]
    log.write_text(
        "--- FINAL SCORECARD REPORT ---\n" + json.dumps(scorecard, indent=2)
    )
    try:
        probe.parse_scorecard(log)
    except RuntimeError as error:
        assert "environment IDs differ" in str(error)
    else:
        raise AssertionError("partial scorecard was accepted")


def write_recording(path: Path, game: str, *, frames: int = 80) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "timestamp": "2026-09-01T00:00:00+00:00",
                    "data": {
                        "game_id": f"{game}-version",
                        "state": "NOT_FINISHED",
                        "levels_completed": 0,
                        "win_levels": 6,
                        "full_reset": False,
                        "action_input": {"id": 1, "data": {}},
                    },
                }
            )
            + "\n"
            for _ in range(frames)
        )
    )


def test_recording_summary_requires_exact_complete_public_inventory(
    tmp_path: Path,
) -> None:
    probe = load_module(PROBE_PATH, "public25_probe_recordings")
    for game in probe.EXPECTED_GAMES:
        write_recording(tmp_path / f"{game}.myagent.80.guid.recording.jsonl", game)

    summary = probe.summarize_recordings(tmp_path)
    assert summary["game_count"] == 25
    assert summary["total_recorded_action_results"] == 25 * 80
    assert summary["evidence_kind"] == "official_agent_recordings_not_official_score"

    next(tmp_path.glob("wa30.*")).unlink()
    try:
        probe.summarize_recordings(tmp_path)
    except RuntimeError as error:
        assert "recording game IDs differ" in str(error)
    else:
        raise AssertionError("partial recording inventory was accepted")


def test_nonterminal_recording_must_exhaust_configured_official_loop(
    tmp_path: Path,
) -> None:
    probe = load_module(PROBE_PATH, "public25_probe_recording_cap")
    for game in probe.EXPECTED_GAMES:
        frames = 79 if game == "wa30" else 80
        write_recording(
            tmp_path / f"{game}.myagent.80.guid.recording.jsonl",
            game,
            frames=frames,
        )
    try:
        probe.summarize_recordings(tmp_path)
    except RuntimeError as error:
        assert "did not exhaust the configured official loop" in str(error)
    else:
        raise AssertionError("truncated nonterminal recording was accepted")


def test_solver_session_summary_requires_exact_actions_and_ready_metadata(
    tmp_path: Path,
) -> None:
    probe = load_module(PROBE_PATH, "public25_probe_solver_sessions")
    for game in probe.EXPECTED_GAMES:
        session = tmp_path / f"{game}-session"
        (session / "workspace" / game).mkdir(parents=True)
        (session / "worker-metadata.json").write_text(
            json.dumps({"treatment": "decision_discipline"}) + "\n"
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
                for sequence in range(probe.PUBLIC_MAX_ACTIONS)
            )
        )

    summary = probe.summarize_solver_sessions(tmp_path)
    assert summary["game_count"] == 25
    assert summary["total_solver_actions"] == 25 * 80
    assert all(
        game["worker_metadata"]["treatment"] == "decision_discipline"
        for game in summary["games"].values()
    )
