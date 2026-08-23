from __future__ import annotations

import json
from pathlib import Path

from tycho.workspace.workspace import GameWorkspace

from tycho_nooa.evidence_parity import compare_workspace, run_manifest


async def test_frozen_workspace_parity_crosses_multiple_query_pages(
    tmp_path: Path,
) -> None:
    workspace = GameWorkspace(
        "parity-game",
        root=str(tmp_path),
        render=False,
        available_actions=["UP"],
    )
    for turn in range(40):
        workspace.record(
            [[turn % 16, (turn + 1) % 16]],
            level=0,
            turn_in_level=turn,
            action="" if turn == 0 else "UP",
            available=["UP"],
        )

    result = await compare_workspace("synthetic-pagination", workspace.dir)

    assert result.status == "passed"
    assert result.frames == 40
    assert result.transitions == 39
    assert result.frame_pages == 2
    assert result.transition_pages == 2
    assert result.capability_calls == 4
    assert result.positive_control_detected is True
    assert result.mismatches == ()


async def test_real_tracked_checkpoint_matches_legacy_run_python_wmlib() -> None:
    repo = Path(__file__).resolve().parents[3]
    workspace = (
        repo
        / "analysis/candidate-retention/20260822-turn18/workspace/ar25"
    )

    result = await compare_workspace("tracked-turn18", workspace)

    assert result.status == "passed"
    assert result.frames == 19
    assert result.transitions == 18
    assert result.frontier == {"level": 0, "turn": 18}
    assert result.positive_control_detected is True


async def test_manifest_report_preserves_the_claim_and_checkpoint_identity(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[3]
    relative_workspace = (
        "analysis/goal-lifecycle/20260822-selected-checkpoint/workspace/ar25"
    )
    manifest = tmp_path / "checkpoints.json"
    manifest.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {"label": "tracked-turn0", "workspace": relative_workspace}
                ]
            }
        )
    )

    report = await run_manifest(manifest, repo)

    assert report.all_passed is True
    assert len(report.checkpoints) == 1
    assert report.checkpoints[0].workspace == relative_workspace
    assert report.checkpoints[0].frontier == {"level": 0, "turn": 0}
