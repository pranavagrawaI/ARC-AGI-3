from __future__ import annotations

import importlib.util
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/derive_public25_runtime_projection.py"
V10_RUN = (
    ROOT.parents[1]
    / "analysis/deployment-probes/20260901T-public3-full-timing-v10"
)


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("public25_runtime_projection", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def log_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def exact_request_count_facts(total_actor_calls: int) -> dict[str, object]:
    total_requests = total_actor_calls + 1
    return {
        "expected_runtime_owned_warmup_requests": 1,
        "observed_runtime_owned_warmup_requests": 1,
        "observed_successful_runtime_owned_warmup_requests": 1,
        "expected_launcher_solver_requests": total_actor_calls,
        "observed_launcher_solver_requests": total_actor_calls,
        "observed_successful_launcher_solver_requests": total_actor_calls,
        "expected_total_chat_completion_accesses": total_requests,
        "observed_total_chat_completion_accesses": total_requests,
        "observed_total_successful_chat_completion_accesses": total_requests,
        "ready_marker_count": 1,
        "successful_freeze_gc_accesses": 1,
        "whole_model_request_count_verified": True,
    }


def copy_source_identity_artifacts(root: Path) -> None:
    for name in (
        "EXPERIMENT-CARD.md",
        "kaggle-v10-launch-identity.json",
        "kaggle-v10-push-record.json",
        "kernel-v8-preterminal-source-equality.json",
        "kernel-v8-preterminal-source-pull-record.json",
        "kernel-v8-source-preterminal-sha256-manifest.json",
    ):
        shutil.copy2(V10_RUN / name, root / name)
    shutil.copytree(
        V10_RUN / "kernel-v8-source-preterminal",
        root / "kernel-v8-source-preterminal",
    )


def rebind_artifacts(
    module: ModuleType,
    root: Path,
    *,
    refresh_stdout_facts: bool = False,
    refresh_request_facts: bool = False,
) -> None:
    result_root = root / "tycho-nooa-public25"
    outer_path = result_root / "result.json"
    official_path = result_root / "official-launcher-result.json"
    stdout_path = result_root / "public25-launcher.stdout.log"
    system_path = result_root / "system-before.json"
    reconciliation_path = result_root / "run-reconciliation.json"
    official = json.loads(official_path.read_text())
    if refresh_stdout_facts:
        official["stdout"] = module.file_facts(stdout_path, root)
        write_json(official_path, official)
    outer = json.loads(outer_path.read_text())
    outer["official_launcher"] = official
    if refresh_request_facts:
        outer["request_count_facts"] = exact_request_count_facts(
            official["solver_sessions"]["total_actor_calls"]
        )
    write_json(outer_path, outer)
    policy = json.loads(module.DEFAULT_POLICY.read_text())
    source_manifest = {
        "outer_result": module.file_facts(outer_path, root),
        "official_launcher_result": module.file_facts(official_path, root),
        "official_launcher_stdout": module.file_facts(stdout_path, root),
        "system_preflight": module.file_facts(system_path, root),
        **module.source_identity_file_facts(root, policy),
    }
    reconciliation = json.loads(reconciliation_path.read_text())
    reconciliation["source_artifact_manifest_sha256"] = module.sha256_bytes(
        module.canonical_json(source_manifest)
    )
    write_json(reconciliation_path, reconciliation)


def valid_output(module: ModuleType, root: Path) -> tuple[Path, Path, Path]:
    result_root = root / "tycho-nooa-public25"
    result_root.mkdir(parents=True)
    started = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    created = {
        "ls20": started + timedelta(seconds=1),
        "s5i5": started + timedelta(seconds=2),
        "sb26": started + timedelta(seconds=3),
    }
    first_gap = {"ls20": 10, "s5i5": 11, "sb26": 12}
    successor_gap = {"ls20": 4, "s5i5": 5, "sb26": 6}
    events_by_game: dict[str, list[datetime]] = {}
    lines: list[tuple[datetime, str]] = []
    for game in module.EXPECTED_GAMES:
        lines.append(
            (
                created[game],
                (
                    f"{log_time(created[game])} | INFO | created new recording for "
                    f"{game}.myagent.1 into /tmp/{game}.recording.jsonl"
                ),
            )
        )
        current = created[game] + timedelta(seconds=first_gap[game])
        events_by_game[game] = []
        for index in range(module.EXPECTED_ACTIONS_PER_GAME):
            if index:
                current += timedelta(seconds=successor_gap[game])
            events_by_game[game].append(current)
            lines.append(
                (
                    current,
                    (
                        f"{log_time(current)} | INFO | {game} - ACTION1: count {index}, "
                        "levels completed 0, avg fps 0.0)"
                    ),
                )
            )
        available = current + timedelta(seconds=7)
        lines.append(
            (
                available,
                (
                    f"{log_time(available)} | INFO | recording for {game}.myagent.1 "
                    f"is available in /tmp/{game}.recording.jsonl"
                ),
            )
        )
    stdout_path = result_root / "public25-launcher.stdout.log"
    stdout_path.write_text("\n".join(line for _, line in sorted(lines)) + "\n")

    progress_games: dict[str, object] = {}
    session_games: dict[str, object] = {}
    for game in module.EXPECTED_GAMES:
        progress_games[game] = {
            "recorded_action_results": module.EXPECTED_ACTIONS_PER_GAME,
            "final_state": "NOT_FINISHED",
        }
        session_games[game] = {
            "actions": [
                {"sequence": index, "validated_arguments": {"action": "UP"}}
                for index in range(module.EXPECTED_ACTIONS_PER_GAME)
            ],
            "worker_metadata": {
                **module.EXPECTED_WORKER_METADATA,
                "workspace": f"/kaggle/working/{game}",
            },
            "model_evidence": {
                "actor_call_count": 37,
                "actor_calls_by_sequence": {
                    str(index): 6 if index == 0 else 1
                    for index in range(module.EXPECTED_ACTIONS_PER_GAME)
                },
            },
        }
    official_ended = max(values[-1] for values in events_by_game.values()) + timedelta(
        seconds=8
    )
    official = {
        "schema": module.EXPECTED_OFFICIAL_SCHEMA,
        "status": "passed",
        "returncode": 0,
        "launcher_returncode": 0,
        "timed_out": False,
        "residual_group_term_required": False,
        "started_at": started.isoformat(),
        "ended_at": official_ended.isoformat(),
        "operation_mode": "offline",
        "model": module.EXPECTED_MODEL,
        "public_games": list(module.EXPECTED_GAMES),
        "stdout": module.file_facts(stdout_path, root),
        "public_progress": {
            "game_count": 3,
            "total_recorded_action_results": 96,
            "games": progress_games,
        },
        "solver_sessions": {
            "game_count": 3,
            "total_solver_actions": 96,
            "total_actor_calls": 111,
            "games": session_games,
        },
    }
    official_path = result_root / "official-launcher-result.json"
    write_json(official_path, official)
    outer_path = result_root / "result.json"
    outer = {
        "schema": module.EXPECTED_OUTER_SCHEMA,
        "mode": module.EXPECTED_OUTER_MODE,
        "status": "passed",
        "started_at": (started - timedelta(seconds=2)).isoformat(),
        "ended_at": (official_ended + timedelta(seconds=1)).isoformat(),
        "lifecycle": {"status": "passed"},
        "r8_l3b_gate": {
            "lifecycle_passed": True,
            "request_path_passed": True,
            "teardown_log_accepted": True,
        },
        "request_count_facts": exact_request_count_facts(111),
        "official_launcher": official,
    }
    write_json(outer_path, outer)
    system_path = result_root / "system-before.json"
    write_json(
        system_path,
        {
            "captured_at": (started - timedelta(seconds=1)).isoformat(),
            "gpu": {
                "command": module.EXPECTED_GPU_COMMAND,
                "returncode": 0,
                "stdout": (
                    "NVIDIA RTX PRO 6000 Blackwell Server Edition, "
                    "97887, 97250, 580.159.04\n"
                ),
            },
        },
    )
    copy_source_identity_artifacts(root)
    policy = json.loads(module.DEFAULT_POLICY.read_text())
    source_manifest = {
        "outer_result": module.file_facts(outer_path, root),
        "official_launcher_result": module.file_facts(official_path, root),
        "official_launcher_stdout": module.file_facts(stdout_path, root),
        "system_preflight": module.file_facts(system_path, root),
        **module.source_identity_file_facts(root, policy),
    }
    write_json(
        result_root / "run-reconciliation.json",
        {
            "schema": module.RECONCILIATION_SCHEMA,
            "status": "canonical_valid",
            "kernel": {
                "slug": policy["expected_candidate_identity"][
                    "observed_kernel_slug"
                ],
                "version": policy["expected_candidate_identity"][
                    "observed_kernel_version"
                ],
            },
            "candidate_identity": policy["expected_candidate_identity"],
            "candidate_artifact_manifest_sha256": policy[
                "expected_candidate_artifact_manifest_sha256"
            ],
            "source_artifact_manifest_sha256": module.sha256_bytes(
                module.canonical_json(source_manifest)
            ),
            "decision": {
                "canonical_source_identity_verified": True,
                "metadata_identity_verified": True,
                "output_preserved_before_derivation": True,
            },
        },
    )
    return outer_path, official_path, stdout_path


def test_exact_3x32_stream_produces_maximum_bound_projection(tmp_path: Path) -> None:
    module = load_module()
    _, official_path, _ = valid_output(module, tmp_path)

    result = module.derive(tmp_path)

    projection = result["runtime_projection"]
    assert projection["initial_decision_seconds"] == 12
    assert projection["successor_decision_seconds"] == 6
    assert projection["per_game_overhead_seconds"] == 10
    assert projection["per_game_scheduler_reserve_seconds"] == 90
    assert projection["safety_multiplier"] == 1.25
    assert projection["source_result_sha256"] == module.sha256_file(official_path)
    assert result["observations"]["successor_interval_count"] == 93
    assert result["cohort"] == {
        "solver_decision_ceiling": 1_056,
        "whole_model_request_ceiling": 1_181,
    }
    assert [
        game
        for generation in result["schedule"]["generations"]
        for game in generation["games"]
    ] == list(module.load_harness().RUN_ORDER)


def test_derivation_is_byte_deterministic(tmp_path: Path) -> None:
    module = load_module()
    valid_output(module, tmp_path)

    first = module.derive(tmp_path)
    second = module.derive(tmp_path)

    assert module.canonical_json(first) == module.canonical_json(second)


def test_canonical_reconciliation_and_embedded_official_are_required(
    tmp_path: Path,
) -> None:
    module = load_module()
    _outer_path, _, _ = valid_output(module, tmp_path)
    reconciliation_path = next(tmp_path.rglob("run-reconciliation.json"))
    reconciliation = json.loads(reconciliation_path.read_text())
    reconciliation["candidate_identity"]["frozen_notebook_raw_sha256"] = "0" * 64
    write_json(reconciliation_path, reconciliation)
    with pytest.raises(RuntimeError, match="reconciliation identity differs"):
        module.derive(tmp_path)

    valid_output(module, tmp_path / "embedded")
    embedded_outer = next((tmp_path / "embedded").rglob("result.json"))
    outer = json.loads(embedded_outer.read_text())
    outer["official_launcher"]["schema"] = "borrowed"
    write_json(embedded_outer, outer)
    with pytest.raises(RuntimeError, match="embedded official launcher result differs"):
        module.derive(tmp_path / "embedded")


@pytest.mark.parametrize("version", [7, 999])
def test_exact_observed_kernel_version_is_required(
    tmp_path: Path, version: int
) -> None:
    module = load_module()
    valid_output(module, tmp_path)
    reconciliation_path = next(tmp_path.rglob("run-reconciliation.json"))
    reconciliation = json.loads(reconciliation_path.read_text())
    reconciliation["kernel"]["version"] = version
    write_json(reconciliation_path, reconciliation)

    with pytest.raises(RuntimeError, match="reconciliation identity differs"):
        module.derive(tmp_path)


def test_preserved_remote_source_is_verified_not_self_asserted(
    tmp_path: Path,
) -> None:
    module = load_module()
    valid_output(module, tmp_path)
    remote = next(
        tmp_path.rglob("prototype-tycho-nooa-public-3-full-timing.ipynb")
    )
    remote.write_bytes(remote.read_bytes() + b"\n")

    with pytest.raises(RuntimeError, match="source identity"):
        module.derive(tmp_path)


@pytest.mark.parametrize(
    ("target", "value", "match"),
    [
        ("outer_mode", "trigger", "outer continued-play result did not pass"),
        ("official_schema", "wrong", "official launcher result did not pass"),
        ("worker_agent", "wrong.Agent", "Solver execution envelope differs"),
    ],
)
def test_execution_identity_drift_is_rejected(
    tmp_path: Path, target: str, value: str, match: str
) -> None:
    module = load_module()
    outer_path, official_path, _ = valid_output(module, tmp_path)
    if target == "outer_mode":
        outer = json.loads(outer_path.read_text())
        outer["mode"] = value
        write_json(outer_path, outer)
    else:
        official = json.loads(official_path.read_text())
        if target == "official_schema":
            official["schema"] = value
        else:
            official["solver_sessions"]["games"]["ls20"]["worker_metadata"][
                "agent"
            ] = value
        write_json(official_path, official)
        outer = json.loads(outer_path.read_text())
        outer["official_launcher"] = official
        write_json(outer_path, outer)
    with pytest.raises(RuntimeError, match=match):
        module.derive(tmp_path)


def test_stale_gpu_capture_and_bad_lifecycle_chronology_are_rejected(
    tmp_path: Path,
) -> None:
    module = load_module()
    valid_output(module, tmp_path)
    system_path = next(tmp_path.rglob("system-before.json"))
    system = json.loads(system_path.read_text())
    system["captured_at"] = "1999-01-01T00:00:00+00:00"
    write_json(system_path, system)
    with pytest.raises(RuntimeError, match="chronology differs"):
        module.derive(tmp_path)

    valid_output(module, tmp_path / "official")
    official_path = next((tmp_path / "official").rglob("official-launcher-result.json"))
    official = json.loads(official_path.read_text())
    official["ended_at"] = official["started_at"]
    write_json(official_path, official)
    rebind_artifacts(module, tmp_path / "official")
    with pytest.raises(RuntimeError, match="chronology differs"):
        module.derive(tmp_path / "official")


def test_launcher_action_counts_are_exact_and_early_win_is_admitted(
    tmp_path: Path,
) -> None:
    module = load_module()
    _, _, stdout_path = valid_output(module, tmp_path)
    stdout_path.write_text(stdout_path.read_text().replace("ACTION1: count 1,", "ACTION1: count 999,"))
    rebind_artifacts(module, tmp_path, refresh_stdout_facts=True)
    with pytest.raises(RuntimeError, match="ACTION counts differ"):
        module.derive(tmp_path)

    early_root = tmp_path / "early-win"
    _, official_path, _ = valid_output(module, early_root)
    official = json.loads(official_path.read_text())
    game = "ls20"
    session = official["solver_sessions"]["games"][game]
    session["actions"] = session["actions"][:1]
    session["model_evidence"]["actor_calls_by_sequence"] = {"0": 6}
    session["model_evidence"]["actor_call_count"] = 6
    official["public_progress"]["games"][game]["final_state"] = "WIN"
    official["solver_sessions"]["total_solver_actions"] = 65
    official["solver_sessions"]["total_actor_calls"] = 80
    write_json(official_path, official)
    rebind_artifacts(module, early_root, refresh_request_facts=True)
    result = module.derive(early_root)
    assert result["observations"]["successor_interval_count"] == 62


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("outer_failed", "outer continued-play result did not pass"),
        ("partial_solver", "Solver action total differs"),
        ("scribe_policy", "Solver execution envelope differs"),
        ("stdout_drift", "stdout bytes differ"),
        ("wrong_gpu", "physical RTX PRO 6000 evidence is absent"),
        ("extra_provider_request", "whole-model request count differs"),
    ],
)
def test_invalid_or_drifted_evidence_is_rejected(
    tmp_path: Path, mutation: str, match: str
) -> None:
    module = load_module()
    outer_path, official_path, stdout_path = valid_output(module, tmp_path)
    if mutation == "outer_failed":
        outer = json.loads(outer_path.read_text())
        outer["status"] = "invalid"
        write_json(outer_path, outer)
    elif mutation == "partial_solver":
        official = json.loads(official_path.read_text())
        official["solver_sessions"]["total_solver_actions"] = 95
        write_json(official_path, official)
        outer = json.loads(outer_path.read_text())
        outer["official_launcher"] = official
        write_json(outer_path, outer)
    elif mutation == "scribe_policy":
        official = json.loads(official_path.read_text())
        official["solver_sessions"]["games"]["sb26"]["worker_metadata"][
            "level_boundary_policy"
        ] = "model_scribe"
        write_json(official_path, official)
        outer = json.loads(outer_path.read_text())
        outer["official_launcher"] = official
        write_json(outer_path, outer)
    elif mutation == "stdout_drift":
        stdout_path.write_text(stdout_path.read_text() + "\n")
    elif mutation == "wrong_gpu":
        system_path = next(tmp_path.rglob("system-before.json"))
        system = json.loads(system_path.read_text())
        system["gpu"]["stdout"] = "Tesla P100-PCIE-16GB\n"
        write_json(system_path, system)
    elif mutation == "extra_provider_request":
        outer = json.loads(outer_path.read_text())
        outer["request_count_facts"]["observed_launcher_solver_requests"] += 1
        outer["request_count_facts"]["observed_total_chat_completion_accesses"] += 1
        outer["request_count_facts"][
            "observed_total_successful_chat_completion_accesses"
        ] += 1
        write_json(outer_path, outer)
        rebind_artifacts(module, tmp_path)
    else:
        raise AssertionError(mutation)

    with pytest.raises(RuntimeError, match=match):
        module.derive(tmp_path)


def test_ambiguous_reset_alignment_uses_worst_feasible_intervals() -> None:
    module = load_module()
    start = datetime(2026, 9, 1, tzinfo=UTC)
    events = [
        {"action_id": 0, "timestamp": start + timedelta(seconds=1)},
        {"action_id": 1, "timestamp": start + timedelta(seconds=2)},
        {"action_id": 1, "timestamp": start + timedelta(seconds=10)},
        {"action_id": 1, "timestamp": start + timedelta(seconds=20)},
    ]

    result = module.alignment_facts(
        [1, 1], events, start, start + timedelta(seconds=21)
    )

    assert result["alignment_count"] == 3
    assert result["initial_max_seconds"] == 10
    assert result["successor_max_seconds"] == 18
    assert result["teardown_max_seconds"] == 11


def test_no_monotonic_action_alignment_is_rejected() -> None:
    module = load_module()
    start = datetime(2026, 9, 1, tzinfo=UTC)
    events = [{"action_id": 1, "timestamp": start + timedelta(seconds=1)}]
    with pytest.raises(RuntimeError, match="no monotonic official ACTION alignment"):
        module.alignment_facts([2], events, start, start + timedelta(seconds=2))


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), 0.9])
def test_policy_rejects_unsafe_or_nonfinite_safety_multiplier(
    tmp_path: Path, bad: object
) -> None:
    module = load_module()
    policy = json.loads(module.DEFAULT_POLICY.read_text())
    policy["safety_multiplier"] = bad
    path = tmp_path / "policy.json"
    write_json(path, policy)
    with pytest.raises(RuntimeError, match="safety multiplier"):
        module.load_policy(path)


def test_schedule_rejects_a_shard_that_cannot_fit_one_generation() -> None:
    module = load_module()
    projection = {
        "initial_decision_seconds": 1,
        "successor_decision_seconds": 1_000,
        "per_game_overhead_seconds": 1,
        "per_game_scheduler_reserve_seconds": 90,
        "safety_multiplier": 1.25,
    }
    with pytest.raises(RuntimeError, match="cannot fit one generation"):
        module.build_schedule(module.load_harness(), projection, 23_940)


def test_v4_failure_is_not_projection_evidence() -> None:
    module = load_module()
    v4 = (
        Path(__file__).resolve().parents[3]
        / "analysis/deployment-probes/20260901T-public3-one-action-v1/kernel-output-v4"
    )
    with pytest.raises(RuntimeError, match="canonical run reconciliation"):
        module.derive(v4)
