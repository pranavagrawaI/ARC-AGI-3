from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
MODULE_PATH = HERE / "resumable_public25.py"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("resumable_public25", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contract(module: ModuleType) -> dict[str, object]:
    return {
        "schema": module.CONTRACT_SCHEMA,
        "public_games": list(module.PUBLIC_GAMES),
        "action_caps": module.FIRST_LEVEL_ACTION_CAPS,
        "run_order": list(module.RUN_ORDER),
        "evidence_kind": "public_progress_not_official_score",
        "admission_policy": "C1_sequential_game_shards",
        "solver_envelope": {
            "context_window": 32_768,
            "initial_actor_calls": 6,
            "initial_max_tokens": 16_000,
            "successor_actor_calls": 1,
            "successor_max_tokens": 16_000,
            "model_retries": 0,
            "mode": "single",
            "treatment": "single_decision_discipline",
            "level_boundary_policy": "deterministic_level_checkpoint",
        },
        "artifact_hashes": {key: "1" * 64 for key in module.ARTIFACT_HASH_KEYS},
        "runtime_projection": {
            "initial_decision_seconds": 100,
            "successor_decision_seconds": 10,
            "per_game_overhead_seconds": 30,
            "per_game_scheduler_reserve_seconds": 90,
            "safety_multiplier": 1.25,
            "next_scratch_bound_bytes": 8_000_000,
            "storage_publication_margin_bytes": 1_000_000,
            "source_result_sha256": "2" * 64,
            "estimator_schema": "tycho-nooa-public25-runtime-projection-evidence-v1",
            "projection_policy_sha256": "3" * 64,
            "source_artifact_manifest_sha256": "4" * 64,
            "derivation_artifact_sha256": "5" * 64,
            "canonical_run_identity_sha256": (
                module.CANONICAL_TIMING_RUN_IDENTITY_SHA256
            ),
        },
    }


def make_game_scratch(
    module: ModuleType, root: Path, game: str, *, actions: int, levels: int = 1
) -> dict[str, object]:
    session = root / "solver-workspaces" / f"{game}-session"
    session.mkdir(parents=True)
    trace = []
    for sequence in range(actions):
        trace.append({
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "reasoning_tokens": 3,
        })
    (session / "llm-trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trace)
    )
    (root / "recordings").mkdir()
    (root / "recordings" / f"{game}.recording.jsonl").write_text("{}\n")
    (root / "public25-launcher.stdout.log").write_text("passed\n")
    (root / "public25-launcher.stderr.log").write_text("")
    (root / "orchestrator-command.json").write_text("{}\n")
    official = root / "ARC-AGI-3-Agents"
    official.mkdir()
    (official / "logs.log").write_text("scorecard evidence\n")
    result = {
        "status": "passed",
        "public_games": [game],
        "public_progress": {
            "game_count": 1,
            "games": {
                game: {
                    "final_state": "NOT_FINISHED",
                    "final_levels_completed": levels,
                    "max_levels_completed": levels,
                    "recorded_action_results": actions,
                }
            },
        },
        "solver_sessions": {
            "game_count": 1,
            "total_solver_actions": actions,
            "total_actor_calls": actions,
            "games": {game: {"session": str(session)}},
        },
    }
    (root / "game-result.json").write_text(json.dumps(result) + "\n")
    return result


def test_exact_roster_and_first_level_cohort_caps() -> None:
    module = load_module()
    assert len(module.PUBLIC_GAMES) == 25
    assert set(module.FIRST_LEVEL_ACTION_CAPS) == set(module.PUBLIC_GAMES)
    assert sum(module.FIRST_LEVEL_ACTION_CAPS.values()) == 1_056
    assert [module.FIRST_LEVEL_ACTION_CAPS[game] for game in module.RUN_ORDER] == sorted(
        module.FIRST_LEVEL_ACTION_CAPS.values()
    )
    assert {cap: list(module.FIRST_LEVEL_ACTION_CAPS.values()).count(cap) for cap in (24, 40, 64, 80)} == {
        24: 9,
        40: 9,
        64: 5,
        80: 2,
    }
    assert module.validate_contract(contract(module)) == module.sha256_bytes(
        module.canonical_json(contract(module))
    )


def test_contract_refuses_unmeasured_projection_and_cap_drift() -> None:
    module = load_module()
    value = contract(module)
    value["runtime_projection"]["source_result_sha256"] = "unmeasured"  # type: ignore[index]
    with pytest.raises(RuntimeError, match="not bound"):
        module.validate_contract(value)

    value = contract(module)
    value["action_caps"] = {**module.FIRST_LEVEL_ACTION_CAPS, "ls20": 80}
    with pytest.raises(RuntimeError, match="action caps differ"):
        module.validate_contract(value)

    value = contract(module)
    envelope = value["solver_envelope"]
    assert isinstance(envelope, dict)
    envelope["level_boundary_policy"] = "model_scribe"
    with pytest.raises(RuntimeError, match="Solver envelope differs"):
        module.validate_contract(value)

    value = contract(module)
    envelope = value["solver_envelope"]
    assert isinstance(envelope, dict)
    del envelope["level_boundary_policy"]
    with pytest.raises(RuntimeError, match="Solver envelope differs"):
        module.validate_contract(value)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -1])
def test_contract_refuses_nonfinite_or_boolean_projection_values(bad: object) -> None:
    module = load_module()
    value = contract(module)
    value["runtime_projection"]["successor_decision_seconds"] = bad  # type: ignore[index]
    with pytest.raises(RuntimeError, match="positive measured bounds"):
        module.validate_contract(value)


def test_contract_refuses_safety_multiplier_below_one() -> None:
    module = load_module()
    value = contract(module)
    value["runtime_projection"]["safety_multiplier"] = 0.9  # type: ignore[index]
    with pytest.raises(RuntimeError, match="below one"):
        module.validate_contract(value)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("next_scratch_bound_bytes", 0),
        ("next_scratch_bound_bytes", True),
        ("next_scratch_bound_bytes", 1.5),
        ("storage_publication_margin_bytes", -1),
        ("storage_publication_margin_bytes", False),
        ("storage_publication_margin_bytes", 1.5),
    ],
)
def test_contract_requires_positive_integer_storage_projection(
    field: str, bad: object
) -> None:
    module = load_module()
    value = contract(module)
    value["runtime_projection"][field] = bad  # type: ignore[index]
    with pytest.raises(RuntimeError, match="storage byte bounds"):
        module.validate_contract(value)

    value = contract(module)
    del value["runtime_projection"][field]  # type: ignore[index]
    with pytest.raises(RuntimeError, match="provenance fields differ"):
        module.validate_contract(value)


def test_storage_admission_reserves_two_scratches_plus_publication_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    value = contract(module)
    projection = value["runtime_projection"]
    assert isinstance(projection, dict)
    projection["next_scratch_bound_bytes"] = 100
    projection["storage_publication_margin_bytes"] = 17

    def statvfs_with_available(available: int) -> SimpleNamespace:
        return SimpleNamespace(
            f_bavail=available,
            f_bfree=available,
            f_blocks=1_000,
            f_frsize=1,
        )

    monkeypatch.setattr(module.os, "statvfs", lambda _path: statvfs_with_available(216))
    with pytest.raises(RuntimeError, match="requires 217 available bytes; found 216"):
        module.storage_admission(value, tmp_path)

    monkeypatch.setattr(module.os, "statvfs", lambda _path: statvfs_with_available(217))
    admission = module.storage_admission(value, tmp_path)
    assert admission == {
        "next_scratch_bound_bytes": 100,
        "storage_publication_margin_bytes": 17,
        "required_available_bytes": 217,
        "available_bytes": 217,
        "passed": True,
    }


def test_main_refuses_storage_before_creating_game_scratch(tmp_path: Path) -> None:
    module = load_module()
    value = contract(module)
    projection = value["runtime_projection"]
    assert isinstance(projection, dict)
    projection["next_scratch_bound_bytes"] = 10**18
    projection["storage_publication_margin_bytes"] = 1
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(value) + "\n")
    probe = tmp_path / "must-not-run.py"
    probe.write_text("raise AssertionError('storage admission started the game')\n")

    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--base-url", "http://127.0.0.1:30000/v1",
            "--model", "fake",
            "--workspace-root", str(tmp_path / "work"),
            "--output", str(tmp_path / "result.json"),
            "--timeout-s", "2000",
            "--contract", str(contract_path),
            "--template-root", str(tmp_path / "unused-template"),
            "--game-probe", str(probe),
            "--checkpoint-root", str(tmp_path / "checkpoint"),
            "--stop-reserve-s", "0",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode != 0
    assert "storage admission requires 2000000000000000001" in completed.stderr
    assert not (tmp_path / "work" / "scratch").exists()
    ledger_rows = [
        json.loads(line)
        for line in (tmp_path / "work" / "storage-ledger.jsonl").read_text().splitlines()
    ]
    assert [row["event"] for row in ledger_rows] == [
        "after_predecessor_transport_import",
        "after_predecessor_copy",
        "pre_game_admission",
    ]
    assert ledger_rows[-1]["admission"]["passed"] is False


def test_contract_requires_exact_route_and_projection_provenance() -> None:
    module = load_module()
    value = contract(module)
    del value["artifact_hashes"]["outer_runtime"]  # type: ignore[index]
    with pytest.raises(RuntimeError, match="artifact inventory differs"):
        module.validate_contract(value)

    value = contract(module)
    value["runtime_projection"]["canonical_run_identity_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(RuntimeError, match="canonical run identity differs"):
        module.validate_contract(value)

    value = contract(module)
    value["runtime_projection"]["derivation_artifact_sha256"] = "unbound"  # type: ignore[index]
    with pytest.raises(RuntimeError, match="derivation_artifact_sha256 is malformed"):
        module.validate_contract(value)


def test_v1_contract_checkpoint_is_not_a_v2_predecessor(tmp_path: Path) -> None:
    module = load_module()
    v2 = contract(module)
    v2_hash = module.validate_contract(v2)
    v1 = contract(module)
    v1["schema"] = "tycho-nooa-public25-cohort-contract-v1"
    envelope = v1["solver_envelope"]
    assert isinstance(envelope, dict)
    del envelope["level_boundary_policy"]
    v1_hash = module.sha256_bytes(module.canonical_json(v1))
    assert v1_hash != v2_hash

    checkpoint_root = tmp_path / "v1-checkpoint"
    (checkpoint_root / "games").mkdir(parents=True)
    module.write_json_atomic(
        checkpoint_root / "checkpoint.json",
        module.empty_checkpoint(v1_hash),
    )
    with pytest.raises(RuntimeError, match="checkpoint contract differs"):
        module.verify_checkpoint(checkpoint_root, v2_hash)

def test_completed_game_is_atomic_manifested_and_aggregated(tmp_path: Path) -> None:
    module = load_module()
    contract_hash = module.validate_contract(contract(module))
    checkpoint_root = tmp_path / "checkpoint"
    (checkpoint_root / "games").mkdir(parents=True)
    checkpoint = module.empty_checkpoint(contract_hash)
    scratch = tmp_path / "scratch"
    game = "ls20"
    actions = module.FIRST_LEVEL_ACTION_CAPS[game]
    result = make_game_scratch(module, scratch, game, actions=actions, levels=1)

    module.publish_completed_game(
        scratch, checkpoint_root, checkpoint, game=game, result=result
    )

    assert checkpoint["completed_games"] == [game]
    verified = module.verify_checkpoint(checkpoint_root, contract_hash)
    metrics = module.aggregate(verified)
    assert metrics["completed_game_count"] == 1
    assert metrics["total_solver_actions"] == actions
    assert metrics["total_actor_calls"] == actions
    assert metrics["total_prompt_tokens"] == 10 * actions
    assert metrics["total_max_public_levels_completed"] == 1
    assert metrics["public_levels_per_solver_action"] == 1 / actions
    assert game not in metrics["remaining_games"]


def test_predecessor_import_is_hash_bound_and_tamper_evident(tmp_path: Path) -> None:
    module = load_module()
    contract_hash = module.validate_contract(contract(module))
    predecessor = tmp_path / "predecessor"
    (predecessor / "games").mkdir(parents=True)
    checkpoint = module.empty_checkpoint(contract_hash)
    scratch = tmp_path / "scratch"
    game = "s5i5"
    result = make_game_scratch(
        module,
        scratch,
        game,
        actions=module.FIRST_LEVEL_ACTION_CAPS[game],
    )
    module.publish_completed_game(
        scratch, predecessor, checkpoint, game=game, result=result
    )
    manifest_hash = module.sha256_file(predecessor / "checkpoint.json")

    imported = module.import_predecessor(
        predecessor,
        tmp_path / "current",
        contract_sha256=contract_hash,
        expected_manifest_sha256=manifest_hash,
    )
    assert imported["generation"] == 1
    assert imported["predecessor_manifest_sha256"] == manifest_hash
    assert imported["completed_games"] == [game]

    bundle_file = next(
        path
        for path in (predecessor / "games" / game).rglob("*")
        if path.is_file() and path.name != "bundle.json"
    )
    bundle_file.write_bytes(bundle_file.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="inventory or bytes differ"):
        module.verify_checkpoint(predecessor, contract_hash)


def test_uncommitted_orphan_game_is_ignored_on_restart(tmp_path: Path) -> None:
    module = load_module()
    contract_hash = module.validate_contract(contract(module))
    predecessor = tmp_path / "predecessor"
    (predecessor / "games").mkdir(parents=True)
    checkpoint = module.empty_checkpoint(contract_hash)
    module.write_json_atomic(predecessor / "checkpoint.json", checkpoint)
    manifest_hash = module.sha256_file(predecessor / "checkpoint.json")

    # This is the only crash window in publication: the complete directory was
    # renamed, but checkpoint.json still names the preceding generation.
    orphan = predecessor / "games" / "ls20"
    orphan.mkdir()
    (orphan / "unbound.txt").write_text("must not be trusted\n")

    verified = module.verify_checkpoint(predecessor, contract_hash)
    assert verified["completed_games"] == []
    imported = module.import_predecessor(
        predecessor,
        tmp_path / "current",
        contract_sha256=contract_hash,
        expected_manifest_sha256=manifest_hash,
    )
    assert imported["completed_games"] == []
    assert list((tmp_path / "current" / "games").iterdir()) == []


def test_nonwinning_game_must_reach_its_exact_cap(tmp_path: Path) -> None:
    module = load_module()
    game = "sb26"
    result = make_game_scratch(module, tmp_path, game, actions=1, levels=0)
    with pytest.raises(RuntimeError, match="stopped before its cap"):
        module.result_metrics(result, game, module.FIRST_LEVEL_ACTION_CAPS[game])


def test_projection_reserves_measured_whole_game_time() -> None:
    module = load_module()
    value = contract(module)
    # The child receives the measured+safety budget. Publication and process
    # containment keep their separate fixed reserve.
    assert module.projected_child_seconds(value, "ls20") == 450
    assert module.projected_game_seconds(value, "ls20") == 540


def test_fake_all25_run_publishes_exact_complete_checkpoint(tmp_path: Path) -> None:
    module = load_module()
    value = contract(module)
    value["runtime_projection"] = {
        "initial_decision_seconds": 0.01,
        "successor_decision_seconds": 0.01,
        "per_game_overhead_seconds": 0.01,
        "per_game_scheduler_reserve_seconds": 1,
        "safety_multiplier": 1.01,
        "next_scratch_bound_bytes": 8_000_000,
        "storage_publication_margin_bytes": 1_000_000,
        "source_result_sha256": "3" * 64,
        "estimator_schema": "tycho-nooa-public25-runtime-projection-evidence-v1",
        "projection_policy_sha256": "4" * 64,
        "source_artifact_manifest_sha256": "5" * 64,
        "derivation_artifact_sha256": "6" * 64,
        "canonical_run_identity_sha256": (
            module.CANONICAL_TIMING_RUN_IDENTITY_SHA256
        ),
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(value) + "\n")
    template = tmp_path / "template"
    official = template / "ARC-AGI-3-Agents"
    official.mkdir(parents=True)
    (official / ".env").write_text("RECORDINGS_DIR=/old/path\n")
    (template / "run_official_agent.py").write_text("# fixture\n")
    fake_probe = tmp_path / "fake_probe.py"
    fake_probe.write_text(
        """import argparse, json, os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--base-url'); p.add_argument('--model')
p.add_argument('--workspace-root', type=Path); p.add_argument('--output', type=Path)
p.add_argument('--timeout-s')
a=p.parse_args(); game=os.environ['TYCHO_PUBLIC_GAME_IDS']; cap=int(os.environ['TYCHO_KAGGLE_MAX_ACTIONS'])
session=a.workspace_root/f'{game}-session'; session.mkdir(parents=True)
(session/'llm-trace.jsonl').write_text(''.join(json.dumps({'prompt_tokens':10,'completion_tokens':4,'reasoning_tokens':3})+'\\n' for _ in range(cap)))
scratch=a.output.parent; (scratch/'recordings').mkdir(exist_ok=True)
(scratch/'recordings'/f'{game}.recording.jsonl').write_text('{}\\n')
(scratch/'public25-launcher.stdout.log').write_text('passed\\n')
(scratch/'public25-launcher.stderr.log').write_text('')
(scratch/'ARC-AGI-3-Agents'/'logs.log').write_text('scorecard\\n')
result={'status':'passed','public_games':[game],
'public_progress':{'game_count':1,'games':{game:{'final_state':'NOT_FINISHED','final_levels_completed':1,'max_levels_completed':1,'recorded_action_results':cap}}},
'solver_sessions':{'game_count':1,'total_solver_actions':cap,'total_actor_calls':cap,'games':{game:{'session':str(session)}}}}
a.output.write_text(json.dumps(result)+'\\n')
"""
    )
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--base-url", "http://127.0.0.1:30000/v1",
            "--model", "fake",
            "--workspace-root", str(tmp_path / "work"),
            "--output", str(output),
            "--timeout-s", "2000",
            "--contract", str(contract_path),
            "--template-root", str(template),
            "--game-probe", str(fake_probe),
            "--checkpoint-root", str(checkpoint),
            "--stop-reserve-s", "0",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    first_command = json.loads(
        (
            checkpoint
            / "games"
            / module.RUN_ORDER[0]
            / "orchestrator-command.json"
        ).read_text()
    )
    assert first_command["timeout_seconds"] == 61
    result = json.loads(output.read_text())
    assert result["status"] == "passed"
    assert result["qualification_status"] == "complete"
    assert result["all25_complete"] is True
    assert result["games_attempted_this_run"] == list(module.RUN_ORDER)
    assert result["games_completed_this_run"] == list(module.RUN_ORDER)
    assert result["metrics"]["completed_game_count"] == 25
    assert result["metrics"]["total_solver_actions"] == 1_056
    assert result["metrics"]["total_max_public_levels_completed"] == 25
    assert result["metrics"]["official_score"] is None
    scratch_root = tmp_path / "work" / "scratch"
    assert list(scratch_root.iterdir()) == []
    ledger_rows = [
        json.loads(line)
        for line in (tmp_path / "work" / "storage-ledger.jsonl").read_text().splitlines()
    ]
    ledger_path = tmp_path / "work" / "storage-ledger.jsonl"
    assert result["storage_safety"] == {
        "ledger_schema": module.STORAGE_LEDGER_SCHEMA,
        "ledger_path": str(ledger_path),
        "ledger_sha256": module.sha256_file(ledger_path),
        "ledger_record_count": len(ledger_rows),
        "next_scratch_bound_bytes": 8_000_000,
        "storage_publication_margin_bytes": 1_000_000,
    }
    assert [row["sequence"] for row in ledger_rows] == list(range(len(ledger_rows)))
    assert [row["event"] for row in ledger_rows[:2]] == [
        "after_predecessor_transport_import",
        "after_predecessor_copy",
    ]
    assert sum(row["event"] == "pre_game_admission" for row in ledger_rows) == 25
    assert sum(row["event"] == "post_game_prepublication" for row in ledger_rows) == 25
    assert sum(row["event"] == "post_publication_pre_cleanup" for row in ledger_rows) == 25
    assert sum(row["event"] == "post_publication_cleanup" for row in ledger_rows) == 25
    admission_rows = [row for row in ledger_rows if row["event"] == "pre_game_admission"]
    assert all(row["admission"]["passed"] is True for row in admission_rows)
    assert all(
        row["admission"]["required_available_bytes"] == 17_000_000
        for row in admission_rows
    )
    assert all(
        row["components"]["scratch"]["exists"] is False
        for row in ledger_rows
        if row["event"] == "post_publication_cleanup"
    )
    verified = module.verify_checkpoint(
        checkpoint, module.validate_contract(value)
    )
    assert verified["completed_games"] == list(module.PUBLIC_GAMES)


def test_failed_game_keeps_scratch_and_manifests_complete_failure_copy(
    tmp_path: Path,
) -> None:
    module = load_module()
    value = contract(module)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(value) + "\n")
    template = tmp_path / "template"
    official = template / "ARC-AGI-3-Agents"
    official.mkdir(parents=True)
    (official / ".env").write_text("RECORDINGS_DIR=/old/path\n")
    (template / "run_official_agent.py").write_text("# fixture\n")
    failing_probe = tmp_path / "failing_probe.py"
    failing_probe.write_text(
        """import argparse
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--base-url'); p.add_argument('--model')
p.add_argument('--workspace-root', type=Path); p.add_argument('--output', type=Path)
p.add_argument('--timeout-s'); a=p.parse_args()
a.workspace_root.mkdir(parents=True)
(a.workspace_root/'failure-marker.bin').write_bytes(b'preserve-me')
raise SystemExit(9)
"""
    )
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--base-url", "http://127.0.0.1:30000/v1",
            "--model", "fake",
            "--workspace-root", str(tmp_path / "work"),
            "--output", str(output),
            "--timeout-s", "2000",
            "--contract", str(contract_path),
            "--template-root", str(template),
            "--game-probe", str(failing_probe),
            "--checkpoint-root", str(checkpoint),
            "--stop-reserve-s", "0",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    game = module.RUN_ORDER[0]
    scratch = tmp_path / "work" / "scratch" / game
    failure = checkpoint / "failures" / game
    assert scratch.is_dir()
    assert failure.is_dir()
    assert module.regular_file_manifest(failure) == module.regular_file_manifest(scratch)
    result = json.loads(output.read_text())
    assert result["qualification_status"] == "partial_failure"
    assert result["failure"] == {
        "game": game,
        "returncode": 9,
        "result_status": None,
    }
    ledger_rows = [
        json.loads(line)
        for line in (tmp_path / "work" / "storage-ledger.jsonl").read_text().splitlines()
    ]
    assert result["storage_safety"]["ledger_record_count"] == len(ledger_rows)
    assert result["storage_safety"]["ledger_sha256"] == module.sha256_file(
        tmp_path / "work" / "storage-ledger.jsonl"
    )
    assert [row["event"] for row in ledger_rows[-4:]] == [
        "pre_game_admission",
        "post_game_prepublication",
        "pre_failure_copy",
        "post_failure_copy",
    ]
    assert ledger_rows[-1]["components"]["scratch"]["exists"] is True
    assert ledger_rows[-1]["components"]["failure_evidence"]["exists"] is True


def test_run_game_timeout_contains_the_probe_process_group(tmp_path: Path) -> None:
    module = load_module()
    template = tmp_path / "template"
    official = template / "ARC-AGI-3-Agents"
    official.mkdir(parents=True)
    (official / ".env").write_text("RECORDINGS_DIR=/old/path\n")
    (template / "run_official_agent.py").write_text("# fixture\n")
    fake_probe = tmp_path / "hanging_probe.py"
    fake_probe.write_text(
        """import argparse, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--base-url'); p.add_argument('--model')
p.add_argument('--workspace-root'); p.add_argument('--output', type=Path)
p.add_argument('--timeout-s'); a=p.parse_args()
pid_path=a.output.parent/'grandchild.pid'
code=("import os,time; from pathlib import Path; "
      f"Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)")
subprocess.Popen([sys.executable, '-c', code])
print('probe-started-grandchild', flush=True)
time.sleep(60)
"""
    )
    scratch = tmp_path / "scratch"
    args = SimpleNamespace(
        template_root=template,
        base_url="http://127.0.0.1:30000/v1",
        model="no-model-request",
        game_probe=fake_probe,
    )

    completed, result = module.run_game(
        game="bp35",
        timeout_s=0.1,
        args=args,
        scratch=scratch,
        communication_grace_s=0,
        termination_grace_s=0.5,
    )

    assert completed.returncode == 124
    assert result is None
    command = json.loads((scratch / "orchestrator-command.json").read_text())
    assert command["timed_out"] is True
    assert command["process_group_closed"] is True
    assert command["term_sent"] is True
    assert "probe-started-grandchild" in command["stdout"]
    grandchild_pid = int((scratch / "grandchild.pid").read_text())
    deadline = time.monotonic() + 2
    while Path(f"/proc/{grandchild_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{grandchild_pid}").exists()
