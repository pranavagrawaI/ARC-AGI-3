from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = CANDIDATE_ROOT / "resumable_public25.py"
TRANSPORT_PATH = CANDIDATE_ROOT / "checkpoint_dataset_transport.py"


def load_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_modules() -> tuple[ModuleType, ModuleType]:
    harness = load_path("resumable_public25", HARNESS_PATH)
    transport = load_path("checkpoint_dataset_transport", TRANSPORT_PATH)
    return harness, transport


def contract(harness: ModuleType) -> dict[str, object]:
    return {
        "schema": harness.CONTRACT_SCHEMA,
        "public_games": list(harness.PUBLIC_GAMES),
        "action_caps": harness.FIRST_LEVEL_ACTION_CAPS,
        "run_order": list(harness.RUN_ORDER),
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
        "artifact_hashes": {key: "1" * 64 for key in harness.ARTIFACT_HASH_KEYS},
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
                harness.CANONICAL_TIMING_RUN_IDENTITY_SHA256
            ),
        },
    }


def make_partial_checkpoint(harness: ModuleType, root: Path) -> tuple[str, str]:
    contract_sha256 = harness.validate_contract(contract(harness))
    checkpoint = harness.empty_checkpoint(contract_sha256)
    (root / "games").mkdir(parents=True)
    scratch = root.parent / "scratch"
    game = "s5i5"
    cap = harness.FIRST_LEVEL_ACTION_CAPS[game]
    session = scratch / "solver-workspaces" / f"{game}-session"
    session.mkdir(parents=True)
    (session / "llm-trace.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "reasoning_tokens": 3,
                }
            )
            + "\n"
            for _ in range(cap)
        )
    )
    (scratch / "recordings").mkdir()
    (scratch / "recordings" / f"{game}.recording.jsonl").write_text("{}\n")
    (scratch / "public25-launcher.stdout.log").write_text("passed\n")
    (scratch / "public25-launcher.stderr.log").write_text("")
    (scratch / "orchestrator-command.json").write_text("{}\n")
    (scratch / "ARC-AGI-3-Agents").mkdir()
    (scratch / "ARC-AGI-3-Agents" / "logs.log").write_text("scorecard\n")
    result = {
        "status": "passed",
        "public_games": [game],
        "public_progress": {
            "game_count": 1,
            "games": {
                game: {
                    "final_state": "NOT_FINISHED",
                    "final_levels_completed": 1,
                    "max_levels_completed": 1,
                    "recorded_action_results": cap,
                }
            },
        },
        "solver_sessions": {
            "game_count": 1,
            "total_solver_actions": cap,
            "total_actor_calls": cap,
            "games": {game: {"session": str(session)}},
        },
    }
    (scratch / "game-result.json").write_text(json.dumps(result) + "\n")
    harness.publish_completed_game(
        scratch,
        root,
        checkpoint,
        game=game,
        result=result,
    )
    return contract_sha256, harness.sha256_file(root / "checkpoint.json")


def export_fixture(tmp_path: Path) -> tuple[ModuleType, ModuleType, Path, dict[str, object]]:
    harness, transport = load_modules()
    checkpoint_root = tmp_path / "checkpoint-source"
    contract_sha256, checkpoint_sha256 = make_partial_checkpoint(
        harness, checkpoint_root
    )
    staging = tmp_path / "transport-staging"
    manifest = transport.export_checkpoint_dataset(
        checkpoint_root,
        staging,
        expected_contract_sha256=contract_sha256,
        expected_checkpoint_manifest_sha256=checkpoint_sha256,
        owner="pranavagrawai",
        slug_prefix="tycho-nooa-p25-ckpt",
    )
    return harness, transport, staging, manifest


def downloaded_copy(staging: Path, destination: Path) -> None:
    shutil.copytree(
        staging,
        destination,
        ignore=shutil.ignore_patterns("dataset-metadata.json"),
    )


def verify_kwargs(manifest: dict[str, object]) -> dict[str, str]:
    return {
        "expected_transport_manifest_sha256": str(
            manifest["transport_manifest_sha256"]
        ),
        "expected_contract_sha256": str(manifest["contract_sha256"]),
        "expected_predecessor_manifest_sha256": str(
            manifest["checkpoint_manifest_sha256"]
        ),
        "expected_dataset_id": str(manifest["dataset_id"]),
    }


def test_export_readback_and_atomic_successor_import(tmp_path: Path) -> None:
    harness, transport, staging, manifest = export_fixture(tmp_path)

    assert (staging / "transport-manifest.json").is_file()
    assert (staging / "checkpoint.tar").is_file()
    assert not (staging / "checkpoint").exists()
    metadata = json.loads((staging / "dataset-metadata.json").read_text())
    assert metadata["id"] == manifest["dataset_id"]
    assert manifest["intended_visibility"] == "private"
    assert str(manifest["publication_identity"])[:20] in metadata["id"]
    assert manifest["completed_games"] == ["s5i5"]

    second_staging = tmp_path / "transport-staging-second"
    second = transport.export_checkpoint_dataset(
        tmp_path / "checkpoint-source",
        second_staging,
        expected_contract_sha256=str(manifest["contract_sha256"]),
        expected_checkpoint_manifest_sha256=str(
            manifest["checkpoint_manifest_sha256"]
        ),
        owner="pranavagrawai",
        slug_prefix="tycho-nooa-p25-ckpt",
    )
    assert second == manifest
    assert transport.regular_file_manifest(second_staging) == (
        transport.regular_file_manifest(staging)
    )

    downloaded = tmp_path / "downloaded"
    downloaded_copy(staging, downloaded)
    verified = transport.verify_downloaded_readback(downloaded, **verify_kwargs(manifest))
    assert verified == manifest

    mounted = tmp_path / "kaggle-input" / str(manifest["dataset_id"]).split("/")[1]
    mounted.parent.mkdir()
    shutil.copytree(downloaded, mounted)
    destination = tmp_path / "imported-checkpoint"
    imported = transport.import_mounted_transport(
        mounted.parent,
        destination,
        **verify_kwargs(manifest),
    )
    assert imported["checkpoint_root"] == str(destination)
    harness.verify_checkpoint(destination, str(manifest["contract_sha256"]))
    assert not list(destination.parent.glob(f".{destination.name}.import-*"))


def test_partial_publication_and_duplicate_mount_are_rejected(tmp_path: Path) -> None:
    _, transport, staging, manifest = export_fixture(tmp_path)
    partial = tmp_path / "partial-mount" / "partial"
    partial.parent.mkdir()
    shutil.copytree(staging, partial)
    (partial / "transport-manifest.json").unlink()
    with pytest.raises(RuntimeError, match="exactly one mounted checkpoint transport"):
        transport.import_mounted_transport(
            partial.parent,
            tmp_path / "never-imported",
            **verify_kwargs(manifest),
        )

    missing_archive = tmp_path / "missing-archive"
    downloaded_copy(staging, missing_archive)
    (missing_archive / "checkpoint.tar").unlink()
    with pytest.raises(RuntimeError, match="archive inventory or bytes differ"):
        transport.verify_downloaded_readback(
            missing_archive, **verify_kwargs(manifest)
        )

    mounts = tmp_path / "mounts"
    shutil.copytree(staging, mounts / "one")
    shutil.copytree(staging, mounts / "two")
    with pytest.raises(RuntimeError, match="exactly one mounted checkpoint transport"):
        transport.import_mounted_transport(
            mounts,
            tmp_path / "duplicate-import",
            **verify_kwargs(manifest),
        )


def test_tamper_symlink_and_unexpected_file_are_rejected(tmp_path: Path) -> None:
    _, transport, staging, manifest = export_fixture(tmp_path)

    tampered = tmp_path / "tampered"
    downloaded_copy(staging, tampered)
    payload = tampered / "checkpoint.tar"
    payload.write_bytes(payload.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="inventory or bytes differ"):
        transport.verify_downloaded_readback(tampered, **verify_kwargs(manifest))

    linked = tmp_path / "linked"
    downloaded_copy(staging, linked)
    victim = linked / "checkpoint.tar"
    replacement = victim.with_suffix(victim.suffix + ".real")
    victim.rename(replacement)
    victim.symlink_to(replacement.name)
    with pytest.raises(RuntimeError, match="symlink"):
        transport.verify_downloaded_readback(linked, **verify_kwargs(manifest))

    extra = tmp_path / "extra"
    downloaded_copy(staging, extra)
    (extra / "not-declared.txt").write_text("drift\n")
    with pytest.raises(RuntimeError, match="inventory or bytes differ"):
        transport.verify_downloaded_readback(extra, **verify_kwargs(manifest))


def test_contract_predecessor_and_identity_drift_are_rejected(tmp_path: Path) -> None:
    _, transport, staging, manifest = export_fixture(tmp_path)
    downloaded = tmp_path / "downloaded"
    downloaded_copy(staging, downloaded)

    wrong = verify_kwargs(manifest)
    wrong["expected_contract_sha256"] = "a" * 64
    with pytest.raises(RuntimeError, match="contract differs"):
        transport.verify_downloaded_readback(downloaded, **wrong)

    wrong = verify_kwargs(manifest)
    wrong["expected_predecessor_manifest_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="predecessor checkpoint differs"):
        transport.verify_downloaded_readback(downloaded, **wrong)

    wrong = verify_kwargs(manifest)
    wrong["expected_dataset_id"] = "pranavagrawai/wrong-identity"
    with pytest.raises(RuntimeError, match="dataset identity differs"):
        transport.verify_downloaded_readback(downloaded, **wrong)

    wrong = verify_kwargs(manifest)
    wrong["expected_transport_manifest_sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="transport manifest hash differs"):
        transport.verify_downloaded_readback(downloaded, **wrong)


def test_failed_import_preserves_destination_and_leaves_no_partial(tmp_path: Path) -> None:
    _, transport, staging, manifest = export_fixture(tmp_path)
    mounts = tmp_path / "mounts"
    downloaded_copy(staging, mounts / "transport")
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "sentinel.txt").write_text("preserve me\n")

    with pytest.raises(RuntimeError, match="destination already exists"):
        transport.import_mounted_transport(
            mounts,
            destination,
            **verify_kwargs(manifest),
        )
    assert (destination / "sentinel.txt").read_text() == "preserve me\n"
    assert not list(destination.parent.glob(f".{destination.name}.import-*"))


def test_export_requires_exact_frozen_contract_and_checkpoint(tmp_path: Path) -> None:
    harness, transport = load_modules()
    checkpoint_root = tmp_path / "checkpoint-source"
    contract_sha256, checkpoint_sha256 = make_partial_checkpoint(
        harness, checkpoint_root
    )
    with pytest.raises(RuntimeError, match="checkpoint contract differs"):
        transport.export_checkpoint_dataset(
            checkpoint_root,
            tmp_path / "wrong-contract",
            expected_contract_sha256="d" * 64,
            expected_checkpoint_manifest_sha256=checkpoint_sha256,
            owner="pranavagrawai",
        )
    with pytest.raises(RuntimeError, match="checkpoint manifest hash differs"):
        transport.export_checkpoint_dataset(
            checkpoint_root,
            tmp_path / "wrong-checkpoint",
            expected_contract_sha256=contract_sha256,
            expected_checkpoint_manifest_sha256="e" * 64,
            owner="pranavagrawai",
        )


def test_failure_byte_manifest_is_carried_and_cli_imports_it(tmp_path: Path) -> None:
    harness, transport = load_modules()
    checkpoint_root = tmp_path / "checkpoint-source"
    contract_sha256, _ = make_partial_checkpoint(harness, checkpoint_root)
    checkpoint = json.loads((checkpoint_root / "checkpoint.json").read_text())
    failure_root = checkpoint_root / "failures" / "sb26"
    failure_root.mkdir(parents=True)
    (failure_root / "child.stderr.log").write_text("synthetic child failure\n")
    (failure_root / "orchestrator-command.json").write_text(
        json.dumps({"returncode": 70}) + "\n"
    )
    checkpoint["failures"].append(
        {
            "game": "sb26",
            "returncode": 70,
            "result_status": None,
            "files": harness.regular_file_manifest(failure_root),
        }
    )
    harness.write_json_atomic(checkpoint_root / "checkpoint.json", checkpoint)
    checkpoint_sha256 = harness.sha256_file(checkpoint_root / "checkpoint.json")
    staging = tmp_path / "staging"
    manifest = transport.export_checkpoint_dataset(
        checkpoint_root,
        staging,
        expected_contract_sha256=contract_sha256,
        expected_checkpoint_manifest_sha256=checkpoint_sha256,
        owner="pranavagrawai",
    )
    assert any(
        record["path"].startswith("checkpoint/failures/sb26/")
        for record in manifest["payload_files"]
    )

    downloaded = tmp_path / "downloaded"
    downloaded_copy(staging, downloaded)
    mounts = tmp_path / "mounts"
    shutil.copytree(downloaded, mounts / "checkpoint-dataset")
    destination = tmp_path / "cli-import"
    command = [
        sys.executable,
        str(TRANSPORT_PATH),
        "import",
        "--input-root",
        str(mounts),
        "--destination",
        str(destination),
        "--expected-transport-manifest-sha256",
        str(manifest["transport_manifest_sha256"]),
        "--expected-contract-sha256",
        contract_sha256,
        "--expected-predecessor-manifest-sha256",
        checkpoint_sha256,
        "--expected-dataset-id",
        str(manifest["dataset_id"]),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["checkpoint_root"] == str(destination)
    verified = harness.verify_checkpoint(destination, contract_sha256)
    assert verified["failures"][0]["game"] == "sb26"
