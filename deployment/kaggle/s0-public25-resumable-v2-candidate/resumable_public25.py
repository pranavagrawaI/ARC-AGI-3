#!/usr/bin/env python3
"""Run the public first-level cohort as atomic, resumable game shards.

This is a preparation harness, not a frozen notebook.  It reuses the qualified
single-run public launcher probe for one game at a time.  A completed game is
published atomically with a complete byte manifest; an interrupted game is
rerun from its public initial state because ARC environment state is not
serializable through the official interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PUBLIC_GAMES = (
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
    "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
    "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33",
    "wa30",
)

# These caps make the public baseline's first level reachable for each game.
# They are an opportunity diagnostic, not an official score or a claim that the
# Solver will match the baseline trajectory.
FIRST_LEVEL_ACTION_CAPS = {
    "ar25": 40,
    "bp35": 24,
    "cd82": 64,
    "cn04": 40,
    "dc22": 64,
    "ft09": 64,
    "g50t": 80,
    "ka59": 40,
    "lf52": 40,
    "lp85": 24,
    "ls20": 24,
    "m0r0": 40,
    "r11l": 24,
    "re86": 40,
    "s5i5": 24,
    "sb26": 24,
    "sc25": 40,
    "sk48": 64,
    "sp80": 40,
    "su15": 24,
    "tn36": 40,
    "tr87": 64,
    "tu93": 24,
    "vc33": 24,
    "wa30": 80,
}
RUN_ORDER = tuple(
    sorted(PUBLIC_GAMES, key=lambda game: (FIRST_LEVEL_ACTION_CAPS[game], PUBLIC_GAMES.index(game)))
)

CONTRACT_SCHEMA = "tycho-nooa-public25-cohort-contract-v2"
CHECKPOINT_SCHEMA = "tycho-nooa-public25-checkpoint-v1"
RESULT_SCHEMA = "tycho-nooa-public25-resumable-result-v1"
HEX64 = re.compile(r"[0-9a-f]{64}")
PROJECTION_EVIDENCE_SCHEMA = "tycho-nooa-public25-runtime-projection-evidence-v1"
STORAGE_LEDGER_SCHEMA = "tycho-nooa-public25-storage-ledger-v1"
CANONICAL_TIMING_RUN_IDENTITY_SHA256 = (
    "b1bec82836612df537b5721925a7e624cfa1b1e67fb4d0ee84d98392c1d7dffe"
)
ARTIFACT_HASH_KEYS = {
    "adapter",
    "checkpoint_transport",
    "continued_play_probe",
    "guarded_runner",
    "local_solver",
    "model_manifest",
    "official_source_manifest",
    "outer_callback_contract",
    "outer_runtime",
    "public_environment_manifest",
    "resumable_harness",
    "runtime_dataset_manifest",
    "runtime_projection_estimator",
    "runtime_projection_policy",
    "solver_wheel",
    "solver_wheelhouse_manifest",
}


class StorageAdmissionError(RuntimeError):
    def __init__(self, facts: dict[str, Any]) -> None:
        self.facts = facts
        super().__init__(
            "public-25 storage admission requires "
            f"{facts['required_available_bytes']} available bytes; "
            f"found {facts['available_bytes']}"
        )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def regular_file_manifest(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"evidence root is not one regular directory: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"evidence contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            value = path.read_bytes()
            records.append(
                {"path": relative, "bytes": len(value), "sha256": sha256_bytes(value)}
            )
        elif not path.is_dir():
            raise RuntimeError(f"evidence contains a non-regular entry: {path}")
    return records


def regular_file_bytes(root: Path | None) -> int:
    """Count regular-file bytes without following model- or host-created symlinks."""
    if root is None or not root.exists():
        return 0
    if root.is_symlink():
        return 0
    root_mode = root.lstat().st_mode
    if stat.S_ISREG(root_mode):
        return root.lstat().st_size
    if not stat.S_ISDIR(root_mode):
        return 0
    total = 0
    for path in root.rglob("*"):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            # Server logs can rotate while the host-owned observer scans them.
            continue
        if stat.S_ISREG(metadata.st_mode):
            total += metadata.st_size
    return total


def filesystem_bytes(root: Path) -> dict[str, int]:
    value = os.statvfs(root)
    return {
        "fragment_bytes": value.f_frsize,
        "total_bytes": value.f_blocks * value.f_frsize,
        "free_bytes": value.f_bfree * value.f_frsize,
        "available_bytes": value.f_bavail * value.f_frsize,
    }


def storage_component(root: Path | None) -> dict[str, Any]:
    return {
        "path": str(root) if root is not None else None,
        "exists": root is not None and root.exists(),
        "regular_file_bytes": regular_file_bytes(root),
    }


def append_storage_ledger(
    ledger_path: Path,
    *,
    sequence: int,
    event: str,
    storage_root: Path,
    predecessor: Path | None,
    checkpoint_root: Path,
    workspace_root: Path,
    scratch: Path | None = None,
    published_bundle: Path | None = None,
    failure_evidence: Path | None = None,
    game: str | None = None,
    admission: dict[str, Any] | None = None,
) -> None:
    """Append one durable host-owned measurement outside model scratch space."""
    record = {
        "schema": STORAGE_LEDGER_SCHEMA,
        "sequence": sequence,
        "captured_at": utc_now(),
        "event": event,
        "game": game,
        "storage_root": str(storage_root),
        "filesystem": filesystem_bytes(storage_root),
        # Components intentionally overlap.  In particular storage_root contains
        # predecessor, checkpoint, workspace, and the active scratch tree.
        "components": {
            "storage_root": storage_component(storage_root),
            "predecessor": storage_component(predecessor),
            "checkpoint": storage_component(checkpoint_root),
            "workspace": storage_component(workspace_root),
            "scratch": storage_component(scratch),
            "published_bundle": storage_component(published_bundle),
            "failure_evidence": storage_component(failure_evidence),
        },
        "admission": admission,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("ab") as handle:
        handle.write(canonical_json(record))
        handle.flush()
        os.fsync(handle.fileno())


def storage_admission(contract: dict[str, Any], storage_root: Path) -> dict[str, Any]:
    projection = contract["runtime_projection"]
    scratch_bound = projection["next_scratch_bound_bytes"]
    publication_margin = projection["storage_publication_margin_bytes"]
    required = 2 * scratch_bound + publication_margin
    available = filesystem_bytes(storage_root)["available_bytes"]
    facts = {
        "next_scratch_bound_bytes": scratch_bound,
        "storage_publication_margin_bytes": publication_margin,
        "required_available_bytes": required,
        "available_bytes": available,
        "passed": available >= required,
    }
    if not facts["passed"]:
        raise StorageAdmissionError(facts)
    return facts


def verify_regular_file_manifest(
    root: Path, records: list[dict[str, Any]], *, label: str
) -> None:
    expected_paths = [record.get("path") for record in records]
    if expected_paths != sorted(expected_paths) or len(expected_paths) != len(
        set(expected_paths)
    ):
        raise RuntimeError(f"{label} file manifest paths are not unique and sorted")
    actual = regular_file_manifest(root)
    if actual != records:
        raise RuntimeError(f"{label} file inventory or bytes differ")


def declared_games_manifest(
    games_root: Path, completed_games: list[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for game in completed_games:
        game_root = games_root / game
        for record in regular_file_manifest(game_root):
            records.append({**record, "path": f"{game}/{record['path']}"})
    return records


def verify_game_bundle(game_root: Path, game: str, record: dict[str, Any]) -> None:
    bundle_path = game_root / "bundle.json"
    if not bundle_path.is_file() or bundle_path.is_symlink():
        raise RuntimeError(f"game bundle manifest is absent for {game}")
    if sha256_file(bundle_path) != record.get("bundle_sha256"):
        raise RuntimeError(f"game bundle manifest hash differs for {game}")
    bundle = json.loads(bundle_path.read_text())
    expected_metrics = {key: value for key, value in record.items() if key != "bundle_sha256"}
    if (
        bundle.get("schema") != "tycho-nooa-public25-game-bundle-v1"
        or bundle.get("game") != game
        or bundle.get("metrics") != expected_metrics
    ):
        raise RuntimeError(f"game bundle identity or metrics differ for {game}")
    actual = [
        item
        for item in regular_file_manifest(game_root)
        if item["path"] != "bundle.json"
    ]
    if bundle.get("files") != actual:
        raise RuntimeError(f"game bundle evidence differs for {game}")


def validate_contract(contract: dict[str, Any]) -> str:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise RuntimeError("public-25 contract schema differs")
    if contract.get("public_games") != list(PUBLIC_GAMES):
        raise RuntimeError("public-25 contract game roster or order differs")
    if contract.get("action_caps") != FIRST_LEVEL_ACTION_CAPS:
        raise RuntimeError("public-25 contract action caps differ")
    if contract.get("run_order") != list(RUN_ORDER):
        raise RuntimeError("public-25 contract run order differs")
    if sum(FIRST_LEVEL_ACTION_CAPS.values()) != 1_056:
        raise AssertionError("public-25 first-level cohort cap changed")
    if contract.get("evidence_kind") != "public_progress_not_official_score":
        raise RuntimeError("public-25 evidence label differs")
    if contract.get("admission_policy") != "C1_sequential_game_shards":
        raise RuntimeError("resumable harness requires the C1 shard policy")
    envelope = contract.get("solver_envelope")
    if envelope != {
        "context_window": 32_768,
        "initial_actor_calls": 6,
        "initial_max_tokens": 16_000,
        "successor_actor_calls": 1,
        "successor_max_tokens": 16_000,
        "model_retries": 0,
        "mode": "single",
        "treatment": "single_decision_discipline",
        "level_boundary_policy": "deterministic_level_checkpoint",
    }:
        raise RuntimeError("public-25 Solver envelope differs")
    artifact_hashes = contract.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != ARTIFACT_HASH_KEYS:
        raise RuntimeError("public-25 contract artifact inventory differs")
    if any(
        not isinstance(name, str)
        or not isinstance(value, str)
        or HEX64.fullmatch(value) is None
        for name, value in artifact_hashes.items()
    ):
        raise RuntimeError("public-25 artifact hashes are malformed")
    projection = contract.get("runtime_projection")
    if not isinstance(projection, dict):
        raise TypeError("public-25 runtime projection is absent")
    expected_projection_fields = {
        "initial_decision_seconds",
        "successor_decision_seconds",
        "per_game_overhead_seconds",
        "per_game_scheduler_reserve_seconds",
        "safety_multiplier",
        "next_scratch_bound_bytes",
        "storage_publication_margin_bytes",
        "source_result_sha256",
        "estimator_schema",
        "projection_policy_sha256",
        "source_artifact_manifest_sha256",
        "derivation_artifact_sha256",
        "canonical_run_identity_sha256",
    }
    if set(projection) != expected_projection_fields:
        raise RuntimeError("runtime projection provenance fields differ")
    required_positive = (
        "initial_decision_seconds",
        "successor_decision_seconds",
        "per_game_overhead_seconds",
        "per_game_scheduler_reserve_seconds",
        "safety_multiplier",
    )
    if any(
        isinstance(projection.get(key), bool)
        or not isinstance(projection.get(key), (int, float))
        or not math.isfinite(projection[key])
        or projection[key] <= 0
        for key in required_positive
    ):
        raise RuntimeError("runtime projection has no positive measured bounds")
    if projection["safety_multiplier"] < 1:
        raise RuntimeError("runtime projection safety multiplier is below one")
    storage_fields = (
        "next_scratch_bound_bytes",
        "storage_publication_margin_bytes",
    )
    if any(
        type(projection.get(key)) is not int or projection[key] <= 0
        for key in storage_fields
    ):
        raise RuntimeError("runtime projection storage byte bounds are not positive integers")
    if HEX64.fullmatch(str(projection.get("source_result_sha256", ""))) is None:
        raise RuntimeError("runtime projection is not bound to predecessor evidence")
    if projection.get("estimator_schema") != PROJECTION_EVIDENCE_SCHEMA:
        raise RuntimeError("runtime projection estimator schema differs")
    for field in (
        "projection_policy_sha256",
        "source_artifact_manifest_sha256",
        "derivation_artifact_sha256",
    ):
        if HEX64.fullmatch(str(projection.get(field, ""))) is None:
            raise RuntimeError(f"runtime projection {field} is malformed")
    if (
        projection.get("canonical_run_identity_sha256")
        != CANONICAL_TIMING_RUN_IDENTITY_SHA256
    ):
        raise RuntimeError("runtime projection canonical run identity differs")
    return sha256_bytes(canonical_json(contract))


def projected_child_seconds(contract: dict[str, Any], game: str) -> int:
    projection = contract["runtime_projection"]
    calls = FIRST_LEVEL_ACTION_CAPS[game]
    raw = (
        projection["initial_decision_seconds"]
        + max(0, calls - 1) * projection["successor_decision_seconds"]
        + projection["per_game_overhead_seconds"]
    )
    return max(61, int(raw * projection["safety_multiplier"] + 0.999))


def projected_game_seconds(contract: dict[str, Any], game: str) -> int:
    projection = contract["runtime_projection"]
    return (
        projected_child_seconds(contract, game)
        + projection["per_game_scheduler_reserve_seconds"]
    )


def empty_checkpoint(contract_sha256: str) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "contract_sha256": contract_sha256,
        "generation": 0,
        "predecessor_manifest_sha256": None,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "completed_games": [],
        "games": {},
        "failures": [],
        "files": [],
    }


def verify_checkpoint(root: Path, contract_sha256: str) -> dict[str, Any]:
    manifest_path = root / "checkpoint.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("predecessor checkpoint manifest is absent")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("predecessor checkpoint schema differs")
    if manifest.get("contract_sha256") != contract_sha256:
        raise RuntimeError("predecessor checkpoint contract differs")
    completed = manifest.get("completed_games")
    if (
        not isinstance(completed, list)
        or completed != [game for game in PUBLIC_GAMES if game in completed]
        or len(completed) != len(set(completed))
    ):
        raise RuntimeError("predecessor completed-game roster differs")
    if set(manifest.get("games", {})) != set(completed):
        raise RuntimeError("predecessor game records differ from completed roster")
    if declared_games_manifest(root / "games", completed) != manifest.get("files", []):
        raise RuntimeError("checkpoint file inventory or bytes differ")
    for game in completed:
        verify_game_bundle(root / "games" / game, game, manifest["games"][game])
    for failure in manifest.get("failures", []):
        game = failure.get("game")
        if game not in PUBLIC_GAMES:
            raise RuntimeError("checkpoint failure game differs")
        verify_regular_file_manifest(
            root / "failures" / game,
            failure.get("files", []),
            label=f"failure evidence for {game}",
        )
    return manifest


def import_predecessor(
    predecessor: Path,
    current: Path,
    *,
    contract_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    if HEX64.fullmatch(expected_manifest_sha256) is None:
        raise RuntimeError("expected predecessor manifest hash is malformed")
    manifest = verify_checkpoint(predecessor, contract_sha256)
    actual_hash = sha256_file(predecessor / "checkpoint.json")
    if actual_hash != expected_manifest_sha256:
        raise RuntimeError("predecessor checkpoint manifest hash differs")
    games_source = predecessor / "games"
    games_target = current / "games"
    if games_target.exists():
        raise RuntimeError("current checkpoint already contains games")
    games_target.mkdir(parents=True)
    for game in manifest["completed_games"]:
        shutil.copytree(games_source / game, games_target / game)
    if declared_games_manifest(games_target, manifest["completed_games"]) != manifest["files"]:
        raise RuntimeError("copied checkpoint file inventory or bytes differ")
    result = dict(manifest)
    result["generation"] = int(manifest["generation"]) + 1
    result["predecessor_manifest_sha256"] = actual_hash
    result["created_at"] = utc_now()
    result["updated_at"] = utc_now()
    result["failures"] = []
    return result


def result_metrics(result: dict[str, Any], game: str, cap: int) -> dict[str, Any]:
    if result.get("status") != "passed" or result.get("public_games") != [game]:
        raise RuntimeError(f"single-game launcher result did not pass for {game}")
    progress = result.get("public_progress")
    sessions = result.get("solver_sessions")
    if not isinstance(progress, dict) or progress.get("game_count") != 1:
        raise RuntimeError(f"public progress is incomplete for {game}")
    if not isinstance(sessions, dict) or sessions.get("game_count") != 1:
        raise RuntimeError(f"Solver sessions are incomplete for {game}")
    if set(progress.get("games", {})) != {game} or set(sessions.get("games", {})) != {game}:
        raise RuntimeError(f"game evidence inventory differs for {game}")
    game_progress = progress["games"][game]
    state = game_progress.get("final_state")
    actions = sessions.get("total_solver_actions")
    if not isinstance(actions, int) or not 1 <= actions <= cap:
        raise RuntimeError(f"Solver action count differs for {game}")
    if state != "WIN" and actions != cap:
        raise RuntimeError(f"non-winning game stopped before its cap for {game}")
    calls = sessions.get("total_actor_calls")
    if not isinstance(calls, int) or not actions <= calls <= actions + 5:
        raise RuntimeError(f"Actor call count differs for {game}")
    session = sessions["games"][game]
    trace_path = Path(session["session"]) / "llm-trace.jsonl"
    prompt_tokens = completion_tokens = reasoning_tokens = 0
    # The path is valid before the scratch bundle is atomically moved.  Token
    # totals are copied into the manifest so later aggregation never trusts it.
    for line in trace_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            prompt_tokens += int(row.get("prompt_tokens") or 0)
            completion_tokens += int(row.get("completion_tokens") or 0)
            reasoning_tokens += int(row.get("reasoning_tokens") or 0)
    return {
        "action_cap": cap,
        "solver_actions": actions,
        "actor_calls": calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "final_state": state,
        "final_levels_completed": game_progress.get("final_levels_completed"),
        "max_levels_completed": game_progress.get("max_levels_completed"),
        "recorded_action_results": game_progress.get("recorded_action_results"),
    }


def publish_completed_game(
    scratch: Path,
    checkpoint_root: Path,
    checkpoint: dict[str, Any],
    *,
    game: str,
    result: dict[str, Any],
) -> None:
    metrics = result_metrics(result, game, FIRST_LEVEL_ACTION_CAPS[game])
    staging = checkpoint_root / ".inflight" / game
    if staging.exists():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True)
    for relative in (
        "game-result.json",
        "orchestrator-command.json",
        "public25-launcher.stdout.log",
        "public25-launcher.stderr.log",
    ):
        source = scratch / relative
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"required game evidence is absent: {source}")
        shutil.copy2(source, staging / relative)
    for relative in ("solver-workspaces", "recordings"):
        source = scratch / relative
        if not source.is_dir() or source.is_symlink():
            raise RuntimeError(f"required game evidence is absent: {source}")
        shutil.copytree(source, staging / relative)
    official_log = scratch / "ARC-AGI-3-Agents" / "logs.log"
    if not official_log.is_file() or official_log.is_symlink():
        raise RuntimeError(f"required official log is absent: {official_log}")
    destination_log = staging / "official" / "logs.log"
    destination_log.parent.mkdir()
    shutil.copy2(official_log, destination_log)
    bundle_files = regular_file_manifest(staging)
    write_json_atomic(staging / "bundle.json", {
        "schema": "tycho-nooa-public25-game-bundle-v1",
        "game": game,
        "metrics": metrics,
        "files": bundle_files,
    })
    destination = checkpoint_root / "games" / game
    if destination.exists():
        raise RuntimeError(f"completed game already exists: {game}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)
    checkpoint["games"][game] = {
        **metrics,
        "bundle_sha256": sha256_file(destination / "bundle.json"),
    }
    checkpoint["completed_games"] = [
        candidate for candidate in PUBLIC_GAMES if candidate in checkpoint["games"]
    ]
    checkpoint["files"] = declared_games_manifest(
        checkpoint_root / "games", checkpoint["completed_games"]
    )
    checkpoint["updated_at"] = utc_now()
    write_json_atomic(checkpoint_root / "checkpoint.json", checkpoint)


def aggregate(checkpoint: dict[str, Any]) -> dict[str, Any]:
    games = checkpoint["games"]
    actions = sum(record["solver_actions"] for record in games.values())
    levels = sum(record["max_levels_completed"] for record in games.values())
    calls = sum(record["actor_calls"] for record in games.values())
    return {
        "evidence_kind": "public_progress_not_official_score",
        "official_score": None,
        "completed_game_count": len(games),
        "remaining_games": [game for game in PUBLIC_GAMES if game not in games],
        "total_solver_actions": actions,
        "total_actor_calls": calls,
        "total_prompt_tokens": sum(record["prompt_tokens"] for record in games.values()),
        "total_completion_tokens": sum(record["completion_tokens"] for record in games.values()),
        "total_reasoning_tokens": sum(record["reasoning_tokens"] for record in games.values()),
        "total_max_public_levels_completed": levels,
        "total_final_public_levels_completed": sum(
            record["final_levels_completed"] for record in games.values()
        ),
        "total_recorded_action_results": sum(
            record["recorded_action_results"] for record in games.values()
        ),
        "games_with_public_level_progress": sum(
            record["max_levels_completed"] > 0 for record in games.values()
        ),
        "public_levels_per_solver_action": (levels / actions if actions else None),
        "actor_calls_per_solver_action": (calls / actions if actions else None),
        "reconciled_solver_action_rate": (1.0 if actions else None),
        "won_public_games": sum(record["final_state"] == "WIN" for record in games.values()),
        "games": games,
    }


def prepare_scratch(template_root: Path, scratch: Path) -> None:
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    shutil.copytree(template_root / "ARC-AGI-3-Agents", scratch / "ARC-AGI-3-Agents")
    shutil.copy2(template_root / "run_official_agent.py", scratch / "run_official_agent.py")
    dotenv = scratch / "ARC-AGI-3-Agents" / ".env"
    lines = []
    found = False
    for line in dotenv.read_text().splitlines():
        if line.startswith("RECORDINGS_DIR="):
            line = f"RECORDINGS_DIR={scratch / 'recordings'}"
            found = True
        lines.append(line)
    if not found:
        raise RuntimeError("official template has no RECORDINGS_DIR")
    dotenv.write_text("\n".join(lines) + "\n")


def discard_completed_scratch(scratch: Path) -> None:
    """Remove a game scratch tree only after its checkpoint bundle is durable."""
    if not scratch.is_dir() or scratch.is_symlink():
        raise RuntimeError("completed game scratch is not one removable directory")
    shutil.rmtree(scratch)
    if scratch.exists():
        raise RuntimeError("completed game scratch was not removed")


def process_group_exists(process_group_id: int) -> bool:
    """Return whether a child-created process group has a live member.

    ``killpg(..., 0)`` also reports groups containing only zombies.  Those
    processes cannot execute or retain resources other than their exit status,
    so on Linux inspect ``/proc`` before treating such a group as uncontained.
    """
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return True
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            state = fields[0]
            process_group = int(fields[2])
        except (FileNotFoundError, IndexError, PermissionError, ValueError):
            continue
        if process_group == process_group_id and state != "Z":
            return True
    return False


def contain_process_group(
    process_group_id: int,
    *,
    termination_grace_s: float,
) -> dict[str, bool]:
    """Terminate one child-owned process group, escalating only if necessary."""
    term_sent = False
    kill_sent = False
    if process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        else:
            term_sent = True

    deadline = time.monotonic() + termination_grace_s
    while process_group_exists(process_group_id) and time.monotonic() < deadline:
        time.sleep(0.01)

    if process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        else:
            kill_sent = True
        deadline = time.monotonic() + max(termination_grace_s, 0.1)
        while process_group_exists(process_group_id) and time.monotonic() < deadline:
            time.sleep(0.01)

    return {
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "process_group_closed": not process_group_exists(process_group_id),
    }


def run_game(
    *,
    game: str,
    timeout_s: int,
    args: argparse.Namespace,
    scratch: Path,
    communication_grace_s: float = 60,
    termination_grace_s: float = 10,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
    prepare_scratch(args.template_root, scratch)
    result_path = scratch / "game-result.json"
    environment = {
        **os.environ,
        "TYCHO_PUBLIC_GAME_IDS": game,
        "TYCHO_KAGGLE_MAX_ACTIONS": str(FIRST_LEVEL_ACTION_CAPS[game]),
        "TYCHO_PUBLIC25_WORKING_ROOT": str(scratch),
    }
    command = [
        sys.executable,
        str(args.game_probe),
        "--base-url", args.base_url,
        "--model", args.model,
        "--workspace-root", str(scratch / "solver-workspaces"),
        "--output", str(result_path),
        "--timeout-s", str(timeout_s),
    ]
    stdout_path = scratch / "orchestrator-child.stdout.log"
    stderr_path = scratch / "orchestrator-child.stderr.log"
    timed_out = False
    residual_group_term_required = False
    spawn_error: str | None = None
    child_returncode: int | None = None
    containment = {
        "term_sent": False,
        "kill_sent": False,
        "process_group_closed": True,
    }

    try:
        with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
            process = subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                env=environment,
                start_new_session=True,
            )
            try:
                child_returncode = process.wait(timeout=timeout_s + communication_grace_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                containment = contain_process_group(
                    process.pid,
                    termination_grace_s=termination_grace_s,
                )
                child_returncode = process.wait(timeout=max(termination_grace_s, 0.1))
            else:
                residual_group_term_required = process_group_exists(process.pid)
                if residual_group_term_required:
                    containment = contain_process_group(
                        process.pid,
                        termination_grace_s=termination_grace_s,
                    )
    except OSError as error:
        spawn_error = f"{type(error).__name__}: {error}"

    stdout = stdout_path.read_text() if stdout_path.is_file() else ""
    stderr = stderr_path.read_text() if stderr_path.is_file() else ""
    if spawn_error is not None:
        stderr = f"{stderr}{spawn_error}\n"
        returncode = 127
    elif timed_out:
        returncode = 124
    elif not containment["process_group_closed"]:
        returncode = 75
    else:
        assert child_returncode is not None
        returncode = child_returncode

    completed = subprocess.CompletedProcess(command, returncode, stdout, stderr)
    write_json_atomic(scratch / "orchestrator-command.json", {
        "returncode": completed.returncode,
        "child_returncode": child_returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timeout_seconds": timeout_s,
        "communication_grace_seconds": communication_grace_s,
        "termination_grace_seconds": termination_grace_s,
        "timed_out": timed_out,
        "residual_group_term_required": residual_group_term_required,
        "term_sent": containment["term_sent"],
        "kill_sent": containment["kill_sent"],
        "process_group_closed": containment["process_group_closed"],
        "spawn_error": spawn_error,
    })
    result = json.loads(result_path.read_text()) if result_path.is_file() else None
    return completed, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=int, required=True)
    parser.add_argument("--contract", type=Path, default=Path("/tmp/public25-contract.json"))
    parser.add_argument("--template-root", type=Path, default=Path("/tmp/public25-template"))
    parser.add_argument("--game-probe", type=Path, default=Path("/tmp/public25_launcher_probe.py"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/kaggle/working/public25-checkpoint"))
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--expected-predecessor-sha256")
    parser.add_argument("--stop-reserve-s", type=int, default=1_200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        raise RuntimeError("resumable public-25 harness refuses competition rerun mode")
    contract = json.loads(args.contract.read_text())
    contract_sha256 = validate_contract(contract)
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    storage_root = args.workspace_root.parent
    ledger_path = args.workspace_root / "storage-ledger.jsonl"
    if ledger_path.exists():
        raise RuntimeError("public-25 storage ledger already exists")
    ledger_sequence = 0

    def record_storage(
        event: str,
        *,
        game: str | None = None,
        scratch: Path | None = None,
        published_bundle: Path | None = None,
        failure_evidence: Path | None = None,
        admission: dict[str, Any] | None = None,
    ) -> None:
        nonlocal ledger_sequence
        append_storage_ledger(
            ledger_path,
            sequence=ledger_sequence,
            event=event,
            storage_root=storage_root,
            predecessor=args.predecessor,
            checkpoint_root=args.checkpoint_root,
            workspace_root=args.workspace_root,
            scratch=scratch,
            published_bundle=published_bundle,
            failure_evidence=failure_evidence,
            game=game,
            admission=admission,
        )
        ledger_sequence += 1

    # The outer transport has already atomically materialized a predecessor by
    # the time this harness starts.  Measure it before duplicating game bundles.
    record_storage("after_predecessor_transport_import")
    args.checkpoint_root.mkdir(parents=True, exist_ok=False)
    if args.predecessor is None:
        if args.expected_predecessor_sha256 is not None:
            raise RuntimeError("predecessor hash supplied without predecessor")
        checkpoint = empty_checkpoint(contract_sha256)
        (args.checkpoint_root / "games").mkdir()
    else:
        if args.expected_predecessor_sha256 is None:
            raise RuntimeError("predecessor is not bound to an expected manifest hash")
        checkpoint = import_predecessor(
            args.predecessor,
            args.checkpoint_root,
            contract_sha256=contract_sha256,
            expected_manifest_sha256=args.expected_predecessor_sha256,
        )
    write_json_atomic(args.checkpoint_root / "checkpoint.json", checkpoint)
    record_storage("after_predecessor_copy")

    remaining = [game for game in RUN_ORDER if game not in checkpoint["games"]]
    if not remaining:
        raise RuntimeError("predecessor already contains the complete public cohort")
    started = time.monotonic()
    deadline = started + args.timeout_s
    stop_reason = "complete"
    failure: dict[str, Any] | None = None
    games_attempted_this_run: list[str] = []
    games_completed_this_run: list[str] = []
    for game in remaining:
        projected = projected_game_seconds(contract, game)
        child_budget = projected_child_seconds(contract, game)
        scheduler_reserve = contract["runtime_projection"][
            "per_game_scheduler_reserve_seconds"
        ]
        available = int(deadline - time.monotonic()) - args.stop_reserve_s
        if available < projected:
            stop_reason = "partial_budget"
            break
        scratch = args.workspace_root / "scratch" / game
        try:
            admission = storage_admission(contract, storage_root)
        except StorageAdmissionError as error:
            record_storage(
                "pre_game_admission",
                game=game,
                scratch=scratch,
                admission=error.facts,
            )
            raise
        record_storage(
            "pre_game_admission",
            game=game,
            scratch=scratch,
            admission=admission,
        )
        games_attempted_this_run.append(game)
        completed, result = run_game(
            game=game,
            timeout_s=min(child_budget, available - scheduler_reserve),
            args=args,
            scratch=scratch,
        )
        record_storage("post_game_prepublication", game=game, scratch=scratch)
        if completed.returncode != 0 or result is None or result.get("status") != "passed":
            stop_reason = "partial_failure"
            failure = {
                "game": game,
                "returncode": completed.returncode,
                "result_status": result.get("status") if result else None,
            }
            failure_root = args.checkpoint_root / "failures" / game
            if failure_root.exists():
                raise RuntimeError(f"failure evidence already exists for {game}")
            record_storage(
                "pre_failure_copy",
                game=game,
                scratch=scratch,
                failure_evidence=failure_root,
            )
            shutil.copytree(scratch, failure_root)
            checkpoint["failures"].append({
                **failure,
                "files": regular_file_manifest(failure_root),
            })
            checkpoint["updated_at"] = utc_now()
            write_json_atomic(args.checkpoint_root / "checkpoint.json", checkpoint)
            record_storage(
                "post_failure_copy",
                game=game,
                scratch=scratch,
                failure_evidence=failure_root,
            )
            break
        publish_completed_game(
            scratch,
            args.checkpoint_root,
            checkpoint,
            game=game,
            result=result,
        )
        published_bundle = args.checkpoint_root / "games" / game
        record_storage(
            "post_publication_pre_cleanup",
            game=game,
            scratch=scratch,
            published_bundle=published_bundle,
        )
        discard_completed_scratch(scratch)
        record_storage(
            "post_publication_cleanup",
            game=game,
            scratch=scratch,
            published_bundle=published_bundle,
        )
        games_completed_this_run.append(game)
    metrics = aggregate(checkpoint)
    if not metrics["remaining_games"]:
        stop_reason = "complete"
    record = {
        # `status` is route execution for compatibility with the accepted R8
        # callback.  `qualification_status` is the only all-25 completion gate.
        "schema": RESULT_SCHEMA,
        "status": "passed",
        "qualification_status": stop_reason,
        "all25_complete": stop_reason == "complete" and not metrics["remaining_games"],
        "games_attempted_this_run": games_attempted_this_run,
        "games_completed_this_run": games_completed_this_run,
        "evidence_kind": "public_progress_not_official_score",
        "partial_selection_bias": "shorter_action_cap_cohorts_run_first",
        "contract_sha256": contract_sha256,
        "checkpoint_manifest_sha256": sha256_file(args.checkpoint_root / "checkpoint.json"),
        "storage_safety": {
            "ledger_schema": STORAGE_LEDGER_SCHEMA,
            "ledger_path": str(ledger_path),
            "ledger_sha256": sha256_file(ledger_path),
            "ledger_record_count": ledger_sequence,
            "next_scratch_bound_bytes": contract["runtime_projection"][
                "next_scratch_bound_bytes"
            ],
            "storage_publication_margin_bytes": contract["runtime_projection"][
                "storage_publication_margin_bytes"
            ],
        },
        "failure": failure,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "metrics": metrics,
    }
    write_json_atomic(args.output, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
