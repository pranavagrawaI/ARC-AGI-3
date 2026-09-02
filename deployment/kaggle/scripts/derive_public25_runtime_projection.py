#!/usr/bin/env python3
"""Derive a bound public-25 schedule from an exact public-three Kaggle run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from types import ModuleType
from typing import Any

EXPECTED_GAMES = ("ls20", "s5i5", "sb26")
EXPECTED_ACTIONS_PER_GAME = 32
EXPECTED_LEVEL_BOUNDARY_POLICY = "deterministic_level_checkpoint"
EXPECTED_GPU_MARKER = "NVIDIA RTX PRO 6000"
EXPECTED_GPU_COMMAND = [
    "/opt/bin/nvidia-smi",
    "--query-gpu=name,memory.total,memory.free,driver_version",
    "--format=csv,noheader,nounits",
]
EXPECTED_OUTER_SCHEMA = "qwen3.8-fp8-r8-l3b-probe-v1"
EXPECTED_OUTER_MODE = "s0-official-launcher"
EXPECTED_OFFICIAL_SCHEMA = "tycho-nooa-public25-launcher-result-v2"
EXPECTED_MODEL = "/kaggle/input/datasets/jakobbrggen/qwen3-8-27b-fp8-hf-snapshot"
EXPECTED_WORKER_METADATA = {
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
    "level_boundary_policy": EXPECTED_LEVEL_BOUNDARY_POLICY,
}
EXPECTED_REQUEST_CEILING = 1_181
EXPECTED_DECISION_CEILING = 1_056
PROJECTION_SCHEMA = "tycho-nooa-public25-runtime-projection-evidence-v1"
POLICY_SCHEMA = "tycho-nooa-public25-runtime-projection-policy-v1"
RECONCILIATION_SCHEMA = "tycho-nooa-public3-canonical-run-reconciliation-v1"
EXPECTED_CANDIDATE_IDENTITY_KEYS = {
    "builder_sha256",
    "experiment_card_sha256",
    "frozen_metadata_id",
    "frozen_metadata_sha256",
    "frozen_notebook_raw_sha256",
    "launch_identity_sha256",
    "notebook_canonical_json_sha256",
    "observed_kernel_slug",
    "observed_kernel_version",
    "outer_runtime_sha256",
    "probe_sha256",
    "push_record_sha256",
    "remote_metadata_sha256",
    "remote_notebook_raw_sha256",
    "remote_source_equality_sha256",
    "remote_source_manifest_entries_sha256",
    "remote_source_manifest_sha256",
    "remote_source_pull_sha256",
    "solver_wheel_sha256",
    "wheelhouse_manifest_sha256",
}
EXPECTED_CANDIDATE_DIGEST_KEYS = {
    key for key in EXPECTED_CANDIDATE_IDENTITY_KEYS if key.endswith("_sha256")
}

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = SCRIPT_ROOT / "public25-runtime-projection-policy-v1.json"
HARNESS_PATH = (
    SCRIPT_ROOT / "s0-public25-resumable-v2-candidate/resumable_public25.py"
)

ACTION_IDS = {
    "RESET": 0,
    "UP": 1,
    "DOWN": 2,
    "LEFT": 3,
    "RIGHT": 4,
    "SPACEBAR": 5,
    "CLICK": 6,
    "UNDO": 7,
}
LOG_TIMESTAMP = r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
CREATED_RECORDING = re.compile(
    rf"^{LOG_TIMESTAMP} \| INFO \| created new recording for "
    r"(?P<game>[a-z0-9]+)\."
)
ACTION_EVENT = re.compile(
    rf"^{LOG_TIMESTAMP} \| INFO \| (?P<game>[a-z0-9]+) - "
    r"ACTION(?P<action_id>[0-7]): count (?P<count>\d+),"
)
RECORDING_AVAILABLE = re.compile(
    rf"^{LOG_TIMESTAMP} \| INFO \| recording for "
    r"(?P<game>[a-z0-9]+)\.myagent\."
)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_facts(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"evidence is not one regular file: {path}")
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = path.name
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} is not a JSON object: {path}")
    return value


def parse_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{label} timestamp is absent")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RuntimeError(f"{label} timestamp is invalid: {value!r}") from error
    if parsed.tzinfo is None:
        raise RuntimeError(f"{label} timestamp is timezone-naive: {value!r}")
    return parsed.astimezone(UTC)


def parse_log_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=UTC)


def unique_file(root: Path, pattern: str, label: str) -> Path:
    matches = [path for path in root.rglob(pattern) if path.is_file() and not path.is_symlink()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label}, found {matches}")
    return matches[0]


def load_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("projection_public25_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("public-25 V2 harness could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def positive_number(value: object, label: str, *, minimum: float = 0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= minimum
    ):
        raise RuntimeError(f"{label} is not a finite number greater than {minimum}")
    return float(value)


def load_policy(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = read_json(path, "runtime projection policy")
    if policy.get("schema") != POLICY_SCHEMA:
        raise RuntimeError("runtime projection policy schema differs")
    safety = positive_number(policy.get("safety_multiplier"), "safety multiplier")
    if safety < 1:
        raise RuntimeError("safety multiplier must be at least one")
    for key in (
        "per_game_scheduler_reserve_seconds",
        "callback_timeout_seconds",
        "stop_reserve_seconds",
    ):
        value = policy.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"runtime projection policy {key} differs")
    usable = policy["callback_timeout_seconds"] - policy["stop_reserve_seconds"]
    if usable <= 0 or policy.get("usable_generation_seconds") != usable:
        raise RuntimeError("runtime projection policy usable generation time differs")
    identity = policy.get("expected_candidate_identity")
    if not isinstance(identity, dict) or set(identity) != EXPECTED_CANDIDATE_IDENTITY_KEYS:
        raise RuntimeError("runtime projection policy candidate identity is absent")
    for key in EXPECTED_CANDIDATE_DIGEST_KEYS:
        if (
            not isinstance(identity[key], str)
            or re.fullmatch(r"[0-9a-f]{64}", identity[key]) is None
        ):
            raise RuntimeError(
                f"runtime projection policy candidate identity {key} differs"
            )
    for key in ("frozen_metadata_id", "observed_kernel_slug"):
        if not isinstance(identity[key], str) or not identity[key].strip():
            raise RuntimeError(
                f"runtime projection policy candidate identity {key} differs"
            )
    observed_version = identity["observed_kernel_version"]
    if (
        isinstance(observed_version, bool)
        or not isinstance(observed_version, int)
        or observed_version < 1
    ):
        raise RuntimeError("runtime projection policy observed kernel version differs")
    if policy.get("expected_candidate_artifact_manifest_sha256") != sha256_bytes(
        canonical_json(identity)
    ):
        raise RuntimeError("runtime projection policy candidate manifest differs")
    facts = file_facts(path, path.parent)
    return policy, facts


def validate_reconciliation(
    reconciliation: dict[str, Any], policy: dict[str, Any]
) -> None:
    kernel = reconciliation.get("kernel")
    if (
        reconciliation.get("schema") != RECONCILIATION_SCHEMA
        or reconciliation.get("status") != "canonical_valid"
        or not isinstance(kernel, dict)
        or kernel.get("slug")
        != policy["expected_candidate_identity"]["observed_kernel_slug"]
        or kernel.get("version")
        != policy["expected_candidate_identity"]["observed_kernel_version"]
        or reconciliation.get("candidate_identity")
        != policy["expected_candidate_identity"]
        or reconciliation.get("candidate_artifact_manifest_sha256")
        != policy["expected_candidate_artifact_manifest_sha256"]
    ):
        raise RuntimeError("canonical run reconciliation identity differs")
    decision = reconciliation.get("decision")
    if decision != {
        "canonical_source_identity_verified": True,
        "metadata_identity_verified": True,
        "output_preserved_before_derivation": True,
    }:
        raise RuntimeError("canonical run reconciliation decision differs")


def source_identity_file_facts(
    output_root: Path, policy: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Verify the preserved V10 launch and remote-source evidence from bytes."""
    identity = policy["expected_candidate_identity"]
    named_paths = {
        "experiment_card": unique_file(
            output_root, "EXPERIMENT-CARD.md", "V10 experiment card"
        ),
        "launch_identity": unique_file(
            output_root,
            "kaggle-v10-launch-identity.json",
            "V10 launch identity",
        ),
        "push_record": unique_file(
            output_root, "kaggle-v10-push-record.json", "V10 push record"
        ),
        "remote_source_equality": unique_file(
            output_root,
            "kernel-v8-preterminal-source-equality.json",
            "V10 remote source equality",
        ),
        "remote_source_pull": unique_file(
            output_root,
            "kernel-v8-preterminal-source-pull-record.json",
            "V10 remote source pull record",
        ),
        "remote_source_manifest": unique_file(
            output_root,
            "kernel-v8-source-preterminal-sha256-manifest.json",
            "V10 remote source manifest",
        ),
    }
    expected_hashes = {
        "experiment_card": "experiment_card_sha256",
        "launch_identity": "launch_identity_sha256",
        "push_record": "push_record_sha256",
        "remote_source_equality": "remote_source_equality_sha256",
        "remote_source_pull": "remote_source_pull_sha256",
        "remote_source_manifest": "remote_source_manifest_sha256",
    }
    facts = {
        key: file_facts(path, output_root) for key, path in named_paths.items()
    }
    for key, identity_key in expected_hashes.items():
        if facts[key]["sha256"] != identity[identity_key]:
            raise RuntimeError(f"source identity {key} bytes differ")

    source_manifest_path = named_paths["remote_source_manifest"]
    source_directory = source_manifest_path.parent / "kernel-v8-source-preterminal"
    remote_metadata_path = source_directory / "kernel-metadata.json"
    remote_notebook_path = (
        source_directory / "prototype-tycho-nooa-public-3-full-timing.ipynb"
    )
    facts["remote_metadata"] = file_facts(remote_metadata_path, output_root)
    facts["remote_notebook"] = file_facts(remote_notebook_path, output_root)
    if facts["remote_metadata"]["sha256"] != identity["remote_metadata_sha256"]:
        raise RuntimeError("source identity remote metadata bytes differ")
    if facts["remote_notebook"]["sha256"] != identity["remote_notebook_raw_sha256"]:
        raise RuntimeError("source identity remote notebook bytes differ")

    launch = read_json(named_paths["launch_identity"], "V10 launch identity")
    if launch != {
        "captured_at_utc": launch.get("captured_at_utc"),
        "competition_submission": False,
        "experiment_card_sha256": identity["experiment_card_sha256"],
        "frozen_metadata_id": identity["frozen_metadata_id"],
        "interpretation": launch.get("interpretation"),
        "metadata_id_status": {"result": "404 Not Found", "returncode": 1},
        "observed_slug_status": {
            "result": "KernelWorkerStatus.RUNNING",
            "returncode": 0,
        },
        "observed_status": "RUNNING",
        "observed_version": identity["observed_kernel_version"],
        "push_record_sha256": identity["push_record_sha256"],
        "push_stdout_observed_kernel": identity["observed_kernel_slug"],
        "schema": "kaggle-private-kernel-launch-identity.v1",
    }:
        raise RuntimeError("source identity launch record differs")
    parse_timestamp(launch.get("captured_at_utc"), "V10 launch capture")
    if not isinstance(launch.get("interpretation"), str) or not launch["interpretation"]:
        raise RuntimeError("source identity launch interpretation is absent")

    push = read_json(named_paths["push_record"], "V10 push record")
    push_stdout = push.get("stdout")
    if (
        push.get("schema") != "kaggle-private-kernel-push.v1"
        or push.get("competition_submission") is not False
        or push.get("returncode") != 0
        or push.get("kernel") != identity["frozen_metadata_id"]
        or not isinstance(push.get("candidate"), str)
        or not push["candidate"]
        or not isinstance(push.get("command"), list)
        or not isinstance(push_stdout, str)
        or f"Kernel version {identity['observed_kernel_version']} successfully pushed"
        not in push_stdout
        or identity["observed_kernel_slug"] not in push_stdout
    ):
        raise RuntimeError("source identity push record differs")
    push_started = parse_timestamp(push.get("started_at_utc"), "V10 push start")
    push_ended = parse_timestamp(push.get("ended_at_utc"), "V10 push end")
    if push_ended <= push_started:
        raise RuntimeError("source identity push chronology differs")

    equality = read_json(
        named_paths["remote_source_equality"], "V10 remote source equality"
    )
    expected_equality = {
        "canonical_equal": True,
        "local_canonical_json_sha256": identity["notebook_canonical_json_sha256"],
        "local_raw_sha256": identity["frozen_notebook_raw_sha256"],
        "remote_canonical_json_sha256": identity[
            "notebook_canonical_json_sha256"
        ],
        "remote_metadata_sha256": identity["remote_metadata_sha256"],
        "remote_raw_sha256": identity["remote_notebook_raw_sha256"],
        "schema": "kaggle-v10-preterminal-source-equality.v1",
    }
    if any(equality.get(key) != value for key, value in expected_equality.items()):
        raise RuntimeError("source identity remote equality record differs")
    for path_key in ("local_path", "remote_metadata_path", "remote_path"):
        if not isinstance(equality.get(path_key), str) or not equality[path_key]:
            raise RuntimeError(f"source identity equality {path_key} is absent")

    pull = read_json(
        named_paths["remote_source_pull"], "V10 remote source pull record"
    )
    if (
        pull.get("schema") != "kaggle-v10-preterminal-source-binding.v1"
        or pull.get("competition_submission") is not False
        or pull.get("returncode") != 0
        or pull.get("slug") != identity["observed_kernel_slug"]
        or pull.get("version_context") != identity["observed_kernel_version"]
        or not isinstance(pull.get("binding_condition"), str)
        or not pull["binding_condition"]
    ):
        raise RuntimeError("source identity remote pull record differs")
    pull_started = parse_timestamp(pull.get("started_at_utc"), "V10 source pull start")
    pull_ended = parse_timestamp(pull.get("ended_at_utc"), "V10 source pull end")
    if pull_ended <= pull_started:
        raise RuntimeError("source identity remote pull chronology differs")

    source_manifest = read_json(source_manifest_path, "V10 remote source manifest")
    actual_entries = [
        {
            "kind": "file",
            "path": "kernel-metadata.json",
            "sha256": facts["remote_metadata"]["sha256"],
            "size": facts["remote_metadata"]["bytes"],
        },
        {
            "kind": "file",
            "path": remote_notebook_path.name,
            "sha256": facts["remote_notebook"]["sha256"],
            "size": facts["remote_notebook"]["bytes"],
        },
    ]
    entries_digest = sha256_bytes(
        json.dumps(actual_entries, sort_keys=True, separators=(",", ":")).encode()
    )
    if (
        source_manifest.get("entries") != actual_entries
        or source_manifest.get("file_count") != len(actual_entries)
        or source_manifest.get("total_file_bytes")
        != sum(entry["size"] for entry in actual_entries)
        or source_manifest.get("canonical_entries_sha256") != entries_digest
        or entries_digest != identity["remote_source_manifest_entries_sha256"]
    ):
        raise RuntimeError("source identity remote source manifest differs")

    remote_metadata = read_json(remote_metadata_path, "V10 remote kernel metadata")
    if (
        remote_metadata.get("id") != identity["observed_kernel_slug"]
        or remote_metadata.get("title") != "PROTOTYPE Tycho NOOA public 3 full timing"
        or remote_metadata.get("code_file") != remote_notebook_path.name
        or remote_metadata.get("language") != "python"
        or remote_metadata.get("kernel_type") != "notebook"
        or remote_metadata.get("is_private") is not True
        or remote_metadata.get("enable_gpu") is not True
        or remote_metadata.get("enable_tpu") is not False
        or remote_metadata.get("enable_internet") is not False
        or remote_metadata.get("machine_shape") != "NvidiaRtxPro6000"
        or remote_metadata.get("competition_sources")
        != ["arc-prize-2026-arc-agi-3"]
        or not isinstance(remote_metadata.get("dataset_sources"), list)
        or "pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v7"
        not in remote_metadata["dataset_sources"]
    ):
        raise RuntimeError("source identity remote kernel metadata differs")
    remote_notebook = read_json(remote_notebook_path, "V10 remote notebook")
    notebook_canonical_sha256 = sha256_bytes(
        json.dumps(remote_notebook, sort_keys=True, separators=(",", ":")).encode()
    )
    if notebook_canonical_sha256 != identity["notebook_canonical_json_sha256"]:
        raise RuntimeError("source identity remote notebook canonical bytes differ")
    return facts


def validate_whole_model_request_count(
    outer: dict[str, Any], total_actor_calls: int
) -> None:
    """Require one successful provider request per persisted Actor call."""
    if type(total_actor_calls) is not int or total_actor_calls <= 0:
        raise RuntimeError("whole-model Actor-call count differs")
    total_requests = total_actor_calls + 1
    expected = {
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
    if outer.get("request_count_facts") != expected:
        raise RuntimeError("whole-model request count differs")


def validate_run(
    outer: dict[str, Any], official: dict[str, Any], system_before: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    if (
        outer.get("schema") != EXPECTED_OUTER_SCHEMA
        or outer.get("mode") != EXPECTED_OUTER_MODE
        or outer.get("status") != "passed"
        or outer.get("request_path_status") not in (None, "passed")
    ):
        raise RuntimeError("outer continued-play result did not pass")
    lifecycle = outer.get("lifecycle")
    gate = outer.get("r8_l3b_gate")
    if not isinstance(lifecycle, dict) or lifecycle.get("status") != "passed":
        raise RuntimeError("outer continued-play lifecycle did not pass")
    if not isinstance(gate, dict) or not gate or not all(value is True for value in gate.values()):
        raise RuntimeError("outer R8-L3B conjunction did not pass")
    gpu = system_before.get("gpu")
    if (
        not isinstance(gpu, dict)
        or gpu.get("command") != EXPECTED_GPU_COMMAND
        or gpu.get("returncode") != 0
        or len(str(gpu.get("stdout", "")).strip().splitlines()) != 1
        or not str(gpu.get("stdout", "")).startswith(EXPECTED_GPU_MARKER)
        or len(str(gpu.get("stdout", "")).strip().split(",")) != 4
    ):
        raise RuntimeError("physical RTX PRO 6000 evidence is absent")
    if outer.get("official_launcher") != official:
        raise RuntimeError("outer embedded official launcher result differs")
    if (
        official.get("schema") != EXPECTED_OFFICIAL_SCHEMA
        or official.get("status") != "passed"
        or official.get("returncode") != 0
        or official.get("launcher_returncode") != 0
        or official.get("timed_out") is not False
        or official.get("residual_group_term_required") is not False
        or official.get("operation_mode") != "offline"
        or official.get("model") != EXPECTED_MODEL
    ):
        raise RuntimeError("official launcher result did not pass")
    if official.get("public_games") != list(EXPECTED_GAMES):
        raise RuntimeError("official public-three roster or order differs")

    progress = official.get("public_progress")
    sessions = official.get("solver_sessions")
    if not isinstance(progress, dict) or not isinstance(sessions, dict):
        raise TypeError("official progress or Solver reconciliation is absent")
    if progress.get("game_count") != len(EXPECTED_GAMES) or sessions.get(
        "game_count"
    ) != len(EXPECTED_GAMES):
        raise RuntimeError("continued-play exposure is not the exact public-three shape")

    progress_games = progress.get("games")
    session_games = sessions.get("games")
    if not isinstance(progress_games, dict) or set(progress_games) != set(EXPECTED_GAMES):
        raise RuntimeError("public progress game inventory differs")
    if not isinstance(session_games, dict) or set(session_games) != set(EXPECTED_GAMES):
        raise RuntimeError("Solver session game inventory differs")
    action_counts: dict[str, int] = {}
    actor_call_sum = 0
    for game in EXPECTED_GAMES:
        progress_game = progress_games[game]
        if not isinstance(progress_game, dict):
            raise TypeError(f"public progress differs for {game}")
        recorded = progress_games[game].get("recorded_action_results")
        if not isinstance(recorded, int) or not 1 <= recorded <= 65:
            raise RuntimeError(f"public action-result count differs for {game}")
        session = session_games[game]
        if not isinstance(session, dict):
            raise TypeError(f"Solver session differs for {game}")
        actions = session.get("actions")
        if not isinstance(actions, list) or not 1 <= len(actions) <= EXPECTED_ACTIONS_PER_GAME:
            raise RuntimeError(f"Solver action count differs for {game}")
        if len(actions) < EXPECTED_ACTIONS_PER_GAME and progress_game.get("final_state") != "WIN":
            raise RuntimeError(f"non-winning Solver session ended before its cap for {game}")
        if [action.get("sequence") for action in actions] != list(
            range(len(actions))
        ):
            raise RuntimeError(f"Solver action sequence differs for {game}")
        metadata = session.get("worker_metadata")
        if not isinstance(metadata, dict) or {
            key: metadata.get(key) for key in EXPECTED_WORKER_METADATA
        } != EXPECTED_WORKER_METADATA:
            raise RuntimeError(f"Solver execution envelope differs for {game}")
        model_evidence = session.get("model_evidence")
        if not isinstance(model_evidence, dict):
            raise TypeError(f"Solver model evidence differs for {game}")
        by_sequence = model_evidence.get("actor_calls_by_sequence")
        expected_keys = [str(index) for index in range(len(actions))]
        if (
            not isinstance(by_sequence, dict)
            or set(by_sequence) != set(expected_keys)
            or any(type(value) is not int for value in by_sequence.values())
            or not 1 <= by_sequence["0"] <= 6
            or any(by_sequence.get(key) != 1 for key in expected_keys[1:])
            or model_evidence.get("actor_call_count") != sum(by_sequence.values())
        ):
            raise RuntimeError(f"Actor-call reconciliation differs for {game}")
        actor_call_sum += model_evidence["actor_call_count"]
        action_counts[game] = len(actions)
    total_actions = sum(action_counts.values())
    if sessions.get("total_solver_actions") != total_actions:
        raise RuntimeError("continued-play Solver action total differs")
    if (
        sessions.get("total_actor_calls") != actor_call_sum
        or actor_call_sum > sum(6 + count - 1 for count in action_counts.values())
    ):
        raise RuntimeError("continued-play Actor-call count exceeds the frozen envelope")
    validate_whole_model_request_count(outer, actor_call_sum)
    return progress_games, session_games, action_counts


def parse_launcher_log(text: str) -> dict[str, dict[str, Any]]:
    facts = {
        game: {"created": None, "available": None, "events": []}
        for game in EXPECTED_GAMES
    }
    for line in text.splitlines():
        for pattern, kind in (
            (CREATED_RECORDING, "created"),
            (ACTION_EVENT, "event"),
            (RECORDING_AVAILABLE, "available"),
        ):
            match = pattern.match(line)
            if match is None:
                continue
            game = match.group("game")
            if game not in facts:
                raise RuntimeError(f"launcher stdout contains an unexpected game: {game}")
            timestamp = parse_log_timestamp(match.group("timestamp"))
            if kind == "event":
                facts[game]["events"].append(
                    {
                        "timestamp": timestamp,
                        "action_id": int(match.group("action_id")),
                        "count": int(match.group("count")),
                    }
                )
            else:
                if facts[game][kind] is not None:
                    raise RuntimeError(f"launcher stdout contains duplicate {kind}: {game}")
                facts[game][kind] = timestamp
            break
    for game, record in facts.items():
        if record["created"] is None or record["available"] is None:
            raise RuntimeError(f"launcher stdout lacks recording anchors for {game}")
        events = record["events"]
        if not events or any(
            left["timestamp"] >= right["timestamp"]
            for left, right in pairwise(events)
        ):
            raise RuntimeError(f"launcher ACTION events are absent or unordered for {game}")
        if [event["count"] for event in events] != list(range(len(events))):
            raise RuntimeError(f"launcher ACTION counts differ for {game}")
    return facts


def solver_action_id(action: dict[str, Any], game: str) -> int:
    arguments = action.get("validated_arguments")
    if not isinstance(arguments, dict):
        raise TypeError(f"Solver action arguments are absent for {game}")
    name = arguments.get("action")
    if not isinstance(name, str) or name.upper() not in ACTION_IDS:
        raise RuntimeError(f"Solver action identity differs for {game}: {arguments}")
    return ACTION_IDS[name.upper()]


def alignment_facts(
    solver_ids: list[int], events: list[dict[str, Any]], created: datetime, available: datetime
) -> dict[str, Any]:
    event_ids = [event["action_id"] for event in events]
    n = len(solver_ids)
    m = len(event_ids)
    prefix = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        prefix[0][j] = 1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prefix[i][j] = prefix[i][j - 1]
            if solver_ids[i - 1] == event_ids[j - 1]:
                prefix[i][j] += prefix[i - 1][j - 1]
    alignment_count = prefix[n][m]
    if alignment_count == 0:
        raise RuntimeError("Solver actions have no monotonic official ACTION alignment")

    suffix = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        suffix[n][j] = 1
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            suffix[i][j] = suffix[i][j + 1]
            if solver_ids[i] == event_ids[j]:
                suffix[i][j] += suffix[i + 1][j + 1]

    feasible: list[list[int]] = []
    for i, action_id in enumerate(solver_ids):
        positions = [
            j
            for j, event_id in enumerate(event_ids)
            if event_id == action_id and prefix[i][j] and suffix[i + 1][j + 1]
        ]
        if not positions:
            raise AssertionError("alignment feasibility calculation is inconsistent")
        feasible.append(positions)

    initial_seconds = max(
        (events[position]["timestamp"] - created).total_seconds()
        for position in feasible[0]
    )
    successor_seconds = 0.0
    for i in range(1, n):
        for before in feasible[i - 1]:
            for after in feasible[i]:
                if before < after:
                    successor_seconds = max(
                        successor_seconds,
                        (events[after]["timestamp"] - events[before]["timestamp"]).total_seconds(),
                    )
    if successor_seconds <= 0 and n > 1:
        raise RuntimeError("continued-play evidence has no successor interval")
    teardown_seconds = max(
        (available - events[position]["timestamp"]).total_seconds()
        for position in feasible[-1]
    )
    if min(initial_seconds, teardown_seconds) < 0:
        raise RuntimeError("aligned Solver actions fall outside recording anchors")
    return {
        "alignment_count": alignment_count,
        "official_event_count": m,
        "solver_action_count": n,
        "initial_max_seconds": initial_seconds,
        "successor_max_seconds": successor_seconds,
        "teardown_max_seconds": teardown_seconds,
        "feasible_event_positions": feasible,
    }


def build_schedule(
    harness: ModuleType, projection: dict[str, Any], usable_seconds: int
) -> dict[str, Any]:
    per_game: dict[str, int] = {}
    for game in harness.RUN_ORDER:
        cap = harness.FIRST_LEVEL_ACTION_CAPS[game]
        raw = (
            projection["initial_decision_seconds"]
            + (cap - 1) * projection["successor_decision_seconds"]
            + projection["per_game_overhead_seconds"]
        )
        child_seconds = max(
            61, math.ceil(raw * projection["safety_multiplier"])
        )
        per_game[game] = (
            child_seconds + projection["per_game_scheduler_reserve_seconds"]
        )
        if per_game[game] > usable_seconds:
            raise RuntimeError(
                f"projected shard cannot fit one generation: {game}={per_game[game]}"
            )
    generations: list[dict[str, Any]] = []
    current_games: list[str] = []
    current_total = 0
    for game in harness.RUN_ORDER:
        value = per_game[game]
        if current_games and current_total + value > usable_seconds:
            generations.append(
                {"games": current_games, "projected_seconds": current_total}
            )
            current_games = []
            current_total = 0
        current_games.append(game)
        current_total += value
    if current_games:
        generations.append({"games": current_games, "projected_seconds": current_total})
    return {
        "usable_seconds_per_generation": usable_seconds,
        "per_game_projected_seconds": per_game,
        "generations": generations,
        "generation_count": len(generations),
        "aggregate_projected_shard_seconds": sum(per_game.values()),
    }


def derive(output_root: Path, policy_path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    if not output_root.is_dir() or output_root.is_symlink():
        raise RuntimeError(f"Kaggle output root is not one regular directory: {output_root}")
    outer_path = unique_file(output_root, "result.json", "outer result")
    official_path = unique_file(
        output_root, "official-launcher-result.json", "official launcher result"
    )
    system_path = unique_file(output_root, "system-before.json", "system preflight")
    reconciliation_path = unique_file(
        output_root, "run-reconciliation.json", "canonical run reconciliation"
    )
    outer = read_json(outer_path, "outer result")
    official = read_json(official_path, "official launcher result")
    system_before = read_json(system_path, "system preflight")
    reconciliation = read_json(reconciliation_path, "canonical run reconciliation")
    policy, policy_facts = load_policy(policy_path)
    identity_files = source_identity_file_facts(output_root, policy)
    validate_reconciliation(reconciliation, policy)
    progress_games, session_games, action_counts = validate_run(
        outer, official, system_before
    )

    embedded_stdout = official.get("stdout")
    if not isinstance(embedded_stdout, dict):
        raise TypeError("official launcher stdout facts are absent")
    stdout_path = unique_file(
        output_root, Path(str(embedded_stdout.get("path"))).name, "official launcher stdout"
    )
    stdout_facts = file_facts(stdout_path, output_root)
    if (
        embedded_stdout.get("bytes") != stdout_facts["bytes"]
        or embedded_stdout.get("sha256") != stdout_facts["sha256"]
    ):
        raise RuntimeError("official launcher stdout bytes differ from the result")
    log = parse_launcher_log(stdout_path.read_text(errors="strict"))

    outer_started = parse_timestamp(outer.get("started_at"), "outer start")
    outer_ended = parse_timestamp(outer.get("ended_at"), "outer end")
    system_captured = parse_timestamp(
        system_before.get("captured_at"), "system preflight capture"
    )
    official_started = parse_timestamp(official.get("started_at"), "official start")
    official_ended = parse_timestamp(official.get("ended_at"), "official end")
    if not outer_started <= system_captured <= official_started < official_ended <= outer_ended:
        raise RuntimeError("outer, preflight, and official chronology differs")
    per_game_alignment: dict[str, dict[str, Any]] = {}
    for game in EXPECTED_GAMES:
        if len(log[game]["events"]) != progress_games[game]["recorded_action_results"]:
            raise RuntimeError(f"official ACTION event count differs for {game}")
        events = log[game]["events"]
        if not (
            official_started
            <= log[game]["created"]
            <= events[0]["timestamp"]
            <= events[-1]["timestamp"]
            <= log[game]["available"]
            <= official_ended
        ):
            raise RuntimeError(f"official launcher chronology differs for {game}")
        solver_ids = [solver_action_id(action, game) for action in session_games[game]["actions"]]
        facts = alignment_facts(
            solver_ids,
            log[game]["events"],
            log[game]["created"],
            log[game]["available"],
        )
        per_game_alignment[game] = {
            **facts,
            "created_at": log[game]["created"].isoformat(),
            "available_at": log[game]["available"].isoformat(),
            "official_events": [
                {
                    **event,
                    "timestamp": event["timestamp"].isoformat(),
                }
                for event in log[game]["events"]
            ],
        }

    initial_raw = max(record["initial_max_seconds"] for record in per_game_alignment.values())
    successor_raw = max(
        record["successor_max_seconds"] for record in per_game_alignment.values()
    )
    if successor_raw <= 0:
        raise RuntimeError("continued-play evidence has no successor sample")
    setup_raw = max(
        (log[game]["created"] - official_started).total_seconds()
        for game in EXPECTED_GAMES
    )
    teardown_raw = max(
        record["teardown_max_seconds"] for record in per_game_alignment.values()
    )
    if setup_raw <= 0:
        raise RuntimeError("official setup bound is not positive")
    measured_overhead = math.ceil(setup_raw + teardown_raw)

    source_manifest = {
        "outer_result": file_facts(outer_path, output_root),
        "official_launcher_result": file_facts(official_path, output_root),
        "official_launcher_stdout": stdout_facts,
        "system_preflight": file_facts(system_path, output_root),
        **identity_files,
    }
    source_manifest_sha256 = sha256_bytes(canonical_json(source_manifest))
    if reconciliation.get("source_artifact_manifest_sha256") != source_manifest_sha256:
        raise RuntimeError("canonical run source artifact manifest differs")
    projection = {
        "initial_decision_seconds": math.ceil(initial_raw),
        "successor_decision_seconds": math.ceil(successor_raw),
        "per_game_overhead_seconds": measured_overhead,
        "per_game_scheduler_reserve_seconds": policy[
            "per_game_scheduler_reserve_seconds"
        ],
        "safety_multiplier": policy["safety_multiplier"],
        "source_result_sha256": source_manifest["official_launcher_result"]["sha256"],
        "estimator_schema": PROJECTION_SCHEMA,
        "projection_policy_sha256": policy_facts["sha256"],
        "source_artifact_manifest_sha256": source_manifest_sha256,
    }
    harness = load_harness()
    if (
        sum(harness.FIRST_LEVEL_ACTION_CAPS.values()) != EXPECTED_DECISION_CEILING
        or EXPECTED_DECISION_CEILING + 5 * len(harness.PUBLIC_GAMES)
        != EXPECTED_REQUEST_CEILING
    ):
        raise RuntimeError("public-25 decision or request ceiling differs")
    schedule = build_schedule(
        harness, projection, policy["usable_generation_seconds"]
    )
    return {
        "schema": PROJECTION_SCHEMA,
        "method": {
            "admission": "C1_whole_decision",
            "alignment": "monotonic_subsequence_all_ambiguities_maximized",
            "initial_statistic": "ceil_max_creation_to_first_solver_action",
            "successor_statistic": "ceil_max_same_game_solver_action_interval",
            "measured_overhead_statistic": "ceil_max_setup_plus_max_teardown",
            "scheduler_reserve_kind": "fixed_non_measured_policy",
        },
        "runtime_projection": projection,
        "observations": {
            "per_game_alignment": per_game_alignment,
            "initial_raw_max_seconds": initial_raw,
            "successor_raw_max_seconds": successor_raw,
            "setup_raw_max_seconds": setup_raw,
            "teardown_raw_max_seconds": teardown_raw,
            "measured_overhead_ceiling_seconds": measured_overhead,
            "successor_interval_count": (
                sum(count - 1 for count in action_counts.values())
            ),
        },
        "policy": {"values": policy, "file": policy_facts},
        "source_artifacts": {
            "files": source_manifest,
            "manifest_sha256": source_manifest_sha256,
            "run_reconciliation": file_facts(reconciliation_path, output_root),
        },
        "harness": file_facts(HARNESS_PATH, SCRIPT_ROOT),
        "cohort": {
            "solver_decision_ceiling": EXPECTED_DECISION_CEILING,
            "whole_model_request_ceiling": EXPECTED_REQUEST_CEILING,
        },
        "canonical_run": {
            "kernel": reconciliation["kernel"],
            "candidate_identity": reconciliation["candidate_identity"],
            "candidate_artifact_manifest_sha256": reconciliation[
                "candidate_artifact_manifest_sha256"
            ],
        },
        "schedule": schedule,
        "outer_model_startup_and_teardown": "budget_separately_per_generation",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path, help="downloaded Kaggle kernel output")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = derive(args.output_root.resolve(), args.policy.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
