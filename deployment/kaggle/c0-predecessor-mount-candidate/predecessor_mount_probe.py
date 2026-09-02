#!/usr/bin/env python3
"""Qualify exact-version Kaggle notebook-output mounting without ARC activity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESULT_SCHEMA = "tycho-nooa-predecessor-mount-qualification-v1"
FIXTURE_SCHEMA = "tycho-nooa-predecessor-output-checkpoint-fixture-v1"


class MountFailure(RuntimeError):
    """A classified failure at the predecessor mount boundary."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def path_has_symlink(root: Path, path: Path) -> bool:
    root = root.absolute()
    path = path.absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise MountFailure(
            "nonregular_predecessor_mount", f"mounted path escapes input root: {path}"
        ) from error
    current = root
    if current.is_symlink():
        return True
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def regular_file_manifest(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise MountFailure(
            "nonregular_predecessor_mount", f"manifest root is not one regular directory: {root}"
        )
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise MountFailure(
                "nonregular_predecessor_mount", f"manifest contains a symlink: {path}"
            )
        if path.is_file():
            value = path.read_bytes()
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": len(value),
                    "sha256": sha256_bytes(value),
                }
            )
        elif not path.is_dir():
            raise MountFailure(
                "nonregular_predecessor_mount", f"manifest contains a non-file entry: {path}"
            )
    return records


def validate_fixture(fixture: dict[str, Any]) -> None:
    if fixture.get("schema") != FIXTURE_SCHEMA:
        raise MountFailure("fixture_invalid", "checkpoint fixture schema differs")
    predecessor = fixture.get("predecessor")
    if predecessor != {
        "arc_actions": 0,
        "competition_sources": 0,
        "kernel_id_no": 132802830,
        "kernel_slug": "pranavagrawai/prototype-tycho-nooa-successor-concurrency",
        "kernel_source": "pranavagrawai/prototype-tycho-nooa-successor-concurrency/1",
        "status": "ERROR",
        "version": 1,
    }:
        raise MountFailure("fixture_invalid", "predecessor identity differs")
    files = fixture.get("files")
    if not isinstance(files, list) or len(files) != 3:
        raise MountFailure("fixture_invalid", "selected predecessor file count differs")
    paths = [record.get("path") for record in files]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise MountFailure("fixture_invalid", "selected file paths are not unique and sorted")
    if any(
        not isinstance(record.get("bytes"), int)
        or record["bytes"] < 0
        or not isinstance(record.get("sha256"), str)
        or len(record["sha256"]) != 64
        for record in files
    ):
        raise MountFailure("fixture_invalid", "selected file manifest is malformed")
    if sha256_bytes(canonical_json(files)) != fixture.get("source_manifest_sha256"):
        raise MountFailure("fixture_invalid", "selected file manifest hash differs")
    negative = fixture.get("negative_control")
    if (
        not isinstance(negative, dict)
        or negative.get("latest_version") != 2
        or negative.get("result_path") != "tycho-nooa-successor-concurrency/result.json"
        or not isinstance(negative.get("result_sha256"), str)
        or len(negative["result_sha256"]) != 64
    ):
        raise MountFailure("fixture_invalid", "latest-version negative control differs")
    result_record = next(
        (record for record in files if record["path"] == negative["result_path"]), None
    )
    if result_record is None or result_record["sha256"] == negative["result_sha256"]:
        raise MountFailure("fixture_invalid", "version controls are not distinguishable")
    if fixture.get("continuation") != {
        "arc_actions": 0,
        "model_requests": 0,
        "operation": "copy_verified_predecessor_files_only",
        "submissions": 0,
    }:
        raise MountFailure("fixture_invalid", "continuation contract differs")


def candidate_root_for_result(input_root: Path, result_path: Path, relative: Path) -> Path:
    candidate = result_path
    for _ in relative.parts:
        candidate = candidate.parent
    if path_has_symlink(input_root, result_path):
        raise MountFailure(
            "nonregular_predecessor_mount", f"mounted result traverses a symlink: {result_path}"
        )
    return candidate


def locate_version_one_mount(
    input_root: Path, fixture: dict[str, Any]
) -> tuple[Path, list[dict[str, Any]]]:
    if not input_root.is_dir() or input_root.is_symlink():
        raise MountFailure("predecessor_not_mounted", "Kaggle input root is absent")
    result_relative = Path(fixture["negative_control"]["result_path"])
    version_one_hash = next(
        record["sha256"]
        for record in fixture["files"]
        if record["path"] == result_relative.as_posix()
    )
    latest_hash = fixture["negative_control"]["result_sha256"]
    matches: list[Path] = []
    latest_matches: list[Path] = []
    mismatches: list[Path] = []
    for result_path in sorted(input_root.rglob(result_relative.name)):
        if not result_path.as_posix().endswith(result_relative.as_posix()):
            continue
        if not result_path.is_file():
            continue
        candidate = candidate_root_for_result(input_root, result_path, result_relative)
        digest = sha256_bytes(result_path.read_bytes())
        if digest == version_one_hash:
            matches.append(candidate)
        elif digest == latest_hash:
            latest_matches.append(candidate)
        else:
            mismatches.append(candidate)
    if matches and (latest_matches or mismatches):
        raise MountFailure(
            "ambiguous_predecessor_mount",
            "input contains exact version 1 bytes plus another predecessor-shaped result",
        )
    if latest_matches and not matches:
        raise MountFailure(
            "latest_version_mounted",
            "Kaggle served current version 2 bytes instead of explicit predecessor version 1",
        )
    if not matches:
        classification = "predecessor_bytes_differ" if mismatches else "predecessor_not_mounted"
        raise MountFailure(classification, "exact predecessor version 1 result is absent")
    if len(matches) != 1:
        raise MountFailure(
            "ambiguous_predecessor_mount",
            f"expected one exact predecessor mount, found {len(matches)}",
        )
    root = matches[0]
    actual: list[dict[str, Any]] = []
    for record in fixture["files"]:
        path = root / record["path"]
        if path_has_symlink(input_root, path):
            raise MountFailure(
                "nonregular_predecessor_mount", f"selected mounted file traverses a symlink: {path}"
            )
        if not path.is_file():
            raise MountFailure(
                "predecessor_bytes_differ", f"selected predecessor file is absent: {record['path']}"
            )
        value = path.read_bytes()
        actual.append(
            {"path": record["path"], "bytes": len(value), "sha256": sha256_bytes(value)}
        )
    if actual != fixture["files"]:
        raise MountFailure(
            "predecessor_bytes_differ", "selected predecessor file bytes differ from manifest"
        )
    return root, actual


def copy_verified_predecessor(
    source_root: Path,
    output_root: Path,
    source_manifest: list[dict[str, Any]],
) -> tuple[Path, list[dict[str, Any]]]:
    destination = output_root / "verified-predecessor"
    staging = output_root / ".verified-predecessor.staging"
    if destination.exists() or staging.exists():
        raise MountFailure("output_not_clean", "predecessor import destination already exists")
    staging.mkdir(parents=True)
    try:
        for record in source_manifest:
            target = staging / record["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source_root / record["path"]).read_bytes())
        imported_manifest = regular_file_manifest(staging)
        if imported_manifest != source_manifest:
            raise MountFailure("copy_verification_failed", "copied predecessor bytes differ")
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination, imported_manifest


def run(input_root: Path, output_root: Path, fixture: dict[str, Any]) -> int:
    started_at = utc_now()
    output_root.mkdir(parents=True, exist_ok=True)
    published_destination = False
    marker_written = False
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "failed",
        "started_at": started_at,
        "continued_after_verification": False,
        "arc_actions": 0,
        "model_requests": 0,
        "competition_sources": 0,
        "submissions": 0,
        "claim_ceiling": (
            "This qualifies exact-version predecessor-output mounting and hash-bound copying "
            "only. It runs no ARC environment and does not qualify output persistence after "
            "timeout, real all-25 checkpoint import, gameplay, scoring, or submission."
        ),
    }
    try:
        validate_fixture(fixture)
        source_root, source_manifest = locate_version_one_mount(input_root, fixture)
        _, imported_manifest = copy_verified_predecessor(
            source_root, output_root, source_manifest
        )
        published_destination = True
        marker = {
            "schema": "tycho-nooa-predecessor-continuation-marker-v1",
            "verified_before_continuation": True,
            "expected_kernel_source": fixture["predecessor"]["kernel_source"],
            "source_manifest_sha256": fixture["source_manifest_sha256"],
            "imported_manifest_sha256": sha256_bytes(canonical_json(imported_manifest)),
            "next_operation": "stop_after_qualification",
            "arc_actions": 0,
        }
        write_json_atomic(output_root / "continuation-marker.json", marker)
        marker_written = True
        expected = fixture["predecessor"]
        result.update(
            {
                "status": "passed",
                "failure_classification": None,
                "expected_predecessor": {
                    "kernel_source": expected["kernel_source"],
                    "kernel_id_no": expected["kernel_id_no"],
                    "version": expected["version"],
                },
                "negative_control_latest_version": fixture["negative_control"][
                    "latest_version"
                ],
                "mount_root_relative": source_root.relative_to(input_root).as_posix(),
                "source_manifest_sha256": fixture["source_manifest_sha256"],
                "imported_manifest_sha256": marker["imported_manifest_sha256"],
                "selected_file_count": len(source_manifest),
                "selected_bytes": sum(record["bytes"] for record in source_manifest),
                "continued_after_verification": True,
            }
        )
        returncode = 0
    except MountFailure as error:
        if published_destination:
            shutil.rmtree(output_root / "verified-predecessor", ignore_errors=True)
        if marker_written:
            (output_root / "continuation-marker.json").unlink(missing_ok=True)
        result.update(
            {
                "failure_classification": error.classification,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        returncode = 74
    # Preserve a machine-readable failure even when a provider filesystem error
    # falls outside the classifications above.  It never permits continuation.
    except Exception as error:  # noqa: BLE001
        if published_destination:
            shutil.rmtree(output_root / "verified-predecessor", ignore_errors=True)
        if marker_written:
            (output_root / "continuation-marker.json").unlink(missing_ok=True)
        result.update(
            {
                "failure_classification": "unclassified_failure",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        returncode = 74
    finally:
        result["ended_at"] = utc_now()
        write_json_atomic(output_root / "result.json", result)
    return returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixture = json.loads(args.fixture.read_text())
    return run(args.input_root, args.output_root, fixture)


if __name__ == "__main__":
    raise SystemExit(main())
