"""Credential-safe hashing and structural validation for immutable run evidence."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


MANDATORY_EVIDENCE = ("trace.jsonl", "process.jsonl", "scorecard.json")


def _canonical_node_bytes(path: Path) -> bytes:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        value = {"kind": "symlink", "target": os.readlink(path)}
        return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    if stat.S_ISREG(mode):
        return path.read_bytes()
    kind = (
        "fifo" if stat.S_ISFIFO(mode)
        else "socket" if stat.S_ISSOCK(mode)
        else "block" if stat.S_ISBLK(mode)
        else "character" if stat.S_ISCHR(mode)
        else "unknown"
    )
    value = {"kind": "special", "type": kind}
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _workspace_nodes(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    nodes: list[Path] = []
    for directory, names, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in sorted((*names, *filenames)):
            path = parent / name
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
                continue
            nodes.append(path)
    return nodes


def evidence_hashes(run_dir: Path) -> dict[str, str]:
    """Hash evidence nodes without following contestant-created links or special files."""

    paths = [run_dir / relative for relative in MANDATORY_EVIDENCE]
    payload_dir = run_dir / "request-payloads"
    if payload_dir.is_dir():
        paths.extend(sorted(payload_dir.glob("*.json")))
    response_dir = run_dir / "response-bodies"
    if response_dir.is_dir():
        paths.extend(sorted(response_dir.glob("*")))
    paths.extend(_workspace_nodes(run_dir / "workspace"))
    return {
        path.relative_to(run_dir).as_posix(): sha256(_canonical_node_bytes(path)).hexdigest()
        for path in sorted(paths)
        if path.exists() or path.is_symlink()
    }


def _json_lines_valid(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return bool(lines) and all(isinstance(json.loads(line), Mapping) for line in lines)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def validate_run_evidence(run_dir: Path, committed: Mapping[str, Any]) -> bool:
    """Recompute every evidence hash and require readable mandatory/raw JSON evidence."""

    if not all(isinstance(key, str) and isinstance(value, str) for key, value in committed.items()):
        return False
    actual = evidence_hashes(run_dir)
    if dict(committed) != actual or not all(name in actual for name in MANDATORY_EVIDENCE):
        return False
    if not _json_lines_valid(run_dir / "trace.jsonl"):
        return False
    if not _json_lines_valid(run_dir / "process.jsonl"):
        return False
    try:
        if not isinstance(json.loads((run_dir / "scorecard.json").read_text(encoding="utf-8")), (dict, list)):
            return False
        for path in sorted((run_dir / "request-payloads").glob("*.json")):
            if not isinstance(json.loads(path.read_text(encoding="utf-8")), Mapping):
                return False
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return _model_io_archives_valid(run_dir, committed)


def _model_io_archives_valid(run_dir: Path, committed: Mapping[str, Any]) -> bool:
    try:
        records = [
            json.loads(line)
            for line in (run_dir / "process.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    for record in records:
        if not isinstance(record, Mapping):
            return False
        event_type = record.get("type")
        if event_type not in {"model_response", "model_request_failed"}:
            continue
        request_required = isinstance(record.get("request_payload_sha256"), str)
        if not _archive_valid(
            run_dir,
            committed,
            record,
            archive_field="request_payload_archive",
            archive_hash_field="request_payload_archive_sha256",
            commitment_field="request_payload_sha256",
            required=request_required,
        ):
            return False
        response_required = event_type == "model_response" or isinstance(
            record.get("response_body_sha256"), str
        )
        response_bytes = _archive_bytes(
            run_dir,
            committed,
            record,
            archive_field="response_body_archive",
            archive_hash_field="response_body_archive_sha256",
            commitment_field="response_body_sha256",
            required=response_required,
        )
        if response_required and response_bytes is None:
            return False
        if event_type == "model_response":
            if response_bytes is None:
                return False
            if record.get("response_body_encoding") not in {
                "exact-transport-bytes-v1",
                "utf-8-exact-transport-v1",
            }:
                return False
            try:
                parsed_response = json.loads(response_bytes)
            except (TypeError, UnicodeError, json.JSONDecodeError):
                return False
            if parsed_response != record.get("response_body"):
                return False
    return True


def _archive_valid(
    run_dir: Path,
    committed: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    archive_field: str,
    archive_hash_field: str,
    commitment_field: str,
    required: bool,
) -> bool:
    return _archive_bytes(
        run_dir,
        committed,
        record,
        archive_field=archive_field,
        archive_hash_field=archive_hash_field,
        commitment_field=commitment_field,
        required=required,
    ) is not None or not required


def _archive_bytes(
    run_dir: Path,
    committed: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    archive_field: str,
    archive_hash_field: str,
    commitment_field: str,
    required: bool,
) -> bytes | None:
    relative = record.get(archive_field)
    archive_hash = record.get(archive_hash_field)
    commitment = record.get(commitment_field)
    if not required and relative is None:
        return None
    if (
        not isinstance(relative, str)
        or not relative
        or not isinstance(archive_hash, str)
        or not archive_hash
        or not isinstance(commitment, str)
        or not commitment
    ):
        return None
    relative_path = Path(relative)
    path = run_dir / relative_path
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative not in committed
        or not path.is_file()
        or path.is_symlink()
    ):
        return None
    payload = path.read_bytes()
    actual = sha256(payload).hexdigest()
    if (
        actual != archive_hash
        or actual != commitment
        or committed.get(relative) != actual
    ):
        return None
    return payload
