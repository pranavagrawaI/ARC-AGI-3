#!/usr/bin/env python3
"""Export and import one immutable public-25 checkpoint dataset.

The transport manifest is the publication marker.  Export writes it only after
the checkpoint payload and Kaggle control metadata are complete.  A successor
accepts no mounted bytes until the manifest, the exact payload inventory, and
the checkpoint's own manifests all agree.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import resumable_public25 as checkpoint_runtime

TRANSPORT_SCHEMA = "tycho-nooa-public25-checkpoint-transport-v1"
TRANSPORT_MANIFEST = "transport-manifest.json"
DATASET_METADATA = "dataset-metadata.json"
CHECKPOINT_DIRECTORY = "checkpoint"
CHECKPOINT_ARCHIVE = "checkpoint.tar"
HEX64 = re.compile(r"[0-9a-f]{64}")
OWNER = re.compile(r"[a-z0-9][a-z0-9_-]{1,38}")
SLUG_PREFIX = re.compile(r"[a-z0-9][a-z0-9-]{2,31}")
DATASET_SLUG = re.compile(r"[a-z0-9][a-z0-9-]{5,49}")
MODULE_PATH = Path(__file__).resolve()
CHECKPOINT_RUNTIME_PATH = MODULE_PATH.with_name("resumable_public25.py")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def require_hex64(value: str, label: str) -> None:
    if HEX64.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not one SHA-256 digest")


def validate_file_records(
    records: list[dict[str, Any]], *, label: str, required_prefix: str | None
) -> None:
    paths: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise TypeError(f"{label} contains a malformed byte record")
        path = record["path"]
        byte_count = record["bytes"]
        digest = record["sha256"]
        if not isinstance(path, str) or PurePosixPath(path).as_posix() != path:
            raise RuntimeError(f"{label} contains a non-canonical path")
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise RuntimeError(f"{label} contains a path outside its root")
        if required_prefix is not None and not path.startswith(f"{required_prefix}/"):
            raise RuntimeError(f"{label} contains a path outside {required_prefix}")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise TypeError(f"{label} contains an invalid byte count")
        if not isinstance(digest, str):
            raise TypeError(f"{label} contains a non-string SHA-256 digest")
        require_hex64(digest, f"{label} record hash")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise RuntimeError(f"{label} paths are not unique and sorted")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(), object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not readable JSON: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} is not one JSON object")
    return value


def assert_local_runtime_binding() -> None:
    imported = Path(checkpoint_runtime.__file__).resolve()
    if imported != CHECKPOINT_RUNTIME_PATH:
        raise RuntimeError(
            "checkpoint verifier import is not the colocated resumable_public25.py"
        )


def regular_file_manifest(root: Path, *, prefix: str = "") -> list[dict[str, Any]]:
    records = checkpoint_runtime.regular_file_manifest(root)
    if not prefix:
        return records
    return [{**record, "path": f"{prefix}/{record['path']}"} for record in records]


def write_regular_bytes(destination: Path, value: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def checkpoint_required_paths(manifest: dict[str, Any]) -> list[str]:
    paths = ["checkpoint.json"]
    completed = manifest.get("completed_games")
    failures = manifest.get("failures")
    if not isinstance(completed, list) or not isinstance(failures, list):
        raise TypeError("checkpoint completed games or failures are malformed")
    failure_games: list[str] = []
    for failure in failures:
        if not isinstance(failure, dict) or not isinstance(failure.get("game"), str):
            raise TypeError("checkpoint failure record is malformed")
        failure_games.append(failure["game"])
        files = failure.get("files")
        if not isinstance(files, list):
            raise TypeError("checkpoint failure byte manifest is malformed")
        for record in files:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise TypeError("checkpoint failure byte record is malformed")
            paths.append(f"failures/{failure['game']}/{record['path']}")
    if len(failure_games) != len(set(failure_games)):
        raise RuntimeError("checkpoint failure games are not unique")
    if set(failure_games) & set(completed):
        raise RuntimeError("checkpoint names one game as both failed and completed")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise TypeError("checkpoint game byte manifest is malformed")
    for record in files:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise TypeError("checkpoint game byte record is malformed")
        paths.append(f"games/{record['path']}")
    if len(paths) != len(set(paths)):
        raise RuntimeError("checkpoint required artifact paths are not unique")
    if paths[1:] != sorted(paths[1:]):
        raise RuntimeError("checkpoint required artifact paths are not sorted")
    return paths


def validate_checkpoint_identity(
    checkpoint_root: Path,
    *,
    expected_contract_sha256: str,
    expected_checkpoint_manifest_sha256: str,
) -> dict[str, Any]:
    require_hex64(expected_contract_sha256, "expected contract hash")
    require_hex64(
        expected_checkpoint_manifest_sha256, "expected checkpoint manifest hash"
    )
    assert_local_runtime_binding()
    manifest = checkpoint_runtime.verify_checkpoint(
        checkpoint_root, expected_contract_sha256
    )
    actual_hash = sha256_file(checkpoint_root / "checkpoint.json")
    if actual_hash != expected_checkpoint_manifest_sha256:
        raise RuntimeError("checkpoint manifest hash differs")
    generation = manifest.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise RuntimeError("checkpoint generation is not a nonnegative integer")
    chain_predecessor = manifest.get("predecessor_manifest_sha256")
    if chain_predecessor is not None:
        if not isinstance(chain_predecessor, str):
            raise RuntimeError("checkpoint chain predecessor is malformed")
        require_hex64(chain_predecessor, "checkpoint chain predecessor")
    checkpoint_required_paths(manifest)
    return manifest


def required_file_manifest(
    source_root: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in checkpoint_required_paths(manifest):
        source = source_root / relative
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"required checkpoint artifact is not regular: {source}")
        value = source.read_bytes()
        records.append(
            {
                "path": f"{CHECKPOINT_DIRECTORY}/{relative}",
                "bytes": len(value),
                "sha256": sha256_bytes(value),
            }
        )
    return records


def write_checkpoint_archive(
    source_root: Path,
    archive_path: Path,
    payload_files: list[dict[str, Any]],
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("xb") as raw:
        with tarfile.open(
            fileobj=raw, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            for record in payload_files:
                name = record["path"]
                relative = name.removeprefix(f"{CHECKPOINT_DIRECTORY}/")
                value = (source_root / relative).read_bytes()
                info = tarfile.TarInfo(name=name)
                info.size = len(value)
                info.mode = 0o444
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                archive.addfile(info, io.BytesIO(value))
        raw.flush()
        os.fsync(raw.fileno())


def extract_checkpoint_archive(
    archive_path: Path,
    destination_root: Path,
    payload_files: list[dict[str, Any]],
) -> None:
    if not archive_path.is_file() or archive_path.is_symlink():
        raise RuntimeError("checkpoint archive is absent or a symlink")
    validate_file_records(
        payload_files,
        label="checkpoint archive payload",
        required_prefix=CHECKPOINT_DIRECTORY,
    )
    expected = {record["path"]: record for record in payload_files}
    destination_root.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if names != list(expected):
                raise RuntimeError("checkpoint archive member inventory differs")
            for member in members:
                if not member.isfile() or member.issym() or member.islnk():
                    raise RuntimeError("checkpoint archive contains a non-regular member")
                record = expected[member.name]
                if member.size != record["bytes"]:
                    raise RuntimeError("checkpoint archive member size differs")
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError("checkpoint archive member is unreadable")
                value = stream.read()
                if len(value) != record["bytes"] or sha256_bytes(value) != record["sha256"]:
                    raise RuntimeError("checkpoint archive member bytes differ")
                relative = member.name.removeprefix(f"{CHECKPOINT_DIRECTORY}/")
                write_regular_bytes(destination_root / relative, value)
    except (tarfile.TarError, OSError) as error:
        raise RuntimeError(f"checkpoint archive is unreadable: {error}") from error
    actual = regular_file_manifest(destination_root, prefix=CHECKPOINT_DIRECTORY)
    if actual != payload_files:
        raise RuntimeError("extracted checkpoint inventory or bytes differ")


def publication_seed(
    *,
    contract_sha256: str,
    checkpoint_manifest_sha256: str,
    checkpoint_generation: int,
    chain_predecessor_manifest_sha256: str | None,
    completed_games: list[str],
    payload_files: list[dict[str, Any]],
    checkpoint_archive: dict[str, Any],
    checkpoint_verifier_sha256: str,
    transport_implementation_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": TRANSPORT_SCHEMA,
        "contract_sha256": contract_sha256,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "checkpoint_generation": checkpoint_generation,
        "chain_predecessor_manifest_sha256": chain_predecessor_manifest_sha256,
        "completed_games": completed_games,
        "payload_files": payload_files,
        "checkpoint_archive": checkpoint_archive,
        "checkpoint_verifier_sha256": checkpoint_verifier_sha256,
        "transport_implementation_sha256": transport_implementation_sha256,
    }


def dataset_identity(
    owner: str, slug_prefix: str, generation: int, publication_identity: str
) -> tuple[str, str]:
    if OWNER.fullmatch(owner) is None:
        raise RuntimeError("Kaggle dataset owner is malformed")
    if SLUG_PREFIX.fullmatch(slug_prefix) is None:
        raise RuntimeError("Kaggle dataset slug prefix is malformed")
    slug = f"{slug_prefix}-g{generation}-{publication_identity[:20]}"
    if len(slug) > 50:
        raise RuntimeError("content-addressed Kaggle dataset slug exceeds 50 characters")
    dataset_id = f"{owner}/{slug}"
    title = f"Tycho NOOA public25 checkpoint g{generation} {publication_identity[:12]}"
    if len(title) > 50:
        raise RuntimeError("content-addressed Kaggle dataset title exceeds 50 characters")
    return dataset_id, title


def metadata_for(dataset_id: str, title: str) -> dict[str, Any]:
    return {
        "id": dataset_id,
        "licenses": [{"name": "other"}],
        "title": title,
    }


def export_checkpoint_dataset(
    checkpoint_root: Path,
    staging_root: Path,
    *,
    expected_contract_sha256: str,
    expected_checkpoint_manifest_sha256: str,
    owner: str,
    slug_prefix: str = "tycho-nooa-p25-ckpt",
) -> dict[str, Any]:
    """Build a fresh upload directory and publish its transport marker last."""
    if staging_root.exists() or staging_root.is_symlink():
        raise RuntimeError(f"transport staging destination already exists: {staging_root}")
    manifest = validate_checkpoint_identity(
        checkpoint_root,
        expected_contract_sha256=expected_contract_sha256,
        expected_checkpoint_manifest_sha256=expected_checkpoint_manifest_sha256,
    )
    staging_root.mkdir(parents=True)
    payload_files = required_file_manifest(checkpoint_root, manifest)
    validate_file_records(
        payload_files,
        label="checkpoint transport payload",
        required_prefix=CHECKPOINT_DIRECTORY,
    )
    write_checkpoint_archive(
        checkpoint_root, staging_root / CHECKPOINT_ARCHIVE, payload_files
    )
    archive_record = regular_file_manifest(staging_root)[0]
    if archive_record["path"] != CHECKPOINT_ARCHIVE:
        raise AssertionError("checkpoint archive was not staged as one regular file")
    checkpoint_verifier_sha256 = sha256_file(CHECKPOINT_RUNTIME_PATH)
    transport_implementation_sha256 = sha256_file(MODULE_PATH)
    seed = publication_seed(
        contract_sha256=expected_contract_sha256,
        checkpoint_manifest_sha256=expected_checkpoint_manifest_sha256,
        checkpoint_generation=manifest["generation"],
        chain_predecessor_manifest_sha256=manifest["predecessor_manifest_sha256"],
        completed_games=manifest["completed_games"],
        payload_files=payload_files,
        checkpoint_archive=archive_record,
        checkpoint_verifier_sha256=checkpoint_verifier_sha256,
        transport_implementation_sha256=transport_implementation_sha256,
    )
    publication_identity = sha256_bytes(canonical_json(seed))
    dataset_id, dataset_title = dataset_identity(
        owner, slug_prefix, manifest["generation"], publication_identity
    )
    metadata = metadata_for(dataset_id, dataset_title)
    write_json_atomic(staging_root / DATASET_METADATA, metadata)
    metadata_record = next(
        record
        for record in regular_file_manifest(staging_root)
        if record["path"] == DATASET_METADATA
    )
    transport = {
        **seed,
        "publication_identity": publication_identity,
        "dataset_id": dataset_id,
        "dataset_title": dataset_title,
        "intended_visibility": "private",
        "payload_bytes": sum(record["bytes"] for record in payload_files),
        "control_metadata": metadata_record,
    }
    # This is the commit marker.  No verifier accepts the staging tree before
    # this atomic write succeeds.
    write_json_atomic(staging_root / TRANSPORT_MANIFEST, transport)
    transport_hash = sha256_file(staging_root / TRANSPORT_MANIFEST)
    verified = verify_transport_root(
        staging_root,
        expected_transport_manifest_sha256=transport_hash,
        expected_contract_sha256=expected_contract_sha256,
        expected_predecessor_manifest_sha256=expected_checkpoint_manifest_sha256,
        expected_dataset_id=dataset_id,
        require_control_metadata=True,
    )
    return verified


def transport_manifest_keys() -> set[str]:
    return {
        "schema",
        "contract_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_generation",
        "chain_predecessor_manifest_sha256",
        "completed_games",
        "payload_files",
        "checkpoint_archive",
        "checkpoint_verifier_sha256",
        "transport_implementation_sha256",
        "publication_identity",
        "dataset_id",
        "dataset_title",
        "intended_visibility",
        "payload_bytes",
        "control_metadata",
    }


def verify_transport_root(
    root: Path,
    *,
    expected_transport_manifest_sha256: str,
    expected_contract_sha256: str,
    expected_predecessor_manifest_sha256: str,
    expected_dataset_id: str,
    require_control_metadata: bool,
) -> dict[str, Any]:
    for value, label in (
        (expected_transport_manifest_sha256, "expected transport manifest hash"),
        (expected_contract_sha256, "expected contract hash"),
        (expected_predecessor_manifest_sha256, "expected predecessor checkpoint hash"),
    ):
        require_hex64(value, label)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("checkpoint transport root is not one regular directory")
    manifest_path = root / TRANSPORT_MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("transport publication manifest is absent or a symlink")
    actual_transport_hash = sha256_file(manifest_path)
    if actual_transport_hash != expected_transport_manifest_sha256:
        raise RuntimeError("transport manifest hash differs")
    manifest = load_json_object(manifest_path, "transport manifest")
    if set(manifest) != transport_manifest_keys():
        raise RuntimeError("transport manifest fields differ")
    if manifest.get("schema") != TRANSPORT_SCHEMA:
        raise RuntimeError("transport schema differs")
    if manifest.get("contract_sha256") != expected_contract_sha256:
        raise RuntimeError("transport contract differs")
    if manifest.get("checkpoint_manifest_sha256") != expected_predecessor_manifest_sha256:
        raise RuntimeError("transport predecessor checkpoint differs")
    if manifest.get("dataset_id") != expected_dataset_id:
        raise RuntimeError("transport dataset identity differs")
    dataset_parts = expected_dataset_id.split("/")
    if (
        len(dataset_parts) != 2
        or OWNER.fullmatch(dataset_parts[0]) is None
        or DATASET_SLUG.fullmatch(dataset_parts[1]) is None
    ):
        raise RuntimeError("expected transport dataset identity is malformed")
    if manifest.get("intended_visibility") != "private":
        raise RuntimeError("transport visibility is not private")
    if manifest.get("checkpoint_verifier_sha256") != sha256_file(
        CHECKPOINT_RUNTIME_PATH
    ):
        raise RuntimeError("checkpoint verifier implementation differs")
    if manifest.get("transport_implementation_sha256") != sha256_file(MODULE_PATH):
        raise RuntimeError("transport implementation differs")
    payload_files = manifest.get("payload_files")
    if not isinstance(payload_files, list):
        raise TypeError("transport payload manifest is malformed")
    validate_file_records(
        payload_files,
        label="transport payload manifest",
        required_prefix=CHECKPOINT_DIRECTORY,
    )
    generation = manifest.get("checkpoint_generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise TypeError("transport checkpoint generation is malformed")
    completed_games = manifest.get("completed_games")
    if not isinstance(completed_games, list) or not all(
        isinstance(game, str) for game in completed_games
    ):
        raise TypeError("transport completed-game roster is malformed")
    chain_predecessor = manifest.get("chain_predecessor_manifest_sha256")
    if chain_predecessor is not None:
        if not isinstance(chain_predecessor, str):
            raise TypeError("transport chain predecessor is malformed")
        require_hex64(chain_predecessor, "transport chain predecessor")
    seed = publication_seed(
        contract_sha256=manifest["contract_sha256"],
        checkpoint_manifest_sha256=manifest["checkpoint_manifest_sha256"],
        checkpoint_generation=manifest["checkpoint_generation"],
        chain_predecessor_manifest_sha256=manifest[
            "chain_predecessor_manifest_sha256"
        ],
        completed_games=manifest["completed_games"],
        payload_files=payload_files,
        checkpoint_archive=manifest["checkpoint_archive"],
        checkpoint_verifier_sha256=manifest["checkpoint_verifier_sha256"],
        transport_implementation_sha256=manifest[
            "transport_implementation_sha256"
        ],
    )
    publication_identity = sha256_bytes(canonical_json(seed))
    if manifest.get("publication_identity") != publication_identity:
        raise RuntimeError("transport publication identity differs")
    dataset_slug = expected_dataset_id.split("/", 1)[-1]
    expected_suffix = f"-g{manifest['checkpoint_generation']}-{publication_identity[:20]}"
    if not dataset_slug.endswith(expected_suffix):
        raise RuntimeError("content-addressed dataset identity differs")
    expected_title = (
        f"Tycho NOOA public25 checkpoint g{manifest['checkpoint_generation']} "
        f"{publication_identity[:12]}"
    )
    if manifest.get("dataset_title") != expected_title:
        raise RuntimeError("transport dataset title differs")
    archive_record = manifest.get("checkpoint_archive")
    if not isinstance(archive_record, dict) or archive_record.get("path") != CHECKPOINT_ARCHIVE:
        raise TypeError("checkpoint archive record is malformed")
    validate_file_records(
        [archive_record], label="checkpoint archive record", required_prefix=None
    )
    actual_archive = next(
        (
            record
            for record in regular_file_manifest(root)
            if record["path"] == CHECKPOINT_ARCHIVE
        ),
        None,
    )
    if actual_archive != archive_record:
        raise RuntimeError("transport archive inventory or bytes differ")
    if manifest.get("payload_bytes") != sum(
        record["bytes"] for record in payload_files
    ):
        raise RuntimeError("transport payload byte total differs")
    control = manifest.get("control_metadata")
    if not isinstance(control, dict) or control.get("path") != DATASET_METADATA:
        raise RuntimeError("transport control metadata record differs")
    validate_file_records(
        [control], label="transport control metadata", required_prefix=None
    )
    metadata_path = root / DATASET_METADATA
    if require_control_metadata and (
        not metadata_path.is_file() or metadata_path.is_symlink()
    ):
        raise RuntimeError("dataset control metadata is absent from local staging")
    expected_root_files = [archive_record]
    expected_root_files.append(
        {
            "path": TRANSPORT_MANIFEST,
            "bytes": manifest_path.stat().st_size,
            "sha256": actual_transport_hash,
        }
    )
    if metadata_path.exists() or metadata_path.is_symlink():
        if not metadata_path.is_file() or metadata_path.is_symlink():
            raise RuntimeError("dataset control metadata is a symlink or non-file")
        actual_control = regular_file_manifest(root)
        actual_control_record = next(
            record for record in actual_control if record["path"] == DATASET_METADATA
        )
        if actual_control_record != control:
            raise RuntimeError("dataset control metadata bytes differ")
        if load_json_object(metadata_path, "dataset control metadata") != metadata_for(
            manifest["dataset_id"], manifest["dataset_title"]
        ):
            raise RuntimeError("dataset control metadata identity differs")
        expected_root_files.append(control)
    expected_root_files.sort(key=lambda record: record["path"])
    actual_root_files = regular_file_manifest(root)
    if actual_root_files != expected_root_files:
        raise RuntimeError("transport root inventory or bytes differ")
    with tempfile.TemporaryDirectory(prefix="checkpoint-transport-verify-") as temporary:
        extracted_root = Path(temporary) / CHECKPOINT_DIRECTORY
        extract_checkpoint_archive(
            root / CHECKPOINT_ARCHIVE, extracted_root, payload_files
        )
        checkpoint = validate_checkpoint_identity(
            extracted_root,
            expected_contract_sha256=expected_contract_sha256,
            expected_checkpoint_manifest_sha256=expected_predecessor_manifest_sha256,
        )
    if (
        checkpoint.get("generation") != manifest.get("checkpoint_generation")
        or checkpoint.get("predecessor_manifest_sha256")
        != manifest.get("chain_predecessor_manifest_sha256")
        or checkpoint.get("completed_games") != manifest.get("completed_games")
    ):
        raise RuntimeError("transport and checkpoint identity fields differ")
    return {**manifest, "transport_manifest_sha256": actual_transport_hash}


def verify_downloaded_readback(
    root: Path,
    *,
    expected_transport_manifest_sha256: str,
    expected_contract_sha256: str,
    expected_predecessor_manifest_sha256: str,
    expected_dataset_id: str,
) -> dict[str, Any]:
    """Verify exact downloaded dataset contents; Kaggle omits control metadata."""
    return verify_transport_root(
        root,
        expected_transport_manifest_sha256=expected_transport_manifest_sha256,
        expected_contract_sha256=expected_contract_sha256,
        expected_predecessor_manifest_sha256=expected_predecessor_manifest_sha256,
        expected_dataset_id=expected_dataset_id,
        require_control_metadata=False,
    )


def locate_single_mounted_transport(input_root: Path) -> Path:
    if not input_root.is_dir() or input_root.is_symlink():
        raise RuntimeError("Kaggle input root is not one regular directory")
    candidates = sorted(input_root.glob(f"*/{TRANSPORT_MANIFEST}"))
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one mounted checkpoint transport; "
            f"found {len(candidates)}"
        )
    manifest_path = candidates[0]
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("mounted transport manifest is not one regular file")
    root = manifest_path.parent
    current = root
    while current != input_root:
        if current.is_symlink():
            raise RuntimeError("mounted transport path contains a symlink")
        current = current.parent
    return root


def import_mounted_transport(
    input_root: Path,
    destination: Path,
    *,
    expected_transport_manifest_sha256: str,
    expected_contract_sha256: str,
    expected_predecessor_manifest_sha256: str,
    expected_dataset_id: str,
) -> dict[str, Any]:
    """Verify one mount, copy to a sibling temporary tree, then rename it."""
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"checkpoint import destination already exists: {destination}")
    transport_root = locate_single_mounted_transport(input_root)
    manifest = verify_downloaded_readback(
        transport_root,
        expected_transport_manifest_sha256=expected_transport_manifest_sha256,
        expected_contract_sha256=expected_contract_sha256,
        expected_predecessor_manifest_sha256=expected_predecessor_manifest_sha256,
        expected_dataset_id=expected_dataset_id,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_container = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.import-", dir=destination.parent
        )
    )
    temporary = temporary_container / CHECKPOINT_DIRECTORY
    try:
        extract_checkpoint_archive(
            transport_root / CHECKPOINT_ARCHIVE,
            temporary,
            manifest["payload_files"],
        )
        validate_checkpoint_identity(
            temporary,
            expected_contract_sha256=expected_contract_sha256,
            expected_checkpoint_manifest_sha256=expected_predecessor_manifest_sha256,
        )
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(
                f"checkpoint import destination already exists: {destination}"
            )
        os.replace(temporary, destination)
        temporary_container.rmdir()
        fsync_directory(destination.parent)
    except BaseException:
        if temporary_container.exists():
            shutil.rmtree(temporary_container)
        raise
    return {
        "schema": "tycho-nooa-public25-checkpoint-import-receipt-v1",
        "checkpoint_root": str(destination),
        "dataset_id": manifest["dataset_id"],
        "transport_manifest_sha256": manifest["transport_manifest_sha256"],
        "checkpoint_manifest_sha256": manifest["checkpoint_manifest_sha256"],
        "contract_sha256": manifest["contract_sha256"],
    }


def add_expected_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-transport-manifest-sha256", required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--expected-predecessor-manifest-sha256", required=True)
    parser.add_argument("--expected-dataset-id", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="build a fresh private dataset staging tree")
    export.add_argument("--checkpoint-root", type=Path, required=True)
    export.add_argument("--staging-root", type=Path, required=True)
    export.add_argument("--expected-contract-sha256", required=True)
    export.add_argument("--expected-checkpoint-manifest-sha256", required=True)
    export.add_argument("--owner", required=True)
    export.add_argument("--slug-prefix", default="tycho-nooa-p25-ckpt")

    readback = commands.add_parser(
        "verify-readback", help="verify a downloaded dataset byte for byte"
    )
    readback.add_argument("--root", type=Path, required=True)
    add_expected_arguments(readback)

    importer = commands.add_parser(
        "import", help="atomically import exactly one mounted predecessor"
    )
    importer.add_argument("--input-root", type=Path, required=True)
    importer.add_argument("--destination", type=Path, required=True)
    add_expected_arguments(importer)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "export":
        result = export_checkpoint_dataset(
            args.checkpoint_root,
            args.staging_root,
            expected_contract_sha256=args.expected_contract_sha256,
            expected_checkpoint_manifest_sha256=(
                args.expected_checkpoint_manifest_sha256
            ),
            owner=args.owner,
            slug_prefix=args.slug_prefix,
        )
    elif args.command == "verify-readback":
        result = verify_downloaded_readback(
            args.root,
            expected_transport_manifest_sha256=(
                args.expected_transport_manifest_sha256
            ),
            expected_contract_sha256=args.expected_contract_sha256,
            expected_predecessor_manifest_sha256=(
                args.expected_predecessor_manifest_sha256
            ),
            expected_dataset_id=args.expected_dataset_id,
        )
    else:
        result = import_mounted_transport(
            args.input_root,
            args.destination,
            expected_transport_manifest_sha256=(
                args.expected_transport_manifest_sha256
            ),
            expected_contract_sha256=args.expected_contract_sha256,
            expected_predecessor_manifest_sha256=(
                args.expected_predecessor_manifest_sha256
            ),
            expected_dataset_id=args.expected_dataset_id,
        )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
