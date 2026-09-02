from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "wheelhouse-manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_identity(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
    return metadata["Name"], metadata["Version"]


def verify() -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_text())
    failures: list[str] = []
    if manifest.get("schema") != "tycho-nooa-full-wheelhouse-v1":
        failures.append("unexpected manifest schema")
    if manifest.get("artifact_revision") != 8:
        failures.append("unexpected artifact revision")
    expected_files = {record["filename"] for record in manifest["wheels"]}
    actual_files = {path.name for path in ROOT.glob("*.whl")}
    if actual_files != expected_files:
        failures.append(
            f"wheel filenames differ: missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    required_entries = {
        *expected_files,
        *(record["filename"] for record in manifest["support_files"]),
        MANIFEST.name,
    }
    control_entries = {record["filename"] for record in manifest["control_files"]}
    actual_entries = {path.name for path in ROOT.iterdir()}
    missing_entries = required_entries - actual_entries
    unexpected_entries = actual_entries - required_entries - control_entries
    if missing_entries or unexpected_entries:
        failures.append(
            f"artifact entries differ: missing={sorted(missing_entries)}, "
            f"unexpected={sorted(unexpected_entries)}"
        )
    identities: dict[str, str] = {}
    for record in manifest["wheels"]:
        path = ROOT / record["filename"]
        if not path.is_file():
            continue
        if path.stat().st_size != record["bytes"]:
            failures.append(f"{path.name}: byte count differs")
        if sha256(path) != record["sha256"]:
            failures.append(f"{path.name}: sha256 differs")
        name, version = wheel_identity(path)
        if (name, version) != (record["name"], record["version"]):
            failures.append(f"{path.name}: METADATA identity differs")
        canonical_name = name.lower().replace("_", "-")
        if canonical_name in identities:
            failures.append(f"duplicate package identity: {canonical_name}")
        identities[canonical_name] = version
    for record in manifest["support_files"]:
        path = ROOT / record["filename"]
        if not path.is_file():
            failures.append(f"{record['filename']}: missing")
            continue
        if path.stat().st_size != record["bytes"]:
            failures.append(f"{record['filename']}: byte count differs")
        if sha256(path) != record["sha256"]:
            failures.append(f"{record['filename']}: sha256 differs")
    for record in manifest["control_files"]:
        path = ROOT / record["filename"]
        if not path.exists():
            continue
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            failures.append(f"{record['filename']}: control-file bytes differ")
        elif sha256(path) != record["sha256"]:
            failures.append(f"{record['filename']}: control-file sha256 differs")

    source_manifest = json.loads((ROOT / "solver-source-manifest.json").read_text())
    solver_record = next(
        record
        for record in manifest["wheels"]
        if record["name"] == "arc-agi-3-tycho-nooa"
    )
    if source_manifest.get("build", {}).get("wheel") != {
        key: solver_record[key] for key in ("bytes", "filename", "sha256")
    }:
        failures.append("source manifest Solver wheel identity differs")
    with zipfile.ZipFile(ROOT / solver_record["filename"]) as archive:
        actual_members = {
            name: (len(data), sha256_bytes(data))
            for name in archive.namelist()
            if not name.endswith("/")
            for data in (archive.read(name),)
        }
    expected_members = {
        record["path"]: (record["bytes"], record["sha256"])
        for record in source_manifest["wheel_members"]
    }
    if actual_members != expected_members:
        failures.append("Solver wheel members differ from source manifest")
    return {
        "artifact_revision": manifest.get("artifact_revision"),
        "failures": failures,
        "manifest_sha256": sha256(MANIFEST),
        "package_count": len(identities),
        "schema": manifest.get("schema"),
        "verified": not failures,
        "wheel_count": len(actual_files),
    }


if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    sys.exit(0 if result["verified"] else 1)
