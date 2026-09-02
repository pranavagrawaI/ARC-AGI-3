from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "wheelhouse-manifest.json"


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
    expected_files = {record["filename"] for record in manifest["wheels"]}
    actual_files = {path.name for path in ROOT.glob("*.whl")}
    if actual_files != expected_files:
        failures.append(
            f"wheel filenames differ: missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
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
        if not path.is_file() or sha256(path) != record["sha256"]:
            failures.append(f"{record['filename']}: missing or sha256 differs")
    return {
        "schema": manifest["schema"],
        "wheel_count": len(actual_files),
        "package_count": len(identities),
        "failures": failures,
        "verified": not failures,
    }


if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    sys.exit(0 if result["verified"] else 1)
