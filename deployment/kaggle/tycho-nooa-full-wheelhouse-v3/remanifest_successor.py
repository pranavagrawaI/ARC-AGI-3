from __future__ import annotations

import hashlib
import json
import zipfile
from email.parser import BytesParser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
PREDECESSOR = ROOT.parent / "tycho-nooa-full-wheelhouse-v2"
SOLVER_WHEEL = "arc_agi_3_tycho_nooa-0.1.0-py3-none-any.whl"
SOLVER_SOURCE = REPOSITORY / "solvers/tycho_nooa/src"


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


def record(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "filename": path.name,
        "sha256": sha256(path),
    }


def wheel_record(path: Path) -> dict[str, object]:
    name, version = wheel_identity(path)
    return {**record(path), "name": name, "version": version}


def build_source_manifest() -> dict[str, object]:
    current_wheel = ROOT / SOLVER_WHEEL
    old_wheel = PREDECESSOR / SOLVER_WHEEL
    packaged_sources: list[dict[str, object]] = []
    wheel_members: list[dict[str, object]] = []
    changed_sources: list[dict[str, object]] = []
    with (
        zipfile.ZipFile(current_wheel) as current,
        zipfile.ZipFile(old_wheel) as old,
    ):
        old_members = set(old.namelist())
        for name in sorted(current.namelist()):
            if name.endswith("/"):
                continue
            data = current.read(name)
            wheel_members.append(
                {"bytes": len(data), "path": name, "sha256": sha256_bytes(data)}
            )
            if not name.startswith("tycho_nooa/"):
                continue
            source = SOLVER_SOURCE / name
            source_data = source.read_bytes()
            if data != source_data:
                raise RuntimeError(f"wheel member differs from source: {name}")
            packaged_sources.append(
                {
                    "bytes": len(source_data),
                    "path": str(source.relative_to(REPOSITORY)),
                    "sha256": sha256_bytes(source_data),
                    "wheel_path": name,
                }
            )
            old_data = old.read(name) if name in old_members else None
            if old_data != data:
                changed_sources.append(
                    {
                        "path": name,
                        "predecessor_bytes": len(old_data) if old_data is not None else None,
                        "predecessor_sha256": (
                            sha256_bytes(old_data) if old_data is not None else None
                        ),
                        "successor_bytes": len(data),
                        "successor_sha256": sha256_bytes(data),
                    }
                )
    provenance_paths = [
        REPOSITORY / "solvers/tycho_nooa/pyproject.toml",
        REPOSITORY / "solvers/tycho_nooa/uv.lock",
    ]
    return {
        "schema": "tycho-nooa-solver-source-manifest-v1",
        "base_commit": "88108520f0294530feb481f795cf4d3ea0d37111",
        "build": {
            "command": "uv build --project solvers/tycho_nooa --wheel",
            "source_date_epoch": 1787952851,
            "wheel": record(current_wheel),
        },
        "changed_packaged_sources_from_predecessor": changed_sources,
        "packaged_sources": packaged_sources,
        "provenance_files": [
            {
                "bytes": path.stat().st_size,
                "path": str(path.relative_to(REPOSITORY)),
                "sha256": sha256(path),
            }
            for path in provenance_paths
        ],
        "wheel_members": wheel_members,
    }


def main() -> None:
    source_manifest_path = ROOT / "solver-source-manifest.json"
    source_manifest_path.write_text(
        json.dumps(build_source_manifest(), indent=2, sort_keys=True) + "\n"
    )

    predecessor_manifest_path = PREDECESSOR / "wheelhouse-manifest.json"
    predecessor = json.loads(predecessor_manifest_path.read_text())
    manifest = {
        **predecessor,
        "artifact_revision": 3,
        "control_files": [record(ROOT / "dataset-metadata.json")],
        "predecessor_manifest_sha256": sha256(predecessor_manifest_path),
        "source": {
            **predecessor["source"],
            "solver_artifact_note": (
                "The Solver wheel is authoritative for deployment and binds the "
                "exact shared-worktree package sources in solver-source-manifest.json."
            ),
            "solver_source_manifest": "solver-source-manifest.json",
        },
        "support_files": [],
        "wheels": sorted(
            (wheel_record(path) for path in ROOT.glob("*.whl")),
            key=lambda item: str(item["name"]).lower().replace("_", "-"),
        ),
    }
    support_names = [
        "local_solver.py",
        "remanifest_successor.py",
        "requirements.lock.txt",
        "solver-source-manifest.json",
        "uv-export.raw.txt",
        "uv.lock",
        "verify_wheelhouse.py",
    ]
    manifest["support_files"] = [record(ROOT / name) for name in support_names]
    (ROOT / "wheelhouse-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
