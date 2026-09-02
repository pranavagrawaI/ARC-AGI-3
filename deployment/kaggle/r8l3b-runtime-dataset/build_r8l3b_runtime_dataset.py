#!/usr/bin/env python3
"""Build the frozen thin Kaggle runtime dataset for the R8-L3B SGLang wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parents[1]
DATASET_ID = "pranavagrawai/sglang-r8l3b-orderly-shutdown-runtime"
DATASET_TITLE = "SGLang R8-L3B orderly shutdown runtime"
DATASET_VERSION = 1
FIXED_MTIME = 1_704_067_200
SGLANG_REPOSITORY = "https://github.com/sgl-project/sglang.git"
SGLANG_COMMIT = "8e04f66a7046fa083cda3f5b8e8325632a5af8fd"
WHEEL_NAME = "sglang-0.5.19.dev591+g8e04f66a7-cp312-cp312-linux_x86_64.whl"
BASE_WHEEL_SHA256 = "9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d"
PATCHED_WHEEL_SHA256 = "06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0"
LOCK_SHA256 = "d473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c"

SOURCES = {
    WHEEL_NAME: (
        ROOT / "runtime-patches" / "r8l3b" / WHEEL_NAME,
        PATCHED_WHEEL_SHA256,
        "patched_runtime_wheel",
    ),
    "requirements.lock": (
        Path("/tmp/sglang-r8l3-runtime/requirements.lock"),
        LOCK_SHA256,
        "inherited_dependency_lock",
    ),
    "build_sglang_r8l3_wheel.py": (
        ROOT / "scripts" / "build_sglang_r8l3_wheel.py",
        "3ce16555e2625ef72d10bc4e9096d8eed320a9be8f6ea33aa539505900220c6b",
        "wheel_builder",
    ),
    "r8l3b-local-manifest.json": (
        ROOT / "runtime-patches" / "r8l3b" / "manifest.json",
        "adb11a6608be04399bd903de99c96288ace6d69fd58520214de3dc9c98c2a630",
        "accepted_local_manifest",
    ),
    "R8L3B-LOCAL-EXPERIMENT-CARD.md": (
        ROOT / "PROTOTYPE-r8l3b-sglang-orderly-shutdown-EXPERIMENT-CARD.md",
        "ca77e4a1a4bf49c3a32b007ab7b36f0509d79d1e1729a2dffd22691c378ec4aa",
        "accepted_local_card",
    ),
    "R8L3B-LOCAL-IMPLEMENTATION-REVIEW.md": (
        REPOSITORY_ROOT
        / "analysis/deployment-probes/20260831T112704Z-sglang-r8l3b-local/IMPLEMENTATION-REVIEW.md",
        "c8ae13c55560426065949bea983575c8cf5f376fbd8778f0223c1c9e7a96839a",
        "accepted_local_implementation_review",
    ),
    "R8L3B-LOCAL-ARTIFACT-REVIEW.md": (
        REPOSITORY_ROOT
        / "analysis/deployment-probes/20260831T112704Z-sglang-r8l3b-local/ARTIFACT-REVIEW.md",
        "573dd56c62a079454137e0cabb8b62bae6dca81d5f955385e33c1387f5839f50",
        "accepted_local_artifact_review",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": path.name,
        "role": role,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def normalize_file(path: Path) -> None:
    path.chmod(0o644)
    os.utime(path, (FIXED_MTIME, FIXED_MTIME), follow_symlinks=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "r8l3b-runtime-dataset",
        help="new output directory; it must not already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    metadata_path = output / "dataset-metadata.json"
    write_json(
        metadata_path,
        {
            "title": DATASET_TITLE,
            "id": DATASET_ID,
            "licenses": [{"name": "other"}],
        },
    )

    records: list[dict[str, Any]] = [file_record(metadata_path, "kaggle_metadata")]
    for output_name, (source, expected_sha256, role) in SOURCES.items():
        if not source.is_file() or source.is_symlink():
            raise SystemExit(f"required regular source file missing: {source}")
        actual_sha256 = sha256_file(source)
        if actual_sha256 != expected_sha256:
            raise SystemExit(
                f"source hash mismatch for {source}: expected {expected_sha256}, got {actual_sha256}"
            )
        target = output / output_name
        shutil.copyfile(source, target)
        records.append(file_record(target, role))

    builder_copy = output / Path(__file__).name
    shutil.copyfile(Path(__file__).resolve(), builder_copy)
    records.append(file_record(builder_copy, "dataset_builder"))

    manifest_path = output / "runtime-dataset-manifest.json"
    write_json(
        manifest_path,
        {
            "schema": "sglang-r8l3b-thin-runtime-dataset-v1",
            "status": "prepared_not_uploaded_not_live_qualified",
            "authority": (
                "This manifest grants no dataset upload, Kaggle kernel push, GPU execution, "
                "retry, S0, gameplay, submission, or promotion authority."
            ),
            "dataset": {
                "source": DATASET_ID,
                "expected_initial_version": DATASET_VERSION,
                "private": True,
                "internet_required": False,
                "layout": "flat_files",
                "fixed_mtime_epoch_seconds": FIXED_MTIME,
            },
            "runtime": {
                "repository": SGLANG_REPOSITORY,
                "commit": SGLANG_COMMIT,
                "wheel_name": WHEEL_NAME,
                "base_wheel_sha256": BASE_WHEEL_SHA256,
                "patched_wheel_sha256": PATCHED_WHEEL_SHA256,
                "dependency_lock_sha256": LOCK_SHA256,
                "dependency_policy": (
                    "The patched wheel and inherited lock are carried here. All other wheels "
                    "remain inherited unchanged from the separately attached base runtime dataset."
                ),
            },
            "files": sorted(records, key=lambda item: str(item["path"])),
            "claim_ceiling": (
                "These bytes prepare a thin private dataset candidate and preserve local review "
                "provenance. They do not establish successful upload, mounted-byte identity, "
                "offline dependency resolution, GPU startup, request behavior, or lifecycle closure."
            ),
        },
    )

    for path in output.iterdir():
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"unexpected non-regular dataset entry: {path}")
        normalize_file(path)

    print(f"wrote {output}")
    print(f"manifest_sha256={sha256_file(manifest_path)}")


if __name__ == "__main__":
    main()
