from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

import pytest

HERE = Path(__file__).resolve().parents[1]
PROBE_PATH = HERE / "predecessor_mount_probe.py"
FIXTURE_PATH = HERE / "checkpoint-fixture.json"
SOURCE_V1 = HERE / "fixtures" / "predecessor-v1"
SOURCE_V2_RESULT = HERE / "fixtures" / "latest-v2-result.json"


def load_probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("predecessor_mount_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text())


def mount_source(input_root: Path, *, name: str = "mounted-source") -> Path:
    target = input_root / name
    shutil.copytree(SOURCE_V1, target)
    return target


def test_exact_version_one_mount_is_verified_before_continuation(tmp_path: Path) -> None:
    probe = load_probe()
    input_root = tmp_path / "input"
    input_root.mkdir()
    mount_source(input_root)
    output_root = tmp_path / "working" / "qualification"

    returncode = probe.run(input_root, output_root, fixture())

    assert returncode == 0
    result = json.loads((output_root / "result.json").read_text())
    assert result["status"] == "passed"
    assert result["expected_predecessor"] == {
        "kernel_source": "pranavagrawai/prototype-tycho-nooa-successor-concurrency/1",
        "kernel_id_no": 132802830,
        "version": 1,
    }
    assert result["continued_after_verification"] is True
    assert result["selected_file_count"] == 3
    assert result["selected_bytes"] == 1_409
    assert result["arc_actions"] == 0
    assert result["model_requests"] == 0
    assert result["competition_sources"] == 0
    assert result["submissions"] == 0
    assert (output_root / "continuation-marker.json").is_file()
    imported = output_root / "verified-predecessor"
    assert probe.regular_file_manifest(imported) == fixture()["files"]


def test_latest_version_two_bytes_are_rejected_as_a_version_pin_failure(
    tmp_path: Path,
) -> None:
    probe = load_probe()
    input_root = tmp_path / "input"
    input_root.mkdir()
    mounted = mount_source(input_root)
    shutil.copy2(
        SOURCE_V2_RESULT,
        mounted / "tycho-nooa-successor-concurrency" / "result.json",
    )
    output_root = tmp_path / "working" / "qualification"

    returncode = probe.run(input_root, output_root, fixture())

    assert returncode == 74
    result = json.loads((output_root / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["failure_classification"] == "latest_version_mounted"
    assert result["continued_after_verification"] is False
    assert not (output_root / "continuation-marker.json").exists()
    assert not (output_root / "verified-predecessor").exists()


@pytest.mark.parametrize("failure", ["tamper", "missing", "duplicate", "symlink"])
def test_incomplete_ambiguous_or_nonregular_mount_never_continues(
    tmp_path: Path, failure: str
) -> None:
    probe = load_probe()
    input_root = tmp_path / "input"
    input_root.mkdir()
    mounted = mount_source(input_root)
    selected = mounted / "tycho-nooa-successor-concurrency" / "shutdown.json"
    if failure == "tamper":
        selected.write_text("tampered\n")
    elif failure == "missing":
        selected.unlink()
    elif failure == "duplicate":
        mount_source(input_root, name="second-mounted-source")
    elif failure == "symlink":
        original = mounted / "shutdown-original.json"
        selected.rename(original)
        selected.symlink_to(original.name)
    output_root = tmp_path / "working" / "qualification"

    returncode = probe.run(input_root, output_root, fixture())

    assert returncode == 74
    result = json.loads((output_root / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["continued_after_verification"] is False
    assert not (output_root / "continuation-marker.json").exists()
    assert not (output_root / "verified-predecessor").exists()


def test_fixture_manifest_is_canonical_and_bound_to_version_one_bytes() -> None:
    probe = load_probe()
    value = fixture()

    assert value["schema"] == "tycho-nooa-predecessor-output-checkpoint-fixture-v1"
    assert value["predecessor"]["version"] == 1
    assert value["negative_control"]["latest_version"] == 2
    assert value["negative_control"]["result_sha256"] != next(
        record["sha256"] for record in value["files"] if record["path"].endswith("result.json")
    )
    assert probe.regular_file_manifest(SOURCE_V1) == value["files"]
    assert probe.sha256_bytes(probe.canonical_json(value["files"])) == value[
        "source_manifest_sha256"
    ]


def test_preexisting_import_evidence_is_never_deleted_on_refusal(tmp_path: Path) -> None:
    probe = load_probe()
    input_root = tmp_path / "input"
    input_root.mkdir()
    mount_source(input_root)
    output_root = tmp_path / "working" / "qualification"
    existing = output_root / "verified-predecessor"
    existing.mkdir(parents=True)
    sentinel = existing / "preserve-me.txt"
    sentinel.write_text("prior immutable evidence\n")

    returncode = probe.run(input_root, output_root, fixture())

    assert returncode == 74
    result = json.loads((output_root / "result.json").read_text())
    assert result["failure_classification"] == "output_not_clean"
    assert sentinel.read_text() == "prior immutable evidence\n"
