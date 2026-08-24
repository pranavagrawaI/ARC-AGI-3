from __future__ import annotations

import json
from pathlib import Path

import pytest

from tycho_nooa.trusted_runtime import (
    MANIFEST_PATH,
    RuntimeState,
    TrustedRuntimeError,
    current_runtime,
    inspect_runtime,
)


def test_current_runtime_has_a_versioned_byte_exact_manifest(tmp_path: Path) -> None:
    runtime = current_runtime()

    runtime.write_to(tmp_path)

    manifest = json.loads((tmp_path / MANIFEST_PATH).read_text())
    assert manifest["schema_version"] == 1
    assert manifest["runtime_sha256"] == runtime.sha256
    assert manifest["integrity"] == "read_only"
    assert manifest["restore_policy"] == "exact_version"
    assert "transaction_evaluator" in manifest["availability"]
    assert {module["path"] for module in manifest["modules"]} == {
        "dynamics.py",
        "wmlib.py",
        "symmetry.py",
        "host_planner_contract.py",
        "theory_probe.py",
    }
    for module in runtime.modules:
        assert (tmp_path / module.path).read_bytes() == module.content
        assert module.sha256 == manifest_module(manifest, module.path)["sha256"]
        assert manifest_module(manifest, module.path)["provenance"]


def test_declared_runtime_round_trips_from_workspace(tmp_path: Path) -> None:
    expected = current_runtime()
    expected.write_to(tmp_path)

    inspection = inspect_runtime(tmp_path)

    assert inspection.state is RuntimeState.DECLARED
    assert inspection.runtime == expected


def test_declared_runtime_fails_closed_when_a_helper_is_changed(tmp_path: Path) -> None:
    current_runtime().write_to(tmp_path)
    (tmp_path / "symmetry.py").write_text("# modified\n")

    with pytest.raises(TrustedRuntimeError, match="symmetry.py"):
        inspect_runtime(tmp_path)


def test_checkpoint_without_manifest_is_legacy_unknown(tmp_path: Path) -> None:
    (tmp_path / "wmlib.py").write_text("# historical helper\n")
    (tmp_path / "symmetry.py").write_text("# maybe historical\n")

    inspection = inspect_runtime(tmp_path)

    assert inspection.state is RuntimeState.LEGACY_UNKNOWN
    assert inspection.runtime is None
    assert inspection.observed_paths == ("symmetry.py", "wmlib.py")


def test_runtime_constructor_rejects_unsafe_or_duplicate_modules() -> None:
    from tycho_nooa.trusted_runtime import TrustedModule, TrustedRuntime

    module = current_runtime().modules[0]
    with pytest.raises(TrustedRuntimeError, match="at most once"):
        TrustedRuntime.create((module, module))
    with pytest.raises(TrustedRuntimeError, match="unsupported.*module set"):
        TrustedRuntime.create((module,))
    with pytest.raises(TrustedRuntimeError, match="identity"):
        TrustedModule.from_source("../escape.py", "escape", "model", b"bad")


def test_transactional_legacy_resume_rejects_even_an_injected_evaluator(
    tmp_path: Path,
) -> None:
    from tycho.workspace.agent_tools import ToolExecutor
    from tycho.workspace.workspace import GameWorkspace

    from tycho_nooa.agent import _builder_runtime
    from tycho_nooa.treatments import TreatmentPolicy

    workspace = GameWorkspace(
        "legacy", root=str(tmp_path), available_actions=["RIGHT"]
    )
    treatment = TreatmentPolicy("transactional_builder", transactional_builder=True)

    with pytest.raises(RuntimeError, match="legacy checkpoint"):
        _builder_runtime(
            workspace,
            ToolExecutor(workspace),
            treatment,
            evaluator=object(),  # type: ignore[arg-type]
            trusted_runtime=None,
        )


def test_container_command_mounts_every_runtime_path_read_only(tmp_path: Path) -> None:
    from tycho_nooa.wm_probe import _container_runtime_command

    paths = (tmp_path / ".trusted-runtime.json", tmp_path / "symmetry.py")
    command = ["docker", "run", "runtime-image", "python"]

    protected = _container_runtime_command("runtime-image", paths, command)

    image_index = protected.index("runtime-image")
    prefix = protected[:image_index]
    assert all(f"src={path}" in " ".join(prefix) for path in paths)
    assert prefix.count("--mount") == len(paths)
    assert "readonly" in " ".join(prefix)


def manifest_module(manifest: dict[str, object], path: str) -> dict[str, str]:
    modules = manifest["modules"]
    assert isinstance(modules, list)
    return next(module for module in modules if module["path"] == path)
