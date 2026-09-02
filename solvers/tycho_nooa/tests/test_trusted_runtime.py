from __future__ import annotations

import json
import shutil
import sys
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
    generated_source = tmp_path / "generated" / "source.py"
    generated_source.parent.mkdir()
    generated_source.write_text("print('isolated')")
    command = [
        "docker",
        "run",
        "--mount",
        f"type=bind,src={tmp_path},dst=/workspace",
        "runtime-image",
        "python",
    ]

    protected = _container_runtime_command(
        "runtime-image", paths, command, generated_source=generated_source
    )

    image_index = protected.index("runtime-image")
    prefix = protected[:image_index]
    assert all(f"src={path}" in " ".join(prefix) for path in paths)
    assert prefix.count("--mount") == len(paths) + 2
    assert "readonly" in " ".join(prefix)
    assert (
        f"src={generated_source.parent},dst=/tycho-generated,readonly"
        in " ".join(prefix)
    )
    workspace_mount = protected[protected.index("--mount") + 1]
    assert workspace_mount.endswith(",readonly")


def test_host_sandbox_exposes_workspace_values_without_host_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is unavailable")
    from tycho_nooa.wm_probe import TrustedRuntimePythonSandbox

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    current_runtime().write_to(workspace)
    evidence_dir = workspace / "level_0"
    evidence_dir.mkdir()
    evidence = evidence_dir / "turn_000.json"
    evidence.write_text('{"level":0,"turn":0}')
    current_frame = workspace / ".current_frame.json"
    current_frame.write_text('{"path":"level_0/turn_000.txt"}')
    outside = tmp_path / "outside.env"
    outside.write_text("ARC_LLM_API_KEY=outside-secret\n")
    monkeypatch.setenv("ARC_LLM_API_KEY", "environment-secret")
    source = f"""
import json, os
from pathlib import Path

parent_environment_readable = None
parent_environment_secret = None
try:
    parent_environment = Path(f"/proc/{{os.getppid()}}/environ").read_bytes()
    parent_environment_readable = True
    parent_environment_secret = b"ARC_LLM_API_KEY=" in parent_environment
except Exception:
    parent_environment_readable = False

protected_write = None
try:
    Path("symmetry.py").write_text("corrupted")
except Exception as error:
    protected_write = type(error).__name__
evidence_write = None
try:
    Path("level_0/turn_000.json").write_text("corrupted")
except Exception as error:
    evidence_write = type(error).__name__
current_frame_write = None
try:
    Path(".current_frame.json").write_text("corrupted")
except Exception as error:
    current_frame_write = type(error).__name__
future_level_write = None
try:
    Path("level_999").mkdir()
    Path("level_999/terminal.json").write_text("fabricated")
except Exception as error:
    future_level_write = type(error).__name__
future_attempt_write = None
try:
    Path("attempts").mkdir()
except Exception as error:
    future_attempt_write = type(error).__name__
Path("/tmp/scratch.txt").write_text("allowed")
print(json.dumps({{
    "environment_secret": os.getenv("ARC_LLM_API_KEY"),
    "pid": os.getpid(),
    "ppid": os.getppid(),
    "parent_environment_readable": parent_environment_readable,
    "parent_environment_secret": parent_environment_secret,
    "generated_sources": sorted(
        path.name for path in Path(".").glob(".tycho-run-*.py")
    ),
    "outside_visible": Path({str(outside)!r}).exists(),
    "pids": sorted(path.name for path in Path("/proc").iterdir() if path.name.isdigit()),
    "protected_write": protected_write,
    "evidence_write": evidence_write,
    "current_frame_write": current_frame_write,
    "future_level_write": future_level_write,
    "future_attempt_write": future_attempt_write,
    "scratch": Path("/tmp/scratch.txt").read_text(),
}}))
"""

    result = TrustedRuntimePythonSandbox(runtime="host").run_source(
        workspace, source, timeout=10
    )

    report = json.loads(result.stdout)
    assert result.returncode == 0
    assert report["environment_secret"] is None
    assert report["pid"] == 1
    assert report["ppid"] == 0
    assert report["parent_environment_readable"] is False
    assert report["parent_environment_secret"] is None
    assert report["generated_sources"] == []
    assert report["outside_visible"] is False
    assert len(report["pids"]) <= 2
    assert report["protected_write"] == "OSError"
    assert report["evidence_write"] == "OSError"
    assert report["current_frame_write"] == "OSError"
    assert report["future_level_write"] == "OSError"
    assert report["future_attempt_write"] == "OSError"
    assert report["scratch"] == "allowed"
    assert evidence.read_text() == '{"level":0,"turn":0}'
    assert current_frame.read_text() == '{"path":"level_0/turn_000.txt"}'
    assert not (workspace / "scratch.txt").exists()
    assert not (workspace / "level_999").exists()
    assert not (workspace / "attempts").exists()
    assert inspect_runtime(workspace).state is RuntimeState.DECLARED


def test_host_sandbox_command_builds_an_empty_filesystem(tmp_path: Path) -> None:
    from tycho_nooa.wm_probe import _bubblewrap_command

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = _bubblewrap_command(
        "bwrap",
        workspace,
        (),
        ["python", "script.py"],
    )

    for flag in (
        "--new-session",
        "--as-pid-1",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--clearenv",
    ):
        assert flag in command
    assert ["--ro-bind", "/", "/"] not in [
        command[index : index + 3] for index in range(len(command) - 2)
    ]
    assert ["--ro-bind", "/usr", "/usr"] in [
        command[index : index + 3] for index in range(len(command) - 2)
    ]
    venv = str(Path(sys.prefix).resolve())
    assert ["--ro-bind", venv, venv] in [
        command[index : index + 3] for index in range(len(command) - 2)
    ]
    workspace_text = str(workspace.resolve())
    assert ["--ro-bind", workspace_text, workspace_text] in [
        command[index : index + 3] for index in range(len(command) - 2)
    ]


def test_explicit_hardened_host_routes_around_bubblewrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tycho_nooa.hardened_host import (
        GAME_GID_ENV,
        GAME_UID_ENV,
        HARDENED_HOST_ENV,
    )
    from tycho_nooa.wm_probe import _host_runtime_command

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    protected = workspace / "world_model.py"
    protected.write_text("VALUE = 1\n")
    monkeypatch.setenv(HARDENED_HOST_ENV, "1")
    monkeypatch.setenv(GAME_UID_ENV, "60001")
    monkeypatch.setenv(GAME_GID_ENV, "60001")
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    command = _host_runtime_command(
        workspace,
        (protected,),
        [sys.executable, "-B", str(protected)],
    )

    assert command[:3] == [
        sys.executable,
        "-B",
        str(Path(__file__).parents[1] / "src/tycho_nooa/hardened_host.py"),
    ]
    assert command[command.index("--uid") + 1] == "60001"
    assert str(protected) in command


def manifest_module(manifest: dict[str, object], path: str) -> dict[str, str]:
    modules = manifest["modules"]
    assert isinstance(modules, list)
    return next(module for module in modules if module["path"] == path)
