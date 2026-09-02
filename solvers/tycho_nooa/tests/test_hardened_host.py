from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tycho_nooa import hardened_host


def hardening_environment() -> dict[str, str]:
    return {
        hardened_host.HARDENED_HOST_ENV: "1",
        hardened_host.GAME_UID_ENV: "60001",
        hardened_host.GAME_GID_ENV: "60001",
    }


def test_identity_requires_explicit_non_root_ids() -> None:
    assert hardened_host.enabled({}) is False
    assert hardened_host.enabled(hardening_environment()) is True
    assert hardened_host.identity(hardening_environment()) == (60001, 60001)

    for environment in (
        {hardened_host.HARDENED_HOST_ENV: "1"},
        {
            hardened_host.HARDENED_HOST_ENV: "1",
            hardened_host.GAME_UID_ENV: "0",
            hardened_host.GAME_GID_ENV: "0",
        },
    ):
        with pytest.raises(RuntimeError):
            hardened_host.identity(environment)


def test_command_binds_identity_and_real_generated_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    protected = workspace / "world_model.py"
    protected.write_text("VALUE = 1\n")
    generated_root = tmp_path / "generated"
    generated_root.mkdir()
    generated = generated_root / "source.py"
    generated.write_text("print('ok')\n")
    for key, value in hardening_environment().items():
        monkeypatch.setenv(key, value)

    result = hardened_host.command(
        workspace,
        (protected,),
        [
            sys.executable,
            "-B",
            "-c",
            "runpy.run_path('/tycho-generated/source.py')",
        ],
        workspace_read_only=True,
        generated_source=generated,
    )

    assert result[:3] == [
        sys.executable,
        "-B",
        str(Path(hardened_host.__file__).resolve()),
    ]
    assert result[result.index("--uid") + 1] == "60001"
    assert result[result.index("--workspace-mode") + 1] == "read-only"
    assert str(protected) in result
    assert str(generated) in result
    assert not any("/tycho-generated/source.py" in argument for argument in result)


def test_prepare_tree_gives_only_the_game_group_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    editable = workspace / "editable.py"
    editable.write_text("VALUE = 1\n")
    protected = workspace / "protected.py"
    protected.write_text("VALUE = 2\n")
    notes = workspace / "notes"
    notes.mkdir()
    artifact = notes / "validated_plan.json"
    artifact.write_text("{}\n")
    chowns: list[tuple[Path, int, int]] = []
    modes: dict[Path, int] = {}

    def record_chown(path: str | Path, uid: int, gid: int, **_: object) -> None:
        chowns.append((Path(path), uid, gid))

    monkeypatch.setattr(os, "chown", record_chown)
    monkeypatch.setattr(
        os, "chmod", lambda path, mode: modes.__setitem__(Path(path), mode)
    )
    hardened_host.prepare_tree(
        workspace,
        gid=60001,
        writable=True,
        protected_paths=(protected,),
    )

    assert modes == {
        workspace: 0o2750,
        editable: 0o640,
        protected: 0o640,
        notes: 0o2770,
        artifact: 0o660,
    }
    assert {path for path, uid, gid in chowns if uid == 0 and gid == 60001} == {
        workspace,
        editable,
        protected,
        notes,
        artifact,
    }


def test_prepare_tree_rejects_workspace_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("authority\n")
    (workspace / "escape").symlink_to(outside)
    monkeypatch.setattr(os, "chown", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="contains a symlink"):
        hardened_host.prepare_tree(workspace, gid=60001, writable=True)

    assert outside.read_text() == "authority\n"


def test_standard_input_is_replaced_with_devnull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(os, "open", lambda *args: calls.append(args) or 17)
    monkeypatch.setattr(os, "dup2", lambda *args: calls.append(args))
    monkeypatch.setattr(os, "close", lambda *args: calls.append(args))

    hardened_host.seal_standard_input()

    assert calls == [
        ("/dev/null", os.O_RDONLY | os.O_CLOEXEC),
        (17, 0),
        (17,),
    ]


def test_python_c_imports_modules_from_the_current_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "workspace_probe.py").write_text("VALUE = 41\n")
    marker = workspace / "marker.txt"
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in sys.path if entry not in ("", str(workspace))],
    )
    monkeypatch.setattr(sys, "argv", list(sys.argv))

    hardened_host.run_python_command(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, workspace_probe; "
                "pathlib.Path('marker.txt').write_text(str(workspace_probe.VALUE + 1))"
            ),
        ]
    )

    assert marker.read_text() == "42"


def test_python_script_imports_modules_from_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_root = tmp_path / "generated"
    script_root.mkdir()
    (script_root / "script_probe.py").write_text("VALUE = 41\n")
    marker = tmp_path / "marker.txt"
    script = script_root / "source.py"
    script.write_text(
        "import pathlib, script_probe\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(script_probe.VALUE + 1))\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in sys.path if entry not in ("", str(script_root))],
    )
    monkeypatch.setattr(sys, "argv", list(sys.argv))

    hardened_host.run_python_command([sys.executable, str(script)])

    assert marker.read_text() == "42"
