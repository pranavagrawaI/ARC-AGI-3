from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from construction_service import load_transport  # noqa: E402  # identity path
from headless import apply_constructor_argv  # noqa: E402


def test_cli_sets_constructor_env_and_forwards_v11_args(monkeypatch) -> None:
    monkeypatch.delenv("ARC_CONSTRUCTOR_TRANSPORT", raising=False)
    monkeypatch.delenv("ARC_CONSTRUCTOR_FIXTURE", raising=False)
    rest = apply_constructor_argv([
        "--constructor-transport", "http",
        "--constructor-url", "http://127.0.0.1:9/v1",
        "--run-dir", ".",
        "--skill", "interactive-game-solver",
    ])
    assert rest == ["--run-dir", ".", "--skill", "interactive-game-solver"]
    assert os.environ["ARC_CONSTRUCTOR_TRANSPORT"] == "http"
    assert os.environ["ARC_CONSTRUCTOR_URL"] == "http://127.0.0.1:9/v1"


def test_load_transport_does_not_use_tmpdir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    (tmp_path / "constructor-fixture.json").write_text("[]")
    monkeypatch.delenv("ARC_CONSTRUCTOR_TRANSPORT", raising=False)
    monkeypatch.delenv("ARC_CONSTRUCTOR_FIXTURE", raising=False)
    with pytest.raises(ValueError, match="ARC_CONSTRUCTOR_TRANSPORT required"):
        load_transport()
