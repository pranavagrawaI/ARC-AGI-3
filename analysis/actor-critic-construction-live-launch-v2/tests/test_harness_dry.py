from __future__ import annotations

import json
import subprocess
from pathlib import Path

LIVE = Path(__file__).resolve().parents[1]
ROOT = LIVE.parents[1]
PYTHON = ROOT / "deployment/kaggle/.venv/bin/python"


def test_dry_run_does_not_create_raw_or_spend(tmp_path: Path) -> None:
    proc = subprocess.run(
        [str(PYTHON), str(LIVE / "harness.py"), "--out", str(tmp_path / "raw")],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["paid_calls"] == 0
    assert payload["source_verified"] is False
    assert payload["source_bind"] == "not_run"
    assert payload["caps"]["calls"] == 24
    assert payload["caps"]["actor_summarizer_cap"] == 20
    assert payload["caps"]["http_post_seconds"] == 30
    assert payload["caps"]["constructor_hard_seconds"] == 50
    assert not (tmp_path / "raw").exists()


def test_refuse_reused_raw(tmp_path: Path) -> None:
    reused = tmp_path / "raw"
    reused.mkdir()
    proc = subprocess.run(
        [str(PYTHON), str(LIVE / "harness.py"), "--out", str(reused)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode != 0
    assert "refuse reused raw" in proc.stderr or "refuse reused raw" in proc.stdout

