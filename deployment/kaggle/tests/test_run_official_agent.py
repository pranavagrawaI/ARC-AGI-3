from __future__ import annotations

import subprocess
import sys
from pathlib import Path


RUNNER = Path(__file__).parents[1] / "runtime" / "run_official_agent.py"


def run_runner(main_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), str(main_path)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_successful_official_main_returns_zero(tmp_path: Path) -> None:
    main_path = tmp_path / "main.py"
    (tmp_path / "official_sibling.py").write_text("VALUE = 'official-main-complete'\n")
    main_path.write_text("from official_sibling import VALUE\nprint(VALUE)\n")

    result = run_runner(main_path)

    assert result.returncode == 0
    assert "official-main-complete" in result.stdout


def test_unhandled_agent_thread_failure_terminates_process_nonzero(
    tmp_path: Path,
) -> None:
    main_path = tmp_path / "main.py"
    main_path.write_text(
        """\
import threading
import time

def broken_game():
    raise RuntimeError("solver failed")

def healthy_game():
    time.sleep(2)

threads = [
    threading.Thread(target=healthy_game, name="healthy-game"),
    threading.Thread(target=broken_game, name="broken-game"),
]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print("must-not-complete")
"""
    )

    result = run_runner(main_path)

    assert result.returncode != 0
    assert "RuntimeError: solver failed" in result.stderr
    assert "must-not-complete" not in result.stdout


def test_main_arguments_are_forwarded_unchanged(tmp_path: Path) -> None:
    main_path = tmp_path / "main.py"
    main_path.write_text(
        "import json, sys\nprint(json.dumps(sys.argv, separators=(',', ':')))\n"
    )

    result = subprocess.run(
        [sys.executable, str(RUNNER), str(main_path), "--agent", "myagent"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == f'["{main_path}","--agent","myagent"]'
