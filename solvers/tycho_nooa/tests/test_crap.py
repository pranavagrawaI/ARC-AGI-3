from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SOLVER_ROOT = Path(__file__).parents[1]
CRAP_TOOL = SOLVER_ROOT / "tools" / "crap.py"


def test_report_uses_function_coverage_and_sorts_highest_score_first(
    tmp_path: Path,
) -> None:
    source, test_file = write_sample_project(tmp_path)

    result = run_crap(source, test_file, "--format", "json")

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert [function["function"] for function in report["functions"]] == [
        "untouched",
        "exercised",
    ]
    assert report["functions"][0] == {
        "complexity": 2,
        "coverage": 0.0,
        "crap": 6.0,
        "file": str(source),
        "function": "untouched",
        "line": 5,
    }
    assert report["functions"][1]["coverage"] == 100.0
    assert report["functions"][1]["crap"] == 1.0


def test_fail_above_is_strict_and_returns_a_gate_exit_code(tmp_path: Path) -> None:
    source, test_file = write_sample_project(tmp_path)

    at_score = run_crap(source, test_file, "--fail-above", "6")
    below_score = run_crap(source, test_file, "--fail-above", "5.99")

    assert at_score.returncode == 0, at_score.stderr
    assert "0 function(s) strictly above CRAP 6." in at_score.stdout
    assert below_score.returncode == 1, below_score.stderr
    assert "untouched FAIL" in below_score.stdout
    assert "1 function(s) strictly above CRAP 5.99." in below_score.stdout


def test_nested_method_is_reported_under_its_qualified_name(tmp_path: Path) -> None:
    source = tmp_path / "nested.py"
    source.write_text(
        """\
def outer():
    class Inner:
        def method(self, flag):
            if flag:
                return 1
            return 0
    return Inner
"""
    )
    test_file = tmp_path / "test_nested.py"
    test_file.write_text(
        f"""\
import importlib.util

spec = importlib.util.spec_from_file_location("nested", {str(source)!r})
nested = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nested)


def test_outer():
    assert nested.outer().__name__ == "Inner"
"""
    )

    result = run_crap(source, test_file, "--format", "json")

    assert result.returncode == 0, result.stderr
    functions = {
        function["function"]: function
        for function in json.loads(result.stdout)["functions"]
    }
    assert functions["outer.Inner.method"]["coverage"] == 0.0
    assert functions["outer.Inner.method"]["complexity"] == 2
    assert functions["outer"]["coverage"] == 100.0


def test_failing_tests_stop_analysis_without_emitting_a_report(tmp_path: Path) -> None:
    source, test_file = write_sample_project(tmp_path)
    test_file.write_text("def test_failure():\n    assert False\n")

    result = run_crap(source, test_file, "--format", "json")

    assert result.returncode == 3
    assert result.stdout == ""
    assert "1 failed" in result.stderr
    assert "CRAP analysis stopped: pytest exited 1" in result.stderr


def write_sample_project(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "sample.py"
    source.write_text(
        """\
def exercised():
    return 7


def untouched(flag):
    if flag:
        return 1
    return 0
"""
    )
    test_file = tmp_path / "test_sample.py"
    test_file.write_text(
        f"""\
import importlib.util

spec = importlib.util.spec_from_file_location("sample", {str(source)!r})
sample = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sample)


def test_exercised():
    assert sample.exercised() == 7
"""
    )
    return source, test_file


def run_crap(
    source: Path, test_file: Path, *tool_args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CRAP_TOOL),
            "--source",
            str(source),
            *tool_args,
            "--",
            "-s",
            "-p",
            "no:cacheprovider",
            str(test_file),
        ],
        cwd=SOLVER_ROOT,
        env={**os.environ, "TMPDIR": "/tmp"},
        capture_output=True,
        text=True,
        check=False,
    )
