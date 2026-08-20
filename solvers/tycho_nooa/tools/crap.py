#!/usr/bin/env python3
"""Measure function-level CRAP for the Tycho-NOOA solver."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from radon.complexity import cc_visit_ast

SOLVER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("src/tycho_nooa")


class AnalysisError(RuntimeError):
    """The requested CRAP analysis could not produce trustworthy results."""


@dataclass(frozen=True)
class SourceFunction:
    name: str
    line: int
    complexity: int


@dataclass(frozen=True)
class CrapResult:
    crap: float
    complexity: int
    coverage: float
    file: str
    function: str
    line: int

    def as_report_dict(self) -> dict[str, float | int | str]:
        report = asdict(self)
        report["crap"] = round(self.crap, 2)
        report["coverage"] = round(self.coverage, 2)
        return report


class FunctionCollector(ast.NodeVisitor):
    """Identify Python function scopes using Coverage.py's qualified-name form."""

    def __init__(self) -> None:
        self._scope: list[str] = []
        self.functions: list[SourceFunction] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        blocks = cc_visit_ast(node)
        block = next(
            (
                candidate
                for candidate in blocks
                if candidate.name == node.name and candidate.lineno == node.lineno
            ),
            None,
        )
        if block is None:
            raise AnalysisError(
                f"Radon did not return complexity for {node.name!r} on line {node.lineno}"
            )

        qualified_name = ".".join([*self._scope, node.name])
        self.functions.append(
            SourceFunction(
                name=qualified_name,
                line=node.lineno,
                complexity=block.complexity,
            )
        )
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source_files = discover_source_files(args.source)
        coverage_report = measure_coverage(args.source, args.pytest_args)
        results = calculate_results(source_files, coverage_report)
    except AnalysisError as error:
        print(f"CRAP analysis failed: {error}", file=sys.stderr)
        return 2
    except TestRunFailed as error:
        print(error.output, file=sys.stderr, end="")
        print(
            f"CRAP analysis stopped: pytest exited {error.returncode}", file=sys.stderr
        )
        return 3

    violations = [
        result
        for result in results
        if args.fail_above is not None and result.crap > args.fail_above
    ]
    if args.format == "json":
        print_json_report(results, args.fail_above, violations)
    else:
        print_text_report(results, args.fail_above, violations)
    return 1 if violations else 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run solver tests and report function-level CRAP scores."
    )
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        help="Python file or directory to analyze; repeatable (default: src/tycho_nooa)",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--fail-above",
        type=float,
        help="Return exit code 1 when any raw CRAP score is strictly above this value.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed directly to pytest (default: -s).",
    )
    args = parser.parse_args(argv)
    args.source = args.source or [DEFAULT_SOURCE]
    args.pytest_args = args.pytest_args or ["-s"]
    return args


def discover_source_files(sources: Sequence[Path]) -> list[Path]:
    files: set[Path] = set()
    for raw_source in sources:
        source = resolve_from_solver_root(raw_source)
        if source.is_file():
            if source.suffix != ".py":
                raise AnalysisError(f"source file is not Python: {raw_source}")
            files.add(source)
            continue
        if source.is_dir():
            files.update(path.resolve() for path in source.rglob("*.py"))
            continue
        raise AnalysisError(f"source does not exist: {raw_source}")
    if not files:
        raise AnalysisError("no Python source files found")
    return sorted(files)


def measure_coverage(
    sources: Sequence[Path], pytest_args: Sequence[str]
) -> dict[str, object]:
    measurement_roots = sorted(
        {
            (source if source.is_dir() else source.parent).resolve()
            for raw_source in sources
            for source in [resolve_from_solver_root(raw_source)]
        }
    )
    temp_root = Path("/tmp") if Path("/tmp").is_dir() else None
    with tempfile.TemporaryDirectory(prefix="crap-", dir=temp_root) as temp_name:
        temp_dir = Path(temp_name)
        config_path = temp_dir / "coveragerc"
        data_path = temp_dir / ".coverage"
        json_path = temp_dir / "coverage.json"
        config_path.write_text(coverage_config(measurement_roots))

        test_command = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile",
            str(config_path),
            "--data-file",
            str(data_path),
            "-m",
            "pytest",
            *pytest_args,
        ]
        test_run = subprocess.run(
            test_command,
            cwd=SOLVER_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if test_run.returncode:
            raise TestRunFailed(
                test_run.returncode,
                test_run.stdout + test_run.stderr,
            )

        report_run = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "json",
                "--rcfile",
                str(config_path),
                "--data-file",
                str(data_path),
                "-o",
                str(json_path),
            ],
            cwd=SOLVER_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if report_run.returncode:
            raise AnalysisError(
                "coverage json failed:\n" + report_run.stdout + report_run.stderr
            )
        try:
            return json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise AnalysisError(f"could not read Coverage.py JSON: {error}") from error


@dataclass(frozen=True)
class TestRunFailed(Exception):
    returncode: int
    output: str


def coverage_config(source_roots: Sequence[Path]) -> str:
    roots = "\n".join(f"    {root}" for root in source_roots)
    return f"""\
[run]
source_dirs =
{roots}

[report]
skip_empty = true
"""


def calculate_results(
    source_files: Sequence[Path], report: dict[str, object]
) -> list[CrapResult]:
    raw_files = report.get("files")
    if not isinstance(raw_files, dict):
        raise AnalysisError("Coverage.py JSON has no files mapping")
    coverage_files = {
        resolve_from_solver_root(Path(filename)): file_report
        for filename, file_report in raw_files.items()
    }

    results: list[CrapResult] = []
    for source_file in source_files:
        file_report = coverage_files.get(source_file)
        if not isinstance(file_report, dict):
            raise AnalysisError(
                f"Coverage.py did not report {display_path(source_file)}"
            )
        functions = file_report.get("functions")
        if not isinstance(functions, dict):
            raise AnalysisError(
                f"Coverage.py did not report function regions for {display_path(source_file)}"
            )

        collector = FunctionCollector()
        try:
            collector.visit(
                ast.parse(source_file.read_text(), filename=str(source_file))
            )
        except (OSError, SyntaxError) as error:
            raise AnalysisError(
                f"could not parse {display_path(source_file)}: {error}"
            ) from error

        coverage_by_identity = {
            (name, region.get("start_line")): region
            for name, region in functions.items()
            if name and isinstance(region, dict)
        }
        source_identities = {
            (function.name, function.line) for function in collector.functions
        }
        coverage_identities = set(coverage_by_identity)
        if source_identities != coverage_identities:
            missing = sorted(source_identities - coverage_identities)
            unexpected = sorted(coverage_identities - source_identities)
            raise AnalysisError(
                f"function identity mismatch in {display_path(source_file)}; "
                f"missing={missing}, unexpected={unexpected}"
            )

        for function in collector.functions:
            region = coverage_by_identity[(function.name, function.line)]
            summary = region.get("summary")
            if not isinstance(summary, dict):
                raise AnalysisError(
                    f"Coverage.py omitted the summary for {function.name}"
                )
            coverage = summary.get("percent_statements_covered")
            if not isinstance(coverage, int | float):
                raise AnalysisError(
                    f"Coverage.py omitted statement coverage for {function.name}"
                )
            coverage_fraction = coverage / 100
            crap = (
                function.complexity**2 * (1 - coverage_fraction) ** 3
                + function.complexity
            )
            results.append(
                CrapResult(
                    crap=crap,
                    complexity=function.complexity,
                    coverage=float(coverage),
                    file=display_path(source_file),
                    function=function.name,
                    line=function.line,
                )
            )

    return sorted(
        results,
        key=lambda result: (-result.crap, result.file, result.line, result.function),
    )


def print_json_report(
    results: Sequence[CrapResult],
    threshold: float | None,
    violations: Sequence[CrapResult],
) -> None:
    print(
        json.dumps(
            {
                "threshold": threshold,
                "violations": len(violations),
                "functions": [result.as_report_dict() for result in results],
            },
            indent=2,
            sort_keys=True,
        )
    )


def print_text_report(
    results: Sequence[CrapResult],
    threshold: float | None,
    violations: Sequence[CrapResult],
) -> None:
    print("CRAP     CC  COVERAGE  FUNCTION")
    print("-------- --- ---------  -----------------------------------------------")
    for result in results:
        status = " FAIL" if threshold is not None and result.crap > threshold else ""
        print(
            f"{result.crap:8.2f} {result.complexity:3d} {result.coverage:8.2f}%  "
            f"{result.file}:{result.line} {result.function}{status}"
        )
    if threshold is not None:
        print(f"\n{len(violations)} function(s) strictly above CRAP {threshold:g}.")


def resolve_from_solver_root(path: Path) -> Path:
    return (SOLVER_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(SOLVER_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
