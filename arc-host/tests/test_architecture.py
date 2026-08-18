from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "src"


def test_host_imports_neither_solvers_nor_experiments() -> None:
    forbidden = ("solvers", "experiments", "nooa", "tycho")
    violations: list[str] = []
    for path in SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(
                name == root or name.startswith(f"{root}.")
                for name in names
                for root in forbidden
            ):
                violations.append(f"{path.relative_to(SOURCE)}:{node.lineno}")

    assert violations == []
