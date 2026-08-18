from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "src"


def test_solver_does_not_import_experiments() -> None:
    violations: list[str] = []
    for path in SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(
                name == "experiments" or name.startswith("experiments.")
                for name in names
            ):
                violations.append(f"{path.relative_to(SOURCE)}:{node.lineno}")

    assert violations == []
