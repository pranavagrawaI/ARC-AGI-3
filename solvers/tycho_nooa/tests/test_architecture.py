from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "src"
PACKAGE = SOURCE / "tycho_nooa"
DISPOSITIONS = Path(__file__).parents[1] / "MODULE-DISPOSITION.json"
PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


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


def test_worker_runtime_closure_is_explicit_and_research_free() -> None:
    modules = {path.stem: path for path in PACKAGE.glob("*.py")}
    closure = _transitive_internal_imports(("__init__", "worker"), modules)
    document = json.loads(DISPOSITIONS.read_text())
    groups = document["groups"]
    dispositions = {
        module: disposition
        for disposition, members in groups.items()
        for module in members
    }

    assert document["schema"] == "tycho-nooa-module-disposition-v2"
    assert sum(len(members) for members in groups.values()) == len(dispositions)
    assert set(dispositions) == set(modules)
    assert set(document["runtime_closure"]) == closure
    assert all(
        dispositions[module] in {"kernel", "incumbent", "challenger"}
        for module in closure
    )
    runtime_file_modules = {
        Path(item["path"]).stem
        for item in document["runtime_files"]
        if item["path"].endswith(".py")
    }
    assert all(
        dispositions[module] in {"research", "historical"}
        for module in set(modules) - closure - runtime_file_modules
    )
    assert set(document["group_metadata"]) == set(groups)
    assert all(
        document["group_metadata"][group][field]
        for group in groups
        for field in ("reason", "evidence_owner")
    )


def test_wheel_allowlist_equals_declared_runtime_closure() -> None:
    document = json.loads(DISPOSITIONS.read_text())
    pyproject = tomllib.loads(PYPROJECT.read_text())
    wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]
    runtime_files = document["runtime_files"]
    module_dispositions = {
        module: disposition
        for disposition, members in document["groups"].items()
        for module in members
    }
    expected_modules = {
        f"src/tycho_nooa/{module}.py"
        for module in document["runtime_closure"]
    }
    expected_files = {
        f"src/tycho_nooa/{item['path']}"
        for item in runtime_files
    }

    assert document["schema"] == "tycho-nooa-module-disposition-v2"
    assert len({item["path"] for item in runtime_files}) == len(runtime_files)
    assert all(
        item[field]
        for item in runtime_files
        for field in ("path", "disposition", "reason", "evidence_owner")
    )
    assert all(
        not Path(item["path"]).is_absolute()
        and ".." not in Path(item["path"]).parts
        and (PACKAGE / item["path"]).is_file()
        for item in runtime_files
    )
    assert all(
        item["disposition"] in {"kernel", "incumbent", "challenger"}
        for item in runtime_files
    )
    assert all(
        module_dispositions[Path(item["path"]).stem] == item["disposition"]
        for item in runtime_files
        if item["path"].endswith(".py")
    )
    assert "packages" not in wheel
    assert wheel["sources"] == ["src"]
    assert len(wheel["only-include"]) == len(expected_modules | expected_files)
    assert set(wheel["only-include"]) == expected_modules | expected_files


def _transitive_internal_imports(
    entrypoints: tuple[str, ...], modules: dict[str, Path]
) -> set[str]:
    edges = {module: _internal_imports(path, modules) for module, path in modules.items()}
    closure: set[str] = set()
    pending = list(entrypoints)
    while pending:
        module = pending.pop()
        if module in closure:
            continue
        closure.add(module)
        pending.extend(edges[module])
    return closure


def _internal_imports(path: Path, modules: dict[str, Path]) -> set[str]:
    imported: set[str] = set()
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module == "tycho_nooa":
            names = [f"tycho_nooa.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            if not name.startswith("tycho_nooa."):
                continue
            module = name.split(".", 2)[1]
            if module in modules:
                imported.add(module)
    return imported
