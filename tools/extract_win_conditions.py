"""ORACLE COACH ONLY: inspect privileged source for level-advance predicates.

The output is an oracle label for offline analysis, not an admissible contestant
observation or contestant prior. It must not enter solver prompts, runtime
workspaces, policy development, or contestant performance claims except in an
explicitly oracle-qualified study.

Each ARC game calls next_level() in exactly one place (verified across all 25).
That call site plus its enclosing guard is the game's win condition, so this is
a mechanical route from the source corpus to privileged oracle evidence about
what the games actually ask for. Identifiers are obfuscated; the shape of the
predicate is not.

Usage:  python tools/extract_win_conditions.py [--body]
        --body also unparses the predicate function each guard names.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ENV_DIR = Path(__file__).resolve().parent.parent / "environment_files"


def main(argv: list[str]) -> int:
    print(
        "ORACLE COACH ONLY: privileged win predicates; do not use as contestant input."
    )
    show_body = "--body" in argv
    sources = sorted(ENV_DIR.glob("*/*/*.py"))
    if not sources:
        print(
            f"no games under {ENV_DIR}; run tools/pull_game_sources.py first",
            file=sys.stderr,
        )
        return 2

    for src in sources:
        game = src.parts[-3]
        text = src.read_text()
        tree = ast.parse(text)
        for lineno in call_lines(text, "next_level"):
            guard = innermost_if(tree, lineno)
            test = ast.unparse(guard.test) if guard else "(unconditional)"
            print(f"{game}\t{src.name}:{lineno}\tif {test}")
            if show_body:
                for name in predicate_names(guard):
                    print(indent(source_of(tree, name)))
    return 0


def predicate_names(guard: ast.If | None) -> list[str]:
    """Names of functions the guard calls — the predicates worth unfolding."""
    if guard is None:
        return []
    names = []
    for node in ast.walk(guard.test):
        if isinstance(node, ast.Call):
            fn = node.func
            names.append(
                fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            )
    return [n for n in names if n]


def source_of(tree: ast.Module, name: str) -> str:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    return f"# {name}: not a module-level or method definition"


def innermost_if(tree: ast.Module, lineno: int) -> ast.If | None:
    best = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and node.lineno <= lineno <= (node.end_lineno or 0)
            and (best is None or node.lineno > best.lineno)
        ):
            best = node
    return best


def call_lines(text: str, name: str) -> list[int]:
    return [
        i
        for i, line in enumerate(text.splitlines(), 1)
        if name in line and "def " not in line
    ]


def indent(block: str) -> str:
    return "\n".join("    " + line for line in block.splitlines())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
