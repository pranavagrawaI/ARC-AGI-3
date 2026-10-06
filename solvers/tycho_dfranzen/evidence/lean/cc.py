"""Stdlib cyclomatic complexity for named functions (no coverage tool is installed).

CC = 1 + if/for/while/except/ifexp/assert/comprehension-if/match-case + (boolean operands - 1).
Usage: python cc.py file.py[:qualname] ...   (no qualname = every function in the file)
"""
import ast
import json
import sys


def complexity(node):
    score = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler,
                              ast.IfExp, ast.Assert, ast.match_case)):
            score += 1
        elif isinstance(child, ast.BoolOp):
            score += len(child.values) - 1
        elif isinstance(child, ast.comprehension):
            score += len(child.ifs)
    return score


def functions(tree, prefix=""):
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield prefix + node.name, node
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from functions(node, prefix + node.name + ".")


rows = {}
for spec in sys.argv[1:]:
    path, _, wanted = spec.partition(":")
    tree = ast.parse(open(path).read())
    for name, node in functions(tree):
        if not wanted or name == wanted:
            rows[f"{path}:{name}"] = complexity(node)
print(json.dumps(rows, indent=1))
