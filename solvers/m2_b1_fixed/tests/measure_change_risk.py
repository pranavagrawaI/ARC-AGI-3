"""Bounded statement-line CRAP estimate using stdlib trace and a fixed gate.

No repository CRAP gate exists for this harness, and its interpreter lacks
coverage/radon. This local estimate is not the Tycho-NOOA quality gate.
CC counts if/for/while/except/conditional-expression/comprehension branches
and boolean short-circuit operands. Coverage counts non-docstring statement
start lines hit by the unchanged gate_b1.py, before and after the repair.
"""
import ast
import json
from pathlib import Path
import runpy
import sys
import trace

root = Path(__file__).resolve().parents[1]
path = root / 'src/inference/agent/tool_agent.py'
selected = {'_persistent_history_messages', '_persistent_history_messages_pinned',
            '_trim_messages_for_context', '_b1_count',
            '_force_reduce_messages', '_force_reduce_messages_pinned'}
runner = trace.Trace(count=True, trace=False)
runner.runfunc(runpy.run_path, str(root / 'tests/gate_b1.py'), run_name='__main__')
counts = runner.results().counts
rows = []
for node in ast.walk(ast.parse(path.read_text())):
    if not isinstance(node, ast.FunctionDef) or node.name not in selected:
        continue
    nodes = list(ast.walk(node))
    statements = {n.lineno for n in nodes if isinstance(n, ast.stmt)
                  and n is not node and not (isinstance(n, ast.Expr)
                  and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))}
    branches = sum(isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.IfExp)) for n in nodes)
    branches += sum(len(n.values) - 1 for n in nodes if isinstance(n, ast.BoolOp))
    branches += sum(1 + len(n.ifs) for n in nodes if isinstance(n, ast.comprehension))
    cc = 1 + branches
    covered = sum(bool(counts.get((str(path), line))) for line in statements)
    coverage = covered / len(statements) if statements else 1.0
    rows.append({'function': node.name, 'cc': cc, 'statements': len(statements),
                 'covered': covered, 'coverage': coverage,
                 'crap': cc**2 * (1-coverage)**3 + cc})
report = {'measurement': __doc__, 'threshold': None, 'functions': rows,
          'maximum': max(r['crap'] for r in rows)}
out = root / 'evidence' / sys.argv[1]
out.write_text(json.dumps(report, indent=2)+'\n')
print('Risk estimate:', out)
