"""Explicit stdlib-trace statement-line estimate; no repository gate/threshold.

Fixed universe: test_bounded_search.py, including its 200 independent graphs.
CC counts branch nodes, boolean short-circuit operands and comprehensions.
For the public function, nested callback helpers are included in its estimate;
their individual estimates are also reported. This is not Coverage.py/Radon.
"""
import ast
import importlib.util
import json
from pathlib import Path
import trace
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/inference/utils/bounded_search.py'
spec=importlib.util.spec_from_file_location('search_risk_tests',ROOT/'tests/test_bounded_search.py')
tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
tracer=trace.Trace(count=True,trace=False)
result=tracer.runfunc(unittest.TextTestRunner(verbosity=1).run,suite)
assert result.wasSuccessful()
counts=tracer.results().counts
rows=[]
class Visitor(ast.NodeVisitor):
    def __init__(self):self.scope=[]
    def visit_ClassDef(self,node):
        self.scope.append(node.name);self.generic_visit(node);self.scope.pop()
    def visit_FunctionDef(self,node):
        nodes=list(ast.walk(node))
        statements={n.lineno for n in nodes if isinstance(n,ast.stmt) and n is not node
                    and not(isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))}
        cc=1+sum(isinstance(n,(ast.If,ast.For,ast.While,ast.ExceptHandler,ast.IfExp)) for n in nodes)
        cc+=sum(len(n.values)-1 for n in nodes if isinstance(n,ast.BoolOp))
        cc+=sum(1+len(n.ifs) for n in nodes if isinstance(n,ast.comprehension))
        covered=sum(bool(counts.get((str(SOURCE),line))) for line in statements)
        coverage=covered/len(statements) if statements else 1
        rows.append({'function':'.'.join([*self.scope,node.name]),'cc':cc,'statements':len(statements),
                     'covered':covered,'coverage':coverage,'crap':cc**2*(1-coverage)**3+cc})
        self.scope.append(node.name);self.generic_visit(node);self.scope.pop()
Visitor().visit(ast.parse(SOURCE.read_text()))
(ROOT/'evidence/search-risk.json').write_text(json.dumps({'contract':__doc__,'threshold':None,
    'baseline':'New utility; no preexisting function baseline','functions':rows,'maximum':max(x['crap'] for x in rows)},indent=2)+'\n')
print('Search risk estimate recorded')
