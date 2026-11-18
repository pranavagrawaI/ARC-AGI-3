"""Parse the frozen notebook's exact launch arguments, without launching anything."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
notebook = json.loads((HERE/'source/submission.ipynb').read_text())
tree = ast.parse(''.join(notebook['cells'][-1]['source']))
node = next(n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'command' for t in n.targets))
command = eval(compile(ast.Expression(node.value), '<frozen-command>', 'eval'),
               {'sys': sys, 'output': Path('/kaggle/working/qualification')})
spec = importlib.util.spec_from_file_location('frozen_outer', HERE/'runtime-source/exact_request_outer_runtime.py')
outer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(outer)
results = []
for label, startup in [('frozen', '600'), ('single_argument_counterfactual', '1200')]:
    arguments = command[1:].copy()
    arguments[arguments.index('--startup-timeout-s')+1] = startup
    sys.argv = arguments
    error = io.StringIO()
    with contextlib.redirect_stderr(error):
        try:
            outer.parse_args()
            code = 0
        except SystemExit as exc:
            code = exc.code
    results.append(dict(case=label, startup_seconds=startup, parser_exit=code,
                        stderr=error.getvalue()))
assert [r['parser_exit'] for r in results] == [2, 0]
print(json.dumps(dict(scope='argument parsing only; no runtime/model/game execution',
                      command=command, results=results), indent=2))
