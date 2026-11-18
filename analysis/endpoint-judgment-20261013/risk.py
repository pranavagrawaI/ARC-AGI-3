"""Small explicit CRAP measurement: AST decisions and traced executable lines."""
import ast
import json
from pathlib import Path
import runpy
import subprocess
import sys
import trace

import screen


def trace_file(path, args):
    tracer = trace.Trace(count=True, trace=False, ignoredirs=[sys.prefix, str(screen.SOURCE), str(screen.DEFAULT_ENV)])
    old = sys.argv
    sys.argv = [str(path), *args]
    try:
        tracer.runfunc(runpy.run_path, str(path), run_name="__main__")
    except SystemExit as e:
        if e.code:
            raise
    finally:
        sys.argv = old
    return [{"file": filename, "line": line, "count": count} for (filename, line), count in tracer.results().counts.items()
            if filename in {str(screen.HERE / "screen.py"), str(screen.HERE / "worker.py")}]


def measure():
    files = [screen.HERE / "screen.py", screen.HERE / "worker.py"]
    observed = trace_file(screen.HERE / "test_screen.py", [])
    root = screen.HERE / "evidence/risk-worker"
    root.mkdir(exist_ok=True)
    cfg = screen.config(screen.parser().parse_args(["--dry", "--keys", "tn36-L1", "bp35-L1", "--token-budget", "4096"]))
    cfg.pop("output")
    screen.atomic(root / "config.json", cfg)
    for key in cfg["selected"]:
        attempt = root / key
        attempt.mkdir(exist_ok=True)
        env = screen.child_env(attempt, cfg)
        with (attempt / "worker.log").open("w") as f:
            subprocess.run([str(screen.PYTHON), "-B", str(Path(__file__).resolve()), "--trace-worker", str(root / "config.json"), key],
                           env=env, cwd=attempt, stdout=f, stderr=subprocess.STDOUT, check=True)
        observed += screen.load(attempt / "coverage.json")
    counts = {(r["file"], r["line"]): r["count"] for r in observed}
    rows = []
    for p in files:
        tree = ast.parse(p.read_text())
        executable = {node.lineno for node in ast.walk(tree) if isinstance(node, ast.stmt)}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            descendants = list(ast.walk(node))
            # Explicit approximation: nested-function statements stay in their
            # lexical parent; no repository CRAP gate existed for these files.
            cc = 1 + sum(isinstance(n, (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler)) for n in descendants)
            cc += sum(len(n.values) - 1 for n in descendants if isinstance(n, ast.BoolOp))
            cc += sum(len(n.generators) + sum(len(g.ifs) for g in n.generators) for n in descendants if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)))
            lines = {line for line in executable if node.lineno <= line <= node.end_lineno}
            covered = {line for line in lines if counts.get((str(p), line), 0)}
            coverage = len(covered) / len(lines) if lines else 0
            rows.append({"file": p.name, "function": node.name, "line": node.lineno, "cc": cc,
                         "line_coverage": coverage, "crap": cc * cc * (1 - coverage) ** 3 + cc})
    result = {"baseline_functions": 0, "functions": len(rows), "threshold": None, "violations": None,
              "maximum_crap": max(r["crap"] for r in rows), "rows": sorted(rows, key=lambda r: -r["crap"]),
              "contract": "AST decision count above; executable statement-line coverage from trace of offline tests; no gate imposed",
              "limitation": "Statement-line coverage, not branch coverage. Nested functions contribute to their lexical parents under this explicit contract. No numerical gate or global maintainability claim."}
    screen.atomic(screen.HERE / "evidence/risk.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--trace-worker":
        config_path, key = Path(sys.argv[2]), sys.argv[3]
        observed = trace_file(screen.HERE / "worker.py", [str(config_path), key])
        screen.atomic(config_path.parent / key / "coverage.json", observed)
    else:
        measure()
