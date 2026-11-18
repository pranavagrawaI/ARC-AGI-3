"""Offline check of a built session notebook: the ORACLE cells compile, the overlay cell runs as written (writes and
imports overlay.py and the attempt list), and install() builds one StartGame per attempt that lands on its level.

  ONLY_RESET_LEVELS=true PYTHONPATH=<dfranzen tufa src>:<dfranzen ARC3-Inference> solvers/tycho_nooa/.venv/bin/python -B check_notebook.py a0
"""
import ast, json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
session = sys.argv[1]
ident = json.loads((HERE / "sessions" / session / "identity.json").read_text())
nb = json.loads((HERE / "sessions" / session / "kernel" / f"{ident['slug']}.ipynb").read_text())
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
oracle = [s for s in cells if s.startswith("# ORACLE-HINT")]
run = [s for s in cells if "oracle_overlay.install(bm, ORACLE_ATTEMPTS" in s]
assert len(oracle) == 3 and len(run) == 1
for s in oracle + run:
    compile(s, "cell", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
with tempfile.TemporaryDirectory() as d:
    ns = {"WORKING_DIR": Path(d)}
    exec(next(s for s in oracle if s.startswith("# ORACLE-HINT overlay")), ns)
    assert (Path(d) / "oracle/overlay.py").read_text() == (HERE / "overlay.py").read_text()
    attempts = ns["ORACLE_ATTEMPTS"]
    bm = SimpleNamespace(games=None)
    games = ns["oracle_overlay"].install(bm, attempts, Path.home() / "projects/arc-agi-3-reference/competition-env/environment_files")
    landed = 0
    for g in games:
        st = g.start_game()
        assert st.levels_completed == g.start_level, g.external_game_id
        landed += 1
    names = [a["name"] for a in attempts]
    assert len(set(names)) == len(names)
    ev = [json.loads(l) for l in (Path(d) / "oracle/events.jsonl").read_text().splitlines()]
    assert sum(e["kind"] == "landed" for e in ev) == len(attempts)
print(f"OK {session}: {len(attempts)} attempts, {landed} landed via the notebook's own overlay cell; arms "
      f"{sorted({a['arm'] for a in attempts})}; hinted {sum(bool(a['hint']) for a in attempts)}")
