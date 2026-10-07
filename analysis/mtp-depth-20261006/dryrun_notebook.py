"""Dry-run the generated notebook's setup + sweep cells locally (pre-launch gate item 2).

python3 -B dryrun_notebook.py

Runs the notebook's own cell code with three substitutions only: /kaggle is prefixed with a
temp root holding a hard-linked copy of the dataset; the launcher cell (which needs the GPU
wheelhouse) is replaced by stub `args`/`env`/`precache_model_thread`, where args[0] is a shim
that starts mock_server.py; run_sweep gets time_scale=20. Exercises dataset discovery and
readback hashing, flag substitution, factory start/stop, sweep, verdict.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE / "kaggle/notebook/mtp-depth-sweep-20261006.ipynb"
SCALE = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=18001)
    opts = ap.parse_args()
    root = Path(tempfile.mkdtemp(prefix="dryrun-", dir=HERE / "kaggle"))
    ds = root / "kaggle/input/datasets/pranavagrawai/mtp-depth-replay-data-20261006"
    ds.parent.mkdir(parents=True)
    subprocess.run(["cp", "-al", str(HERE / "kaggle/dataset"), str(ds)], check=True)
    shim = root / "fake-sglang"
    shim.write_text(f"""#!{sys.executable}
import os, sys
a = sys.argv[1:]
port = a[a.index('--port') + 1]
steps = a[a.index('--speculative-num-steps') + 1]
assert a[a.index('--speculative-num-draft-tokens') + 1] == str(int(steps) + 1)
print('fake sglang: steps', steps, flush=True)
os.chdir({str(HERE)!r})
os.execv(sys.executable, [sys.executable, '-B', 'mock_server.py', port, '--speed', '{SCALE}', '--load-seconds', '2'])
""")
    shim.chmod(0o755)
    cells = [("".join(c["source"])) for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code"]
    setup, _precache, _launcher, sweep_cell = cells
    fix = lambda s: s.replace("'/kaggle/", f"'{root}/kaggle/").replace("Path('/kaggle", f"Path('{root}/kaggle")
    setup = fix(setup).replace("subprocess.run(['nvidia-smi'", "print(['nvidia-smi'")
    setup = setup.replace('SERVED_MODEL_PORT = 8001', f'SERVED_MODEL_PORT = {opts.port}')
    setup = setup.replace('time.monotonic() + 14400', f'time.monotonic() + {14400 / SCALE}')
    sweep_cell = sweep_cell.replace("subprocess.run(['nvidia-smi', '--query-gpu=memory.used'",
                                    "subprocess.run(['echo', '0'] or ['nvidia-smi', '--query-gpu=memory.used'")
    anchor = 'provenance=PROVENANCE)'
    assert sweep_cell.count(anchor) == 1, 'notebook call changed: cannot apply mock time scale'
    sweep_cell = fix(sweep_cell).replace(anchor, f'provenance=PROVENANCE, time_scale={SCALE})')
    g = {"__name__": "__main__"}
    exec(compile(setup, "setup-cell", "exec"), g)
    g["args"] = [str(shim), "serve", "--port", str(opts.port), "--speculative-algorithm", "NEXTN",
                 "--speculative-num-steps", "3", "--speculative-eagle-topk", "1",
                 "--speculative-num-draft-tokens", "4"]
    g["env"] = dict(os.environ)
    g["precache_model_thread"] = threading.Thread(target=lambda: None)
    t0 = time.time()
    exec(compile(sweep_cell, "sweep-cell", "exec"), g)
    out = g["OUT"]
    print(f"dry run finished in {time.time() - t0:.0f}s; outputs in {out}")
    for lbl in ("M3A", "M1", "M2", "M3C", "M4", "M3B"):
        a = json.loads((out / lbl / "server-args.json").read_text())
        print(lbl, "spec steps", a[a.index("--speculative-num-steps") + 1],
              "draft tokens", a[a.index("--speculative-num-draft-tokens") + 1])


if __name__ == "__main__":
    main()
