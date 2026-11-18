"""Dry-run the SGLang reference notebook locally: its own setup and REF cells against mock_server.py.

  python3 -B dryrun.py

The same substitutions as ../../mtp-depth-20261006/dryrun_notebook.py: /kaggle is moved under a temp
root holding a hard-linked copy of the replay dataset; the launcher cell is replaced by stub
args/env/precache_model_thread, where args[0] is a shim that starts mock_server.py; the time scale is 20.
PASS needs all three blocks valid with zero errors, and the KEEP0 block's request bodies must carry at
most 4 images. That is checked by wrapping the mock's view: the shim server logs image counts per
request.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MTP = HERE.parents[1] / "mtp-depth-20261006"
NB = HERE / "kernel" / "sglang-keep0-ref-20261020.ipynb"
SCALE = 20
PORT = 18011


def main():
    root = Path(tempfile.mkdtemp(prefix="sglref-dryrun-"))
    ds = root / "kaggle/input/datasets/pranavagrawai/mtp-depth-replay-data-20261006"
    ds.parent.mkdir(parents=True)
    subprocess.run(["cp", "-a", str(MTP / "kaggle/dataset"), str(ds)], check=True)
    imglog = root / "images.log"
    # wrap the mock: count image parts per request body before handing it to mock_server's handler
    wrapper = root / "mock_wrap.py"
    wrapper.write_text(f"""import json, runpy, sys
sys.path.insert(0, {str(MTP / 'kaggle/dataset')!r})
import http.server
_orig = http.server.BaseHTTPRequestHandler.__init__
src = open({str(MTP / 'kaggle/dataset/mock_server.py')!r}).read()
src = src.replace('body = json.loads(self.rfile.read(n))', '''body = json.loads(self.rfile.read(n))
        open({str(imglog)!r}, "a").write(str(sum(1 for m in body.get("messages", []) if isinstance(m.get("content"), list) for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url")) + "\\\\n")''')
assert 'images.log' in src
sys.argv = ['mock_server.py'] + sys.argv[1:]
exec(compile(src, 'mock_server.py', 'exec'), {{'__name__': '__main__'}})
""")
    shim = root / "fake-sglang"
    shim.write_text(f"""#!{sys.executable}
import os, sys
a = sys.argv[1:]
port = a[a.index('--port') + 1]
os.execv(sys.executable, [sys.executable, '-B', {str(wrapper)!r}, port, '--speed', '{SCALE}', '--load-seconds', '2'])
""")
    shim.chmod(0o755)
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code"]
    setup, _precache, _launcher, ref = cells
    fix = lambda s: s.replace("'/kaggle/", f"'{root}/kaggle/").replace("Path('/kaggle", f"Path('{root}/kaggle")
    setup = fix(setup).replace("subprocess.run(['nvidia-smi'", "print(['nvidia-smi'")
    setup = setup.replace("SERVED_MODEL_PORT = 8001", f"SERVED_MODEL_PORT = {PORT}")
    setup = setup.replace("time.monotonic() + 14400", f"time.monotonic() + {14400 / SCALE}")
    for a, b in (("slots=10,", f"slots=10, time_scale={SCALE},"), ("< 20 * 60:", f"< 20 * 60 / {SCALE}:"),
                 ("time.monotonic() - t < 900", f"time.monotonic() - t < 900 / {SCALE}")):
        assert ref.count(a) == 1, a
        ref = ref.replace(a, b)
    ref = fix(ref)
    g = {"__name__": "__main__"}
    exec(compile(setup, "setup-cell", "exec"), g)
    g["args"] = [str(shim), "serve", "--port", str(PORT)]
    g["env"] = dict(os.environ)
    g["precache_model_thread"] = threading.Thread(target=lambda: None)
    t0 = time.time()
    exec(compile(ref, "ref-cell", "exec"), g)
    res = g["RESULT"]
    blocks = {b["label"]: b for b in res["blocks"]}
    counts = [int(x) for x in imglog.read_text().split()] if imglog.exists() else []
    print(f"dry run {time.time() - t0:.0f}s; blocks", {k: {kk: v.get(kk) for kk in ("valid", "U_overlap", "errors")} for k, v in blocks.items()})
    print("requests seen", len(counts), "max images/request", max(counts) if counts else None,
          "keep0/full", res.get("keep0_over_full_U_overlap"))
    ok = (all(blocks.get(k, {}).get("valid") and blocks[k]["errors"] == 0 for k in ("FULL_a", "KEEP0", "FULL_b"))
          and counts and max(counts) > 4 and res.get("keep0_over_full_U_overlap") is not None)
    print("DRYRUN", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
