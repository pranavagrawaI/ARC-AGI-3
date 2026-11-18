"""Dry-run smoke.py end to end on the laptop, with mock_server.py standing in for Flare.

  python3 -B dryrun.py

Only these substitutions: /kaggle/{input,working} move under a temp root; the probe bundle is a fake
whose `ld-linux-x86-64.so.2` execs its program and whose `flare` starts the replay's mock server on
the requested port (it accepts every Flare flag); the package is a sparse file of the real size; the
replay runs at time_scale 20. This exercises input discovery, the start/retry path, the six functional
rows, the S=10/14/16 replay, the estimators and result.json. A second pass with --fail-prefix
rehearses the no-prefix-cache fallback.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "mtp-depth-20261006" / "kaggle" / "dataset"
TOTAL = 107011338717

LD = """#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
assert args[0] == '--library-path', args
os.execv(sys.executable, [sys.executable, args[2]] + args[3:])
"""
FLARE = """import os, sys
args = sys.argv[1:]
fail_prefix = {fail_prefix}
if args[:1] == ['--version']:
    print('flare 0.1.0 (fake)'); sys.exit(0)
assert args[0] == 'run' and os.path.getsize(args[1]) == {total}, args
if fail_prefix and '--prefix-cache-memory' in args:
    print('fake: prefix cache rejected', flush=True); sys.exit(4)
port = args[args.index('--port') + 1]
sys.path.insert(0, {data!r})
os.execv(sys.executable, [sys.executable, {data!r} + '/mock_server.py', port, '--speed', '20', '--load-seconds', '3'])
"""


def run(fail_prefix):
    root = Path(tempfile.mkdtemp(prefix="smoke3-dryrun-"))
    inp, work = root / "input", root / "working"
    bundle = inp / "flare-probe-bundle-20261018"
    pkgdir = inp / "datasets" / "pranavagrawai" / "flare-gb202-package-20261020"
    for d in (bundle, pkgdir, work):
        d.mkdir(parents=True)
    os.symlink(DATA, inp / "mtp-depth-replay-data-20261006")
    files = {"ld-linux-x86-64.so.2": LD, "flare": FLARE.format(fail_prefix=fail_prefix, total=TOTAL, data=str(DATA)),
             "libc.so.6": "fake"}
    for name, text in files.items():
        (bundle / name).write_text(text)
    (bundle / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(t.encode()).hexdigest()}  {n}\n" for n, t in files.items()))
    with open(pkgdir / "qwen3.8-flash-next-tp1-c16-mtp5.safetensors", "wb") as f:
        f.truncate(TOTAL)
    code = (HERE / "smoke.py").read_text()
    subs = [("/kaggle/input", str(inp)), ("/kaggle/working", str(work)),
            ("slots=S, header=", "slots=S, time_scale=20, header="),
            ("Path('/tmp/flare')", f"Path('{root}/tmpflare')"), ("Path('/tmp/flare-state')", f"Path('{root}/state')")]
    for a, b in subs:
        assert a in code, a
        code = code.replace(a, b)
    p = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=3600)
    res = json.loads((work / "flare" / "result.json").read_text())
    print(f"--- fail_prefix={fail_prefix} rc={p.returncode} root={root}")
    print(p.stdout[-2500:])
    print(p.stderr[-2500:])
    att = res["start"]["attempts"]
    rows = res.get("functional", {}).get("rows", [])
    rep = res.get("replay", {})
    ok = (p.returncode == 0 and att[-1]["ok"] and len(rows) == 6 and all(r.get("status") == 200 for r in rows)
          and all(rep.get(str(S), rep.get(S, {})).get("valid") for S in (10, 14, 16))
          and "ratio_vs_sglang_m3a" in res and (att[-1]["prefix_cache"] is None) == fail_prefix)
    print("attempts", att)
    print("S10", {k: rep.get("10", rep.get(10, {})).get(k) for k in ("U_resp", "U_overlap", "errors")})
    print("DRYRUN", "PASS" if ok else "FAIL", f"fail_prefix={fail_prefix}")
    return ok


if __name__ == "__main__":
    results = [run(False), run(True)]
    sys.exit(0 if all(results) else 1)
