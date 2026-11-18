"""Dry-run smoke3b end to end on the laptop: both read routes, the real flare_ra.so, a mock server.

  python3 -B dryrun.py

Substitutions only: /kaggle/{input,working} move under a temp root; the 107 GB dataset file and the six
parts are sparse files of the real sizes, each part carrying an 8-byte marker at its start; the bundle's
`ld-linux` honours --library-path/--preload by exec'ing the fake `flare` under LD_PRELOAD; the replay
runs at time_scale 20. The fake `flare` opens its package with O_DIRECT and reads 4 KiB at every part
boundary. Under route B the package is the zero-filled placeholder, so the markers come back only if
flare_ra.so served them from the parts; otherwise it exits 5 and the run fails. Pass 1 forces route B,
pass 2 forces route A.
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
SPAN = 17835223120

LD = """#!/usr/bin/env python3
import os, sys
args, env = sys.argv[1:], dict(os.environ)
while args and args[0] in ('--library-path', '--preload'):
    if args[0] == '--preload':
        env['LD_PRELOAD'] = args[1]
    args = args[2:]
os.execve(sys.executable, [sys.executable, args[0]] + args[1:], env)
"""
FLARE = """import os, sys
args = sys.argv[1:]
if args[:1] == ['--version']:
    print('flare 0.1.0 (fake)'); sys.exit(0)
pkg = args[1]
assert args[0] == 'run' and os.path.getsize(pkg) == {total}, args
fd = os.open(pkg, os.O_RDONLY | os.O_DIRECT)
for i, start in enumerate({starts}):
    got = os.pread(fd, 4096, start)[:8]
    if got != b'PART%04d' % i:
        print('fake flare: wrong bytes at part', i, got, flush=True); sys.exit(5)
print('fake flare: all part markers read through', pkg, flush=True)
port = args[args.index('--port') + 1]
sys.path.insert(0, {data!r})
env = dict(os.environ); env.pop('LD_PRELOAD', None)
os.execve(sys.executable, [sys.executable, {data!r} + '/mock_server.py', port, '--speed', '20', '--load-seconds', '3'], env)
"""


def run(force_route):
    root = Path(tempfile.mkdtemp(prefix="smoke3c-dryrun-"))
    inp, work = root / "input", root / "working"
    bundle = inp / "datasets" / "pranavagrawai" / "flare-probe-bundle-20261018"
    dset = inp / "datasets" / "pranavagrawai" / "flare-gb202-package-20261020"
    for d in (bundle, dset, work):
        d.mkdir(parents=True)
    os.symlink(DATA, inp / "mtp-depth-replay-data-20261006")
    starts = [i * SPAN for i in range(6)]
    for i, s0 in enumerate(starts):
        e0 = min(TOTAL, s0 + SPAN)
        pd = inp / "notebooks" / f"part{i}"
        pd.mkdir(parents=True)
        with open(pd / f"flare-part-{i}", "wb") as f:
            f.write(b"PART%04d" % i)
            f.truncate(e0 - s0)
        (pd / f"flare-part-{i}.json").write_text(json.dumps({"index": i, "start": s0, "end": e0, "bytes": e0 - s0}))
    with open(dset / "qwen3.8-flash-next-tp1-c16-mtp5.safetensors", "wb") as f:  # route A: real markers too
        for s0 in starts:
            f.seek(s0); f.write(b"PART%04d" % starts.index(s0))
        f.truncate(TOTAL)
    files = {"ld-linux-x86-64.so.2": LD, "flare": FLARE.format(total=TOTAL, starts=starts, data=str(DATA)),
             "libc.so.6": "fake"}
    for name, text in files.items():
        (bundle / name).write_text(text)
    (bundle / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(t.encode()).hexdigest()}  {n}\n" for n, t in files.items()))
    code = json.loads((HERE / "kernel" / "flare-smoke-3c-20261020.ipynb").read_text())["cells"][1]["source"]
    code = "".join(code)
    subs = [("/kaggle/input", str(inp)), ("/kaggle/working", str(work)),
            ("slots=S, header=", "slots=S, time_scale=20, header="),
            ("Path('/tmp/flare')", f"Path('{root}/tmpflare')"), ("Path('/tmp/flare-state')", f"Path('{root}/state')"),
            ("Path('/tmp/flare-pkg/", f"Path('{root}/flare-pkg/"),
            ("ROUTE = 'B' if bench['B_staged_parts'] >= bench['A_dataset_gcsfuse'] else 'A'", f"ROUTE = '{force_route}'")]
    for a, b in subs:
        assert a in code, a
        code = code.replace(a, b)
    script = root / "smoke_dry.py"
    script.write_text(code)
    p = subprocess.run([sys.executable, str(script)], cwd=root, capture_output=True, text=True, timeout=3600)
    res = json.loads((work / "flare" / "result.json").read_text())
    print(f"--- route={force_route} rc={p.returncode} root={root}")
    print(p.stdout[-1800:])
    print(p.stderr[-1500:])
    att = res["start"]["attempts"]
    rows = res.get("functional", {}).get("rows", [])
    rep = res.get("replay", {})
    server = (work / "flare" / "flare-server.log").read_text()
    ra = res["start"].get("ra_last")
    ok = (p.returncode == 0 and att[-1]["ok"] and att[-1]["prefix_cache"] == "64G" and len(rows) == 6
          and all(r.get("status") == 200 for r in rows)
          and all(rep.get(str(S), rep.get(S, {})).get("valid") for S in (10, 14, 16))
          and "all part markers read through" in server and res["route"]["chosen"] == force_route
          and (("--verify-weights" in server) == (force_route == "B")) and isinstance(ra, dict) and ra["opens"] >= 1)
    print("route", res["route"], "ra_last", ra)
    print("DRYRUN", "PASS" if ok else "FAIL", f"route={force_route}")
    return ok


if __name__ == "__main__":
    results = [run("B"), run("A")]
    sys.exit(0 if all(results) else 1)
